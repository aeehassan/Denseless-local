# ══════════════════════════════════════════════════════════════════════════════
# app/agent/rag/chains/notes_chain.py
#
# Notes generation chain.
#
# Responsibility:
#   Takes a topic_map from the retriever, generates personalised condensed
#   notes for each section using the student's learner profile attributes,
#   constructs a PDF from the result, re-ingests the PDF into Chroma so
#   the qa_chain has access to both original and condensed material, and
#   returns the PDF file path to the router.
#
# Page limit:
#   Designed for PDFs up to 50 pages in production.
#   During testing, the limit is enforced at 20 pages.
#   Both Gemini 2.5 Pro and Llama3.2:3b / Gemma4 can accommodate more,
#   but these limits are enforced as practical UX boundaries — beyond
#   them, condensed note coherence and section continuity degrade.
#
# Chain flow (runs once per student request):
#   Step 0 → Topic relatedness pre-call (temp=0.0, strict JSON)
#   Step 1 → Sanitise section headings programmatically
#   Step 2 → Section-by-section LLM condensation calls
#   Step 3 → PDF construction from notes_json (reportlab)
#   Step 4 → Post-processing: PDF re-ingested into Chroma
#   Step 5 → PDF file path returned to router
#
# Token guard:
#   @token_guard wraps run_notes_chain() — student_id must be first arg.
#   Every LLM call inside the chain contributes to token deduction because
#   token_guard reads usage_metadata from the final returned response.
#   NOTE: For multi-call chains like this one, total token usage is tracked
#   via a running accumulator and returned as a synthetic usage_metadata
#   object so token_guard can deduct correctly.
#
# Usage:
#   from app.agent.rag.chains.notes_chain import run_notes_chain
#
#   pdf_path = run_notes_chain(
#       student_id    = "abc123",
#       topic_map     = topic_map,
#       current_topic = "Computer Architecture",
#       weak_topics   = ["Memory Management"],
#       strong_topics = ["Boolean Algebra"],
#       learning_pace = "average",
#       llm           = llm,
#       embedder      = embedder,
#       store         = store,
#   )
# ══════════════════════════════════════════════════════════════════════════════

import json
import tempfile
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from json_repair import repair_json

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import colors

from app.agent.rag.prompts import (
    TOPIC_RELATEDNESS_PROMPT,
    NOTES_PROMPT_SLOW,
    NOTES_PROMPT_AVERAGE,
    NOTES_PROMPT_FAST,
    ELABORATION_WEAK,
    ELABORATION_STRONG,
    ELABORATION_NONE,
    RENAME_INSTRUCTION,
    RENAME_INSTRUCTION_NONE,
)
from app.agent.rag.ingestion.data_ingestion import process_and_load_file
from app.agent.rag.ingestion.embeddings import generate_embeddings
from app.agent.rag.ingestion.vector_store import add_documents_to_chroma
from app.agent.rag.chains.qa_chain import view_ltm
from app.services.token_service import token_guard

# Enable httpx request logging so Ollama API calls are visible in output
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Set USE_GEMINI = True when hitting the Gemini API (rate limits apply).
# Set USE_GEMINI = False for local Ollama (no delays, no rate-limit handling).
USE_GEMINI: bool = False

# Seconds to sleep between section LLM calls when USE_GEMINI is True.
# 5 s gap → max 12 RPM, comfortably under the 15 RPM free-tier ceiling.
_REQUEST_DELAY_SECONDS: int = 5

# Backoff schedule (seconds) for RPM errors.
# 60 s guarantees the per-minute window resets before the final retry.
_RPM_BACKOFF_SECONDS: list[int] = [15, 30, 60]

# Max outer (rate-limit) retries per section.
# Mirrors len(_RPM_BACKOFF_SECONDS) — change both together.
_MAX_RATE_LIMIT_RETRIES: int = 3

# Maximum heading length before flagging for LLM renaming
_MAX_HEADING_LENGTH = 65

# Maximum number of retries when LLM returns empty content
_MAX_EMPTY_RETRIES = 2

# Maximum number of retry attempts when LLM returns unparseable output
_MAX_SECTION_RETRIES = 3

# Prompt map keyed by learning pace
_NOTES_PROMPT_MAP = {
    "slow": NOTES_PROMPT_SLOW,
    "average": NOTES_PROMPT_AVERAGE,
    "fast": NOTES_PROMPT_FAST,
}

# Output directory for generated PDFs
_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "notes"

# ══════════════════════════════════════════════════════════════════════════════
# NEW: Custom exception
# ══════════════════════════════════════════════════════════════════════════════


class RateLimitExhaustedError(RuntimeError):
    """
    Raised when API rate limiting cannot be recovered from.

    kind:
        "rpm" — per-minute limit hit but all backoff retries exhausted.
                 Unlikely in practice (60 s backoff resets the window),
                 but included for safety.
        "rpd" — daily quota exhausted. No backoff can recover from this.
                 Caller should terminate the section loop and build a
                 partial PDF from whatever has been collected so far.

    Example:
        raise RateLimitExhaustedError(
            "Daily Gemini quota exhausted after section 3.", kind="rpd"
        )
    """

    def __init__(self, message: str, kind: str):
        super().__init__(message)
        self.kind = kind  # "rpm" | "rpd"


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _check_topic_relatedness(
    llm: BaseChatModel,
    current_topic: str,
    weak_topics: List[str],
    strong_topics: List[str],
) -> dict:
    """
    Pre-call to determine if the current topic semantically builds upon
    any of the student's weak or strong topics.

    Uses a strict temperature (0.0) for a deterministic classification
    result — this is a judgement call, not a creative generation task.
    The LLM receives the full context of weak and strong topics and
    reasons about prerequisite relationships.

    Args:
        llm:           The initialised LLM instance (temperature must be
                       set to 0.0 before this call — see run_notes_chain).
        current_topic: The topic the student is currently studying.
        weak_topics:   Topics the student has previously struggled with.
        strong_topics: Topics the student has a strong grasp of.

    Returns:
        Dict with the relatedness judgement. Examples:

        When current topic builds on a weak prerequisite:
        {
            "is_buildup":    True,
            "related_topic": "Memory Management",
            "relation":      "weak"
        }

        When current topic builds on a strong prerequisite:
        {
            "is_buildup":    True,
            "related_topic": "Boolean Algebra",
            "relation":      "strong"
        }

        When no meaningful prerequisite relation exists:
        {
            "is_buildup":    False,
            "related_topic": "",
            "relation":      "none"
        }

    Raises:
        RuntimeError: If the LLM call fails or returns malformed JSON.
    """
    prompt = TOPIC_RELATEDNESS_PROMPT.format(
        current_topic=current_topic,
        weak_topics=", ".join(weak_topics) if weak_topics else "None",
        strong_topics=", ".join(strong_topics) if strong_topics else "None",
    )

    print(
        f"[notes_chain] Pre-call: checking topic relatedness for '{current_topic}'..."
    )

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if LLM added them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        print(f"[notes_chain] Relatedness result: {result}")
        return result, response

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Topic relatedness pre-call returned malformed JSON. "
            f"Raw response: '{raw[:200]}' | Error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Topic relatedness pre-call failed. | Error: {e}") from e


def _sanitise_heading(heading: str) -> tuple[str, bool]:
    """
    Checks whether a section heading exceeds the maximum allowed length
    and flags it for LLM renaming if so.

    The programmatic rule is simple and consistent:
        heading length > 60 characters → flagged

    The LLM decides what to rename it to — this function only decides
    whether renaming is needed.

    Args:
        heading: The raw section heading string from the topic_map.

    Returns:
        Tuple of (heading, is_flagged):

        Example — heading within limit:
        ("RISC Architecture", False)

        Example — heading too long:
        ("An Overview of Reduced Instruction Set Computer Architecture "
         "and Its Design Philosophy", True)

    """
    is_flagged = len(heading) > _MAX_HEADING_LENGTH
    if is_flagged:
        print(
            f"[notes_chain] Heading flagged for renaming ({len(heading)} chars): '{heading[:60]}...'"
        )
    return heading, is_flagged


def _format_chunks(chunks: List[Document]) -> str:
    """
    Formats a list of Document objects into a single context string
    for injection into the LLM prompt.

    Each chunk is separated by a blank line. The page number is prepended
    to each chunk so the LLM has positional context within the section.

    Args:
        chunks: List of Document objects for one section from topic_map.

    Returns:
        A single formatted string ready for prompt injection. Example:

        "[Page 3]
        Gradient descent is an optimisation algorithm that minimises
        a loss function by iteratively moving in the direction of
        steepest descent as defined by the negative of the gradient.

        [Page 3]
        The learning rate alpha controls the size of each step. A value
        too large causes divergence; too small causes slow convergence."

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot format an empty chunks list into context.")

    parts = []
    for chunk in chunks:
        page = chunk.metadata.get("page_number", "?")
        parts.append(f"[Page {page}]\n{chunk.page_content.strip()}")

    return "\n\n".join(parts)


def _build_elaboration_instruction(relatedness: dict) -> str:
    """
    Builds the elaboration instruction string to inject into the
    section prompt based on the topic relatedness pre-call result.

    Args:
        relatedness: The dict returned by _check_topic_relatedness().

    Returns:
        A formatted elaboration instruction string. Examples:

        When relation is 'weak':
        "IMPORTANT: This topic builds upon 'Memory Management', which this
         student has previously struggled with. Where relevant, briefly
         reinforce key ideas from that prerequisite topic..."

        When relation is 'strong':
        "NOTE: This topic builds upon 'Boolean Algebra', which this student
         has a strong grasp of. You may reference concepts from that topic
         freely as anchors without re-explaining them..."

        When relation is 'none':
        "Treat this topic as standalone — no significant prerequisite topics
         have been identified as relevant to this student's background."
    """
    relation = relatedness.get("relation", "none")
    related_topic = relatedness.get("related_topic", "")

    if relation == "weak":
        return ELABORATION_WEAK.format(related_topic=related_topic)
    elif relation == "strong":
        return ELABORATION_STRONG.format(related_topic=related_topic)
    else:
        return ELABORATION_NONE


def _classify_rate_error(e: Exception) -> str:
    """
    Inspects an exception and classifies it as an RPM limit, RPD (daily)
    limit, or an unrelated error.

    Classification logic (checked in order):
        1. HTTP 429 status code on the exception or its cause.
        2. Message keywords for daily/quota exhaustion  → "rpd"
        3. Message keywords for per-minute rate limits  → "rpm"
        4. Anything else                                → "other"

    The RPD keywords are checked before RPM keywords because an RPD error
    from the Gemini API typically also contains "rate" in its message.

    Args:
        e: Any exception raised during an LLM call.

    Returns:
        "rpm"   — per-minute rate limit (recoverable with backoff + retry).
        "rpd"   — daily quota exhausted (unrecoverable in this session).
        "other" — unrelated error; existing retry logic handles it.

    Examples:
        >>> _classify_rate_error(Exception("429 Resource Exhausted"))
        "rpm"
        >>> _classify_rate_error(Exception("Quota exceeded for quota metric"))
        "rpd"
        >>> _classify_rate_error(Exception("Connection timeout"))
        "other"
    """
    msg = str(e).lower()

    # Check HTTP status on the exception itself or its __cause__
    status = getattr(e, "status_code", None) or getattr(
        getattr(e, "__cause__", None), "status_code", None
    )
    if status == 429:
        # Still need to distinguish RPM vs RPD via message
        if any(k in msg for k in ("quota exceeded", "daily limit", "per day")):
            return "rpd"
        return "rpm"

    # Daily / project quota exhaustion signals (check before rpm)
    if any(k in msg for k in ("quota exceeded", "daily limit", "per day", "billing")):
        return "rpd"

    # Per-minute rate limit signals
    if any(k in msg for k in ("rate limit", "resource exhausted", "too many requests")):
        return "rpm"

    return "other"


def _call_section_llm(
    student_id: str,
    llm: BaseChatModel,
    heading: str,
    is_flagged: bool,
    chunks: List[Document],
    learning_pace: str,
    elaboration_instruction: str,
    running_summary: List[str],
) -> tuple[dict, list]:
    """
    Runs one LLM call for a single section, returning the condensed
    content, (optionally renamed) heading, and a brief section summary.

    Retry architecture (two nested loops):

        Outer loop — rate-limit retries (up to _MAX_RATE_LIMIT_RETRIES):
            Only active when USE_GEMINI is True.
            On RPM errors: backs off [15, 30, 60] s then retries the full
            inner loop. The 60 s final backoff guarantees the per-minute
            window resets before the last attempt.
            On RPD errors: raises RateLimitExhaustedError(kind="rpd")
            immediately — no backoff can recover a daily quota.

        Inner loop — parse-failure retries (up to _MAX_SECTION_RETRIES):
            Unchanged from previous version. Handles unparseable LLM output
            via JsonOutputParser then json_repair fallback.

    Token usage is accumulated across ALL attempts (outer + inner,
    including failed ones) so @token_guard deducts correctly.

    Args:
        llm:                      Initialised LLM instance.
        heading:                  The section heading (may be renamed by LLM).
        is_flagged:               Whether heading exceeded 60 chars.
        chunks:                   All Document objects for this section.
        learning_pace:            "slow" | "average" | "fast"
        elaboration_instruction:  Built from _build_elaboration_instruction().
        running_summary:          List of prior section summaries (strings).

    Returns:
        Tuple of (parsed_dict, all_responses):

        parsed_dict example:
        {
            "section_heading":   "RISC Architecture",
            "condensed_content": "RISC processors use a small set of simple
                                  instructions...\\n\\n- Fixed instruction length
                                  \\n- Large number of registers",
            "section_summary":   "Covered RISC design philosophy, contrasting
                                  simple fixed-length instructions with CISC
                                  complexity."
        }

        all_responses: list of all LangChain response objects from every
                       attempt (including failed ones), so token usage
                       from retries can be accumulated by the caller.

    Raises:
        ValueError:               If learning_pace is not recognised.
        RateLimitExhaustedError:  If daily (RPD) quota is hit, or if all
                                  RPM backoff retries are exhausted without
                                  a successful call.
        RuntimeError:             If the inner parse-retry loop is exhausted
                                  for a reason unrelated to rate limiting.
    """
    if learning_pace not in _NOTES_PROMPT_MAP:
        raise ValueError(
            f"Unrecognised learning_pace: '{learning_pace}'. "
            f"Valid values: {list(_NOTES_PROMPT_MAP.keys())}"
        )

    # Extract LTM context
    ltm_context = view_ltm(student_id)

    # Build rename instruction
    if is_flagged:
        rename_instruction = RENAME_INSTRUCTION.format(original_heading=heading)
    else:
        rename_instruction = RENAME_INSTRUCTION_NONE.format(section_heading=heading)

    # Format running summary
    if running_summary:
        summary_text = "\n".join(f"- {s}" for s in running_summary)
    else:
        summary_text = "This is the first section — no prior content yet."

    # Format chunks into context string
    context = _format_chunks(chunks)

    # Select prompt template based on learning pace
    prompt_template = _NOTES_PROMPT_MAP[learning_pace]

    prompt = prompt_template.format(
        elaboration_instruction=elaboration_instruction,
        ltm_context=ltm_context,
        running_summary=summary_text,
        section_heading=heading,
        context=context,
        rename_instruction=rename_instruction,
    )

    parser = JsonOutputParser()
    all_responses = []  # accumulate across all attempts
    last_error = None

    # ── Outer loop: rate-limit retries ───────────────────────────────────────
    # When running locally (USE_GEMINI = False) this loop executes exactly
    # once (rate_attempt = 1) and never enters the backoff branch.
    max_rate_attempts = _MAX_RATE_LIMIT_RETRIES if USE_GEMINI else 1

    for rate_attempt in range(1, max_rate_attempts + 1):
        # ── Inner loop: parse-failure retries ────────────────────────────────
        inner_succeeded = False

        for attempt in range(1, _MAX_SECTION_RETRIES + 1):
            raw = ""  # initialise for error reporting in the except block

            try:
                response = llm.invoke(prompt)
                all_responses.append(response)
                raw = response.content.strip()

                # Primary: JsonOutputParser handles fence stripping + json.loads()
                try:
                    parsed = parser.parse(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f"Expected a JSON object (dict) but got "
                            f"{type(parsed).__name__}: {str(parsed)[:200]}"
                        )
                    if attempt > 1:
                        print(
                            f"[notes_chain] ✓ Retry {attempt} succeeded for '{heading[:50]}'."
                        )
                    return parsed, all_responses

                # Fallback: json_repair patches malformed JSON (literal newlines,
                # missing delimiters, truncated strings, etc.) before parsing
                except Exception:
                    print(
                        f"[notes_chain] ⚠ JsonOutputParser failed for "
                        f"'{heading[:50]}' — attempting json_repair."
                    )
                    repaired = repair_json(raw)
                    parsed = json.loads(repaired)
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f"json_repair produced {type(parsed).__name__} "
                            f"instead of dict: {str(parsed)[:200]}"
                        )
                    if attempt > 1:
                        print(
                            f"[notes_chain] ✓ Retry {attempt} succeeded "
                            f"(via json_repair) for '{heading[:50]}'."
                        )
                    return parsed, all_responses

            except Exception as e:
                # ── Rate-limit detection (only meaningful when USE_GEMINI) ──
                if USE_GEMINI:
                    kind = _classify_rate_error(e)

                    if kind == "rpd":
                        # Daily quota — no point retrying at all
                        print(
                            f"[notes_chain] ✗ Daily quota exhausted on "
                            f"'{heading[:50]}'. Terminating notes generation."
                        )
                        raise RateLimitExhaustedError(
                            f"Daily Gemini quota exhausted during section "
                            f"'{heading[:60]}'.",
                            kind="rpd",
                        )

                    if kind == "rpm":
                        # Per-minute limit — break inner loop, let outer
                        # loop handle the backoff
                        print(
                            f"[notes_chain] ⚠ RPM limit hit on "
                            f"'{heading[:50]}' (rate_attempt "
                            f"{rate_attempt}/{max_rate_attempts})."
                        )
                        last_error = e
                        inner_succeeded = False
                        break  # exit inner loop → outer loop applies backoff

                # Non-rate-limit parse failure — standard inner-loop retry
                last_error = e
                if attempt < _MAX_SECTION_RETRIES:
                    print(
                        f"[notes_chain] ⚠ Attempt {attempt}/{_MAX_SECTION_RETRIES} "
                        f"failed for '{heading[:50]}': {e}. Retrying..."
                    )
                else:
                    print(
                        f"[notes_chain] ✗ All {_MAX_SECTION_RETRIES} parse attempts "
                        f"exhausted for '{heading[:50]}'. Last error: {e}"
                    )
            else:
                inner_succeeded = True  # set if no exception was raised

        # Inner loop finished without returning — either RPM break or parse exhaustion.
        # If not an RPM issue (inner exhausted parse retries), raise immediately.
        if inner_succeeded or not USE_GEMINI:
            break

        # ── RPM backoff ───────────────────────────────────────────────────────
        if rate_attempt < max_rate_attempts:
            wait = _RPM_BACKOFF_SECONDS[rate_attempt - 1]
            print(
                f"[notes_chain] ⏳ Backing off {wait}s before rate-limit retry "
                f"{rate_attempt + 1}/{max_rate_attempts} for '{heading[:50]}'."
            )
            time.sleep(wait)
        else:
            # All rate-limit retries exhausted (60 s backoff already tried)
            raise RateLimitExhaustedError(
                f"RPM rate limit retries exhausted after {max_rate_attempts} "
                f"attempts for section '{heading[:60]}'. Last error: {last_error}",
                kind="rpm",
            )

    # Reaching here means inner parse-retries exhausted (non-rate-limit path)
    raise RuntimeError(
        f"Section LLM call failed after {_MAX_SECTION_RETRIES} parse attempts "
        f"for heading '{heading[:60]}'. Last raw output: '{raw[:200]}' | "
        f"Last error: {last_error}"
    )


def _preprocess_md_line(line: str) -> tuple[str, bool]:
    """
    Convert a single markdown line to reportlab-compatible XML.

    Returns:
        (processed_line, is_bullet)

    Example:
        "**RISC** uses a *small* set of instructions"
        → ("<b>RISC</b> uses a <i>small</i> set of instructions", False)

        "- Fixed instruction length"
        → ("Fixed instruction length", True)
    """
    # Must escape XML special chars BEFORE inserting tags
    line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Bold: **text** or __text__
    line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
    line = re.sub(r"__(.*?)__", r"<b>\1</b>", line)

    # Italic: *text* or _text_  (after bold to avoid overlap)
    line = re.sub(r"\*(.*?)\*", r"<i>\1</i>", line)
    line = re.sub(r"_(.*?)_", r"<i>\1</i>", line)

    # Inline code: `code`
    line = re.sub(r"`(.*?)`", r'<font name="Courier">\1</font>', line)

    # Strip stray heading markers (## / ###) that bleed into body
    line = re.sub(r"^#{1,6}\s+", "", line)

    # Horizontal rules → skip (caller renders as Spacer)
    if re.match(r"^-{3,}$", line.strip()):
        return "", False

    # Detect bullet: leading -, *, or •
    is_bullet = bool(re.match(r"^[-*•]\s+", line))
    if is_bullet:
        line = re.sub(r"^[-*•]\s+", "", line)

    return line.strip(), is_bullet


def _fix_html_tags(text: str) -> str:
    """
    Uses BeautifulSoup to fix malformed, unclosed, or overlapping
    HTML/XML tags generated by the LLM, making them safe for ReportLab.
    """
    if not text:
        return ""

    # 1. Temporarily protect valid standalone characters like mathematical '<' or '>'
    # so BeautifulSoup doesn't treat them as broken tags.
    # We only treat text inside actual angle brackets containing words/quotes as tags.

    # 2. Parse with BeautifulSoup to force strict, valid nesting trees
    soup = BeautifulSoup(text, "html.parser")
    fixed_text = str(soup)

    # 3. ReportLab paragraph parser does not recognize self-closing tags like <b />.
    # Convert any self-closing tags back to standard tag pairs if they appear.
    fixed_text = re.sub(r"<(\w+)([^>]*)\s*/>", r"<\1\2></\1>", fixed_text)

    return fixed_text


def _build_pdf(
    notes_json: Dict[str, str],
    student_id: str,
    current_topic: str,
) -> Path:
    """
    Constructs a formatted PDF from the notes_json dict using reportlab.

    Each key in notes_json becomes a section heading in the PDF.
    Each value becomes the section body, with newlines and bullet points
    preserved from the LLM output.

    Args:
        notes_json:    Dict mapping section_heading → condensed_content.
        student_id:    Used to namespace the output filename.
        current_topic: Used as the PDF title.

    Returns:
        Path to the generated PDF file. Example:
        Path("data/notes/abc123_Computer_Architecture_20260419_143022.pdf")

    Raises:
        RuntimeError: If PDF construction or file write fails.

    Example PDF structure:
        ┌─────────────────────────────────────┐
        │  Computer Architecture              │  ← title
        │  Condensed Study Notes              │  ← subtitle
        │  Generated: 2026-04-19 14:30        │  ← timestamp
        ├─────────────────────────────────────┤
        │  RISC Architecture                  │  ← section heading
        │                                     │
        │  RISC processors use a small set... │  ← condensed content
        │  • Fixed instruction length         │
        │  • Large number of registers        │
        ├─────────────────────────────────────┤
        │  Memory Management                  │  ← next section
        │  ...                                │
        └─────────────────────────────────────┘
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_topic = current_topic.replace(" ", "_")[:40]
    filename = f"{student_id}_{safe_topic}_{timestamp}.pdf"
    output_path = _OUTPUT_DIR / filename

    try:
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "CogniTitle",
            parent=styles["Title"],
            fontSize=20,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=4,
        )
        subtitle_style = ParagraphStyle(
            "CogniSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#555555"),
            spaceAfter=2,
        )
        timestamp_style = ParagraphStyle(
            "CogniTimestamp",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#999999"),
            spaceAfter=20,
        )
        heading_style = ParagraphStyle(
            "CogniHeading",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#16213e"),
            spaceBefore=14,
            spaceAfter=6,
            borderPad=4,
        )
        body_style = ParagraphStyle(
            "CogniBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#2d2d2d"),
            spaceAfter=6,
            alignment=TA_LEFT,
        )
        bullet_style = ParagraphStyle(
            "CogniBullet",
            parent=body_style,
            leftIndent=14,
            bulletIndent=4,
            spaceAfter=3,
        )

        story = []

        # ── Cover block ───────────────────────────────────────────────────────
        story.append(Paragraph(_fix_html_tags(current_topic), title_style))
        story.append(Paragraph("Condensed Study Notes", subtitle_style))
        story.append(
            Paragraph(
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
                timestamp_style,
            )
        )

        # ── Section content ───────────────────────────────────────────────────
        for section_heading, condensed_content in notes_json.items():
            story.append(Paragraph(_fix_html_tags(section_heading), heading_style))

            if isinstance(condensed_content, list):
                condensed_content = "\n".join(
                    item if isinstance(item, str) else str(item)
                    for item in condensed_content
                )

            for raw_line in condensed_content.split("\n"):
                raw_line = raw_line.strip()
                if not raw_line:
                    story.append(Spacer(1, 6))
                    continue

                line, is_bullet = _preprocess_md_line(raw_line)

                if not line:  # was a horizontal rule — skip
                    story.append(Spacer(1, 8))
                    continue

                # Programmatically repair the nesting structures of the inline tags
                clean_line = _fix_html_tags(line)

                if is_bullet:
                    story.append(Paragraph(f"• {clean_line}", bullet_style))
                else:
                    story.append(Paragraph(clean_line, body_style))

            story.append(Spacer(1, 10))

        doc.build(story)
        print(f"[notes_chain] PDF built: {output_path}")
        return output_path

    except Exception as e:
        raise RuntimeError(
            f"PDF construction failed for student '{student_id}', "
            f"topic '{current_topic}'. | Error: {e}"
        ) from e


def _reingest_pdf(
    pdf_path: Path,
    student_id: str,
    current_topic: str,
    course: str,
    embedder: HuggingFaceEmbeddings,
    store: Chroma,
) -> None:
    """
    Runs the full RAG ingestion pipeline on the generated condensed notes PDF
    so that the qa_chain has access to both the original and condensed material.

    After this call, the student's Chroma collection contains:
        - Original PDF chunks  (raw, verbatim material)
        - Condensed note chunks (LLM-structured, semantically cleaner)

    The qa_chain retrieves from both simultaneously, giving it richer
    and more structured retrieval results.

    Args:
        pdf_path:      Path to the generated condensed notes PDF.
        student_id:    Used for scoping the Chroma collection.
        current_topic: Used as the topic metadata tag on new chunks.
        course:        The course these documents belong to.
        embedder:      The initialised embedding model.
        store:         The student-scoped Chroma store.

    Returns:
        None

    Raises:
        RuntimeError: If ingestion, embedding, or storage fails.
    """
    print(f"[notes_chain] Re-ingesting condensed notes PDF into Chroma...")

    try:
        # Step 1 — Parse PDF into Document chunks
        documents = process_and_load_file(file_path=str(pdf_path))
        print(f"[notes_chain] → {len(documents)} chunk(s) parsed from condensed PDF.")

        # Step 2 — Generate embeddings
        embeddings = generate_embeddings(documents=documents, embedder=embedder)

        # Step 3 — Add to Chroma (deduplication handled inside)
        add_documents_to_chroma(
            store=store,
            embeddings=embeddings,
            documents=documents,
            condensed=True,
            course=course,
            topic=current_topic,
            file_name=pdf_path.name,
        )

        print(f"[notes_chain] ✓ Condensed notes successfully ingested into Chroma.")

    except Exception as e:
        raise RuntimeError(
            f"Re-ingestion of condensed notes PDF failed for student "
            f"'{student_id}'. PDF: {pdf_path} | Error: {e}"
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


@token_guard
def run_notes_chain(
    student_id: str,
    topic_map: Dict[str, List[Document]],
    current_topic: str,
    weak_topics: List[str],
    strong_topics: List[str],
    course: str,
    learning_pace: str,
    llm: BaseChatModel,
    embedder: HuggingFaceEmbeddings,
    store: Chroma,
) -> Any:
    """
    Generate personalised condensed study notes for a student's topic,
    construct a PDF, re-ingest it into Chroma, and return the PDF path.

    This function runs once per student request and handles the full
    lifecycle internally — from topic relatedness classification through
    to PDF delivery and Chroma re-ingestion.

    Rate limiting:
        Controlled by the module-level USE_GEMINI flag (set manually
        before running):

            USE_GEMINI = False  →  local Ollama. No inter-section delay.
                                   Rate-limit handling is completely inactive.

            USE_GEMINI = True   →  Gemini API (free tier). A
                                   _REQUEST_DELAY_SECONDS sleep is inserted
                                   between every section call to stay under
                                   the RPM ceiling. RPM errors inside
                                   _call_section_llm are retried with
                                   exponential backoff [15, 30, 60] s.
                                   The 60 s final wait guarantees the
                                   per-minute window resets before the
                                   last attempt.

        If the daily (RPD) quota is hit mid-generation, the loop terminates
        early and a partial PDF is built from the sections processed so far.
        The partial PDF is valid and returned normally — the caller receives
        whatever was completed before the quota ran out.

    Page limit:
        Designed for PDFs up to 40 pages (production).
        During testing, the limit is enforced at 20 pages.
        These limits are applied at ingestion time, not here.

    Token tracking:
        Each section LLM call + the pre-call consume tokens. This function
        accumulates total usage across all calls and returns a synthetic
        response object carrying usage_metadata so @token_guard can
        deduct correctly from the student's balance.

    Args:
        student_id:    Unique identifier for the student. Required first
                       by @token_guard — do not reorder.
        topic_map:     Dict from get_topic_chunks() mapping section heading
                       → list of Document objects.
        current_topic: The overarching topic being studied (e.g.
                       "Computer Architecture").
        weak_topics:   Topics the student has previously struggled with.
                       From learner profile. May be empty list.
        strong_topics: Topics the student has strong grasp of.
                       From learner profile. May be empty list.
        course:        The course name (e.g. "Computer Architecture").
        learning_pace: "slow" | "average" | "fast"
                       Derived from student's CGPA bracket.
        llm:           Initialised LangChain LLM instance.
        embedder:      Initialised HuggingFaceEmbeddings instance.
        store:         Student-scoped Chroma vector store.

    Returns:
        A synthetic response object with:
            .content        → Path to the generated PDF (as string).
                              May be a partial PDF if daily quota was hit.
            .usage_metadata → Accumulated token usage across all LLM calls:
                {
                    "input_tokens":  1840,
                    "output_tokens": 3210,
                    "total_tokens":  5050
                }

    Raises:
        ValueError:   If learning_pace is invalid or topic_map is empty.
        RuntimeError: If PDF construction fails, or if a non-rate-limit
                      section failure propagates unexpectedly.

    Example call from a router:
        topic_map = get_topic_chunks(store=store)
        result = run_notes_chain(
            student_id    = current_user["id"],
            topic_map     = topic_map,
            current_topic = "Computer Architecture",
            weak_topics   = profile["weak_topics"],
            strong_topics = profile["strong_topics"],
            course        = profile["course"],
            learning_pace = profile["learning_pace"],
            llm           = llm,
            embedder      = embedder,
            store         = store,
        )
        pdf_path = result.content

    Example result.content:
        "data/notes/abc123_Computer_Architecture_20260419_143022.pdf"

    Example result.usage_metadata:
        {
            "input_tokens":  1840,
            "output_tokens": 3210,
            "total_tokens":  5050
        }
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not topic_map:
        raise ValueError(
            "topic_map is empty. Ensure the PDF has been ingested and "
            "get_topic_chunks() returned results before calling run_notes_chain()."
        )

    if learning_pace not in _NOTES_PROMPT_MAP:
        raise ValueError(
            f"Invalid learning_pace: '{learning_pace}'. "
            f"Valid values: {list(_NOTES_PROMPT_MAP.keys())}"
        )

    if not current_topic or not current_topic.strip():
        raise ValueError("current_topic cannot be empty or whitespace.")

    print(f"\n[notes_chain] ═══ Starting notes generation ═══")
    print(f"[notes_chain] Mode: {'Gemini API' if USE_GEMINI else 'Local Ollama'}")
    print(
        f"[notes_chain] Student: {student_id} | Topic: {current_topic} | Pace: {learning_pace}"
    )
    print(f"[notes_chain] Sections to process: {len(topic_map)}")

    # Accumulate token usage across all LLM calls
    total_input_tokens = 0
    total_output_tokens = 0

    # ── Step 0: Topic relatedness pre-call ───────────────────────────────────
    relatedness = {
        "is_buildup": False,
        "related_topic": "",
        "relation": "none",
    }  # safe default

    if weak_topics or strong_topics:
        try:
            relatedness, precall_response = _check_topic_relatedness(
                llm=llm,
                current_topic=current_topic,
                weak_topics=weak_topics,
                strong_topics=strong_topics,
            )
            precall_usage = getattr(precall_response, "usage_metadata", {})
            total_input_tokens += precall_usage.get("input_tokens", 0)
            total_output_tokens += precall_usage.get("output_tokens", 0)

        except RuntimeError:
            # Non-fatal — default to neutral if pre-call fails
            print(
                "[notes_chain] ⚠ Pre-call failed — defaulting to neutral elaboration mode."
            )

    elaboration_instruction = _build_elaboration_instruction(relatedness)

    # ── Step 1 + 2: Sanitise headings + section-by-section LLM calls ─────────
    notes_json: Dict[str, str] = {}
    running_summary: List[str] = []
    total_sections = len(topic_map)
    sections_done = 0

    for idx, (heading, chunks) in enumerate(topic_map.items(), start=1):
        # Inter-section throttle — only when hitting the Gemini API.
        # Applied before the call (except for the very first section)
        # to stay under the free-tier RPM ceiling.
        if USE_GEMINI and idx > 1:
            print(
                f"[notes_chain] ⏳ Throttle: sleeping {_REQUEST_DELAY_SECONDS}s before next section."
            )
            time.sleep(_REQUEST_DELAY_SECONDS)

        print(
            f"\n[notes_chain] Processing section {idx}/{total_sections}: '{heading[:50]}'"
        )

        # Sanitise heading
        _, is_flagged = _sanitise_heading(heading)

        parsed = None
        all_responses = []

        # Retry loop — handles both hard failures (RuntimeError) and
        # silent empty content returned by the LLM
        for attempt in range(1, _MAX_EMPTY_RETRIES + 1):
            try:
                parsed, responses = _call_section_llm(
                    student_id=student_id,
                    llm=llm,
                    heading=heading,
                    is_flagged=is_flagged,
                    chunks=chunks,
                    learning_pace=learning_pace,
                    elaboration_instruction=elaboration_instruction,
                    running_summary=running_summary,
                )
                all_responses.extend(responses)

                # Treat empty condensed_content as a retriable failure —
                # LLM succeeded structurally but returned nothing useful
                condensed_content = parsed.get("condensed_content", "")
                if not condensed_content or not str(condensed_content).strip():
                    print(
                        f"[notes_chain] ⚠ Section '{heading[:50]}' returned empty "
                        f"condensed_content (attempt {attempt}) — retrying."
                    )
                    parsed = None
                    continue

                break  # content is valid — exit retry loop

            except RateLimitExhaustedError as e:
                # Accumulate tokens from any partial attempts before exiting
                for resp in all_responses:
                    resp_usage = getattr(resp, "usage_metadata", None)
                    if resp_usage:
                        total_input_tokens += resp_usage.get("input_tokens", 0)
                        total_output_tokens += resp_usage.get("output_tokens", 0)

                if e.kind == "rpd":
                    # Daily quota — cannot recover. Build partial PDF now.
                    print(
                        f"[notes_chain] ✗ Daily quota exhausted at section "
                        f"{idx}/{total_sections}. Building partial PDF "
                        f"({sections_done} section(s) completed)."
                    )
                else:
                    # RPM retries exhausted despite backoff — treat as terminal
                    print(
                        f"[notes_chain] ✗ RPM retries exhausted at section "
                        f"{idx}/{total_sections}. Building partial PDF "
                        f"({sections_done} section(s) completed)."
                    )

                # Break out of both the empty-retry loop and the section loop
                parsed = None
                break  # exits empty-retry loop

            except RuntimeError as e:
                print(
                    f"[notes_chain] ✗ Section '{heading[:50]}' RuntimeError "
                    f"on attempt {attempt}: {e}"
                )
                continue

        # Accumulate token usage from ALL attempts (including retries)
        for resp in all_responses:
            resp_usage = getattr(resp, "usage_metadata", None)
            if resp_usage is None:
                # Ollama locally doesn't always report token usage.
                # In production against Gemini this will never be None.
                continue
            total_input_tokens += resp_usage.get("input_tokens", 0)
            total_output_tokens += resp_usage.get("output_tokens", 0)

        # Check if we broke out due to RateLimitExhaustedError
        # (detected by no section appended AND quota signals already printed)
        if parsed is None and isinstance(locals().get("e"), RateLimitExhaustedError):
            break  # exit the outer section loop → build partial PDF

        # All retry attempts exhausted — last resort placeholder
        if parsed is None:
            print(
                f"[notes_chain] ✗ Section '{heading[:50]}' failed after "
                f"{_MAX_EMPTY_RETRIES} attempts. Inserting placeholder."
            )
            notes_json[heading] = "[Content could not be generated for this section.]"
            running_summary.append(f"Section '{heading}' could not be processed.")
            sections_done += 1
            continue

        # Extract fields from LLM response
        clean_heading = parsed.get("section_heading", heading)
        condensed_content = parsed.get("condensed_content", "")
        section_summary = parsed.get("section_summary", "")

        notes_json[clean_heading] = condensed_content
        running_summary.append(section_summary)
        sections_done += 1

        print(f"[notes_chain] ✓ Section '{clean_heading[:50]}' condensed.")

    # ── Step 3: Build PDF ─────────────────────────────────────────────────────
    # notes_json may be partial if quota was hit — _build_pdf handles any size.
    pdf_path = _build_pdf(
        notes_json=notes_json,
        student_id=student_id,
        current_topic=current_topic,
    )

    # ── Step 4: Re-ingest condensed PDF into Chroma ───────────────────────────
    # Keep this idle for now
    # _reingest_pdf(...)

    # ── Step 5: Build synthetic response for token_guard ─────────────────────
    total_tokens = total_input_tokens + total_output_tokens

    print(f"\n[notes_chain] ═══ Notes generation complete ═══")
    print(f"[notes_chain] PDF: {pdf_path}")
    print(f"[notes_chain] Sections completed: {sections_done}/{total_sections}")
    print(
        f"[notes_chain] Total tokens — "
        f"in: {total_input_tokens:,} | out: {total_output_tokens:,} | total: {total_tokens:,}"
    )

    class _SyntheticResponse:
        """
        Lightweight response wrapper returned to token_guard.
        Carries the PDF path as content and accumulated token usage.
        content may represent a partial PDF if quota was exhausted.
        """

        def __init__(self, pdf_path: Path, input_t: int, output_t: int, total_t: int):
            self.content = str(pdf_path)
            self.usage_metadata = {
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": total_t,
            }

    return _SyntheticResponse(
        pdf_path=pdf_path,
        input_t=total_input_tokens,
        output_t=total_output_tokens,
        total_t=total_tokens,
    )
