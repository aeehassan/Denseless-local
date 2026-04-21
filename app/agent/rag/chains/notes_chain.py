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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

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
from app.services.token_service import token_guard


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Maximum heading length before flagging for LLM renaming
_MAX_HEADING_LENGTH = 65

# Maximum number of retry attempts when LLM returns unparseable output
_MAX_SECTION_RETRIES = 3

# Prompt map keyed by learning pace
_NOTES_PROMPT_MAP = {
    "slow":    NOTES_PROMPT_SLOW,
    "average": NOTES_PROMPT_AVERAGE,
    "fast":    NOTES_PROMPT_FAST,
}

# Output directory for generated PDFs
_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "notes"


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

    print(f"[notes_chain] Pre-call: checking topic relatedness for '{current_topic}'...")

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
        raise RuntimeError(
            f"Topic relatedness pre-call failed. | Error: {e}"
        ) from e


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
        print(f"[notes_chain] Heading flagged for renaming ({len(heading)} chars): '{heading[:60]}...'")
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


def _call_section_llm(
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

    Retries up to _MAX_SECTION_RETRIES times if the LLM returns
    unparseable output. Token usage is accumulated across all attempts
    (including failed ones) so token_guard deducts correctly.

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
        ValueError:   If learning_pace is not recognised.
        RuntimeError: If LLM call fails after exhausting all retry attempts.
    """
    if learning_pace not in _NOTES_PROMPT_MAP:
        raise ValueError(
            f"Unrecognised learning_pace: '{learning_pace}'. "
            f"Valid values: {list(_NOTES_PROMPT_MAP.keys())}"
        )

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
        running_summary=summary_text,
        section_heading=heading,
        context=context,
        rename_instruction=rename_instruction,
    )

    parser = JsonOutputParser()
    all_responses = []  # track every response for token accounting
    last_error = None

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
                        f"Expected a JSON object (dict) but got {type(parsed).__name__}: {str(parsed)[:200]}"
                    )
                if attempt > 1:
                    print(f"[notes_chain] ✓ Retry {attempt} succeeded for '{heading[:50]}'.")
                return parsed, all_responses

            # Fallback: json_repair patches malformed JSON (literal newlines,
            # missing delimiters, truncated strings, etc.) before parsing
            except Exception:
                print(f"[notes_chain] ⚠ JsonOutputParser failed for '{heading[:50]}' — attempting json_repair.")
                repaired = repair_json(raw)
                parsed = json.loads(repaired)
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"json_repair produced {type(parsed).__name__} instead of dict: {str(parsed)[:200]}"
                    )
                if attempt > 1:
                    print(f"[notes_chain] ✓ Retry {attempt} succeeded (via json_repair) for '{heading[:50]}'.")
                return parsed, all_responses

        except Exception as e:
            last_error = e
            if attempt < _MAX_SECTION_RETRIES:
                print(
                    f"[notes_chain] ⚠ Attempt {attempt}/{_MAX_SECTION_RETRIES} failed for "
                    f"'{heading[:50]}': {e}. Retrying..."
                )
            else:
                print(
                    f"[notes_chain] ✗ All {_MAX_SECTION_RETRIES} attempts exhausted for "
                    f"'{heading[:50]}'. Last error: {e}"
                )

    # All retries exhausted — raise so the caller can decide what to do
    raise RuntimeError(
        f"Section LLM call failed after {_MAX_SECTION_RETRIES} attempts for "
        f"heading '{heading[:60]}'. Last raw output: '{raw[:200]}' | "
        f"Last error: {last_error}"
    )


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

        story = []

        # ── Cover block ───────────────────────────────────────────────────────
        story.append(Paragraph(current_topic, title_style))
        story.append(Paragraph("Condensed Study Notes", subtitle_style))
        story.append(Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            timestamp_style,
        ))

        # ── Section content ───────────────────────────────────────────────────
        for section_heading, condensed_content in notes_json.items():
            story.append(Paragraph(section_heading, heading_style))

            # Split on newlines and render each line as a paragraph
            # to preserve bullet points and spacing from LLM output
            if isinstance(condensed_content, list):
                # LLM returned a list of bullet points — join them
                condensed_content = "\n".join(
                    item if isinstance(item, str) else str(item)
                    for item in condensed_content
                )

            lines = condensed_content.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 6))
                    continue

                # Escape reportlab XML special characters
                line = (
                    line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                )

                story.append(Paragraph(line, body_style))

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
        llm:           Initialised LangChain LLM instance. Should support
                       temperature configuration.
        embedder:      Initialised HuggingFaceEmbeddings instance.
        store:         Student-scoped Chroma vector store.

    Returns:
        A synthetic response object with:
            .content        → Path to the generated PDF (as string)
            .usage_metadata → Accumulated token usage across all LLM calls:
                {
                    "input_tokens":  1840,
                    "output_tokens": 3210,
                    "total_tokens":  5050
                }

    Raises:
        ValueError:   If learning_pace is invalid or topic_map is empty.
        RuntimeError: If any LLM call, PDF construction, or re-ingestion fails.

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
    print(f"[notes_chain] Student: {student_id} | Topic: {current_topic} | Pace: {learning_pace}")
    print(f"[notes_chain] Sections to process: {len(topic_map)}")

    # Accumulate token usage across all LLM calls
    total_input_tokens  = 0
    total_output_tokens = 0

    # ── Step 0: Topic relatedness pre-call ───────────────────────────────────
    relatedness = {"is_buildup": False, "related_topic": "", "relation": "none"}  # safe default

    if weak_topics or strong_topics:
        try:
            relatedness, precall_response = _check_topic_relatedness(
                llm=llm,
                current_topic=current_topic,
                weak_topics=weak_topics,
                strong_topics=strong_topics,
            )
            precall_usage = getattr(precall_response, "usage_metadata", {})
            total_input_tokens  += precall_usage.get("input_tokens", 0)
            total_output_tokens += precall_usage.get("output_tokens", 0)

        except RuntimeError:
            # Non-fatal — default to neutral if pre-call fails
            print("[notes_chain] ⚠ Pre-call failed — defaulting to neutral elaboration mode.")
            # relatedness already set to neutral default above

    elaboration_instruction = _build_elaboration_instruction(relatedness)

    # ── Step 1 + 2: Sanitise headings + section-by-section LLM calls ─────────
    notes_json: Dict[str, str] = {}
    running_summary: List[str] = []

    for idx, (heading, chunks) in enumerate(topic_map.items(), start=1):
        print(f"\n[notes_chain] Processing section {idx}/{len(topic_map)}: '{heading[:50]}'")

        # Sanitise heading
        _, is_flagged = _sanitise_heading(heading)

        try:
            parsed, all_responses = _call_section_llm(
                llm=llm,
                heading=heading,
                is_flagged=is_flagged,
                chunks=chunks,
                learning_pace=learning_pace,
                elaboration_instruction=elaboration_instruction,
                running_summary=running_summary,
            )

            # Accumulate token usage from ALL attempts (including retries)
            for resp in all_responses:
                resp_usage = getattr(resp, "usage_metadata", None)
                if resp_usage is None:
                    # Ollama locally doesn't always report token usage.
                    # In production against Gemini this will never be None.
                    continue
                total_input_tokens  += resp_usage.get("input_tokens", 0)
                total_output_tokens += resp_usage.get("output_tokens", 0)

            # Extract fields from LLM response
            clean_heading      = parsed.get("section_heading", heading)
            condensed_content  = parsed.get("condensed_content", "")
            section_summary    = parsed.get("section_summary", "")

            notes_json[clean_heading] = condensed_content
            running_summary.append(section_summary)

            print(f"[notes_chain] ✓ Section '{clean_heading[:50]}' condensed.")

        except RuntimeError as e:
            # All retry attempts exhausted — last resort placeholder
            print(
                f"[notes_chain] ✗ Section '{heading[:50]}' failed after "
                f"{_MAX_SECTION_RETRIES} retries: {e}. Inserting placeholder."
            )
            notes_json[heading] = "[Content could not be generated for this section.]"
            running_summary.append(f"Section '{heading}' could not be processed.")

    # ── Step 3: Build PDF ─────────────────────────────────────────────────────
    pdf_path = _build_pdf(
        notes_json=notes_json,
        student_id=student_id,
        current_topic=current_topic,
    )

    # ── Step 4: Re-ingest condensed PDF into Chroma ───────────────────────────

    # Keep this idle for now

    # _reingest_pdf(
    #     pdf_path=pdf_path,
    #     student_id=student_id,
    #     current_topic=current_topic,
    #     course=course,
    #     embedder=embedder,
    #     store=store,
    # )

    # ── Step 5: Build synthetic response for token_guard ─────────────────────
    # token_guard expects a response object with usage_metadata.
    # Since this chain makes multiple LLM calls, we accumulate totals
    # and return a lightweight wrapper so token_guard deducts correctly.

    total_tokens = total_input_tokens + total_output_tokens

    print(f"\n[notes_chain] ═══ Notes generation complete ═══")
    print(f"[notes_chain] PDF: {pdf_path}")
    print(f"[notes_chain] Total tokens — in: {total_input_tokens:,} | out: {total_output_tokens:,} | total: {total_tokens:,}")

    class _SyntheticResponse:
        """
        Lightweight response wrapper returned to token_guard.
        Carries the PDF path as content and accumulated token usage.
        """
        def __init__(self, pdf_path: Path, input_t: int, output_t: int, total_t: int):
            self.content = str(pdf_path)
            self.usage_metadata = {
                "input_tokens":  input_t,
                "output_tokens": output_t,
                "total_tokens":  total_t,
            }

    return _SyntheticResponse(
        pdf_path=pdf_path,
        input_t=total_input_tokens,
        output_t=total_output_tokens,
        total_t=total_tokens,
    )

