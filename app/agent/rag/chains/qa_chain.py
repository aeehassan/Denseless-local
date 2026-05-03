"""
qa_chain.py
===========
Chat-aware, rate-limit-aware QA chain for CogniLearn.

Answers student questions by retrieving context from either the original
ingested PDF or the student's condensed notes (controlled by a feature),
then calling the LLM with the retrieved context and a sliding window of
conversation history.

When no relevant chunks are found, the LLM still runs — it answers from
general knowledge and explicitly tells the student the answer is not
sourced from their study material.

Rate limiting mirrors notes_chain.py:
    USE_GEMINI = False  →  local Ollama, no delays, rate-limit handling off.
    USE_GEMINI = True   →  Gemini API, RPM backoff active.

History is managed entirely outside this chain. See router responsibilities
in run_qa_chain's docstring.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from json_repair import repair_json
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import Chroma

from app.agent.rag.retrieval.retriever import get_semantic_chunks
from app.services.token_service import token_guard
from app.agent.rag.prompts import (
    REFORMULATION_PROMPT,
    QA_PROMPT_MAP,
    GROUNDING_INSTRUCTION_CONTEXT,
    GROUNDING_INSTRUCTION_GENERAL,
)

# Expose httpx / Ollama request logs in the output window
logging.basicConfig(level=logging.INFO)


# ── Runtime target ─────────────────────────────────────────────────────────────
# Flip this manually before running. Controls inter-call delays and whether
# rate-limit handling is active at all.
USE_GEMINI: bool = False

# Seconds to sleep between the reformulation pre-call and the main LLM call
# when USE_GEMINI is True. Keeps throughput under the free-tier RPM ceiling.
_REQUEST_DELAY_SECONDS: int = 5

# Backoff schedule (seconds) for RPM errors.
# 60 s guarantees the per-minute window resets before the final retry.
_RPM_BACKOFF_SECONDS: list[int] = [15, 30, 60]

# ── Chain constants ────────────────────────────────────────────────────────────
# Parse-failure retries for the main LLM call.
_MAX_QA_RETRIES: int = 3

# Max outer (rate-limit) retries. Mirrors len(_RPM_BACKOFF_SECONDS).
_MAX_RATE_LIMIT_RETRIES: int = 3

# How many exchanges (user + assistant pairs) the LLM sees in its prompt.
# Older turns are silently dropped to keep token cost bounded.
# The router always writes the full history to disk — only the LLM window
# is capped here.
_MAX_HISTORY_TURNS: int = 4  # 4 pairs = 8 messages


# ══════════════════════════════════════════════════════════════════════════════
# Custom exception (mirrors notes_chain.py)
# ══════════════════════════════════════════════════════════════════════════════


class RateLimitExhaustedError(RuntimeError):
    """
    Raised when API rate limiting cannot be recovered from.

    kind:
        "rpm" — per-minute limit hit, all backoff retries exhausted.
        "rpd" — daily quota exhausted. Caller should terminate and return
                 whatever partial result is available.

    Example:
        raise RateLimitExhaustedError("Daily quota hit.", kind="rpd")
    """

    def __init__(self, message: str, kind: str):
        super().__init__(message)
        self.kind = kind  # "rpm" | "rpd"


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════


def _classify_rate_error(e: Exception) -> str:
    """
    Inspects an exception and classifies it as RPM, RPD, or unrelated.

    Classification order:
        1. HTTP 429 on exception or __cause__ → check message for RPD keywords,
           else default to "rpm".
        2. RPD keywords in message → "rpd"
        3. RPM keywords in message → "rpm"
        4. Anything else          → "other"

    Args:
        e: Any exception raised during an LLM call.

    Returns:
        "rpm"   — per-minute limit, recoverable with backoff.
        "rpd"   — daily quota, unrecoverable in this session.
        "other" — unrelated error.

    Examples:
        >>> _classify_rate_error(Exception("429 Resource Exhausted"))
        "rpm"
        >>> _classify_rate_error(Exception("Quota exceeded for quota metric"))
        "rpd"
        >>> _classify_rate_error(Exception("Connection timeout"))
        "other"
    """
    msg = str(e).lower()

    status = getattr(e, "status_code", None) or getattr(
        getattr(e, "__cause__", None), "status_code", None
    )
    if status == 429:
        if any(k in msg for k in ("quota exceeded", "daily limit", "per day")):
            return "rpd"
        return "rpm"

    if any(k in msg for k in ("quota exceeded", "daily limit", "per day", "billing")):
        return "rpd"

    if any(k in msg for k in ("rate limit", "resource exhausted", "too many requests")):
        return "rpm"

    return "other"


def _format_history(conversation_history: List[Dict]) -> str:
    """
    Formats the windowed conversation history into a plain transcript string
    for injection into the LLM prompt.

    Takes the last (_MAX_HISTORY_TURNS * 2) messages from the list so the
    LLM always sees at most _MAX_HISTORY_TURNS exchanges. Older messages
    are silently dropped — the full history is always preserved on disk by
    the router.

    Args:
        conversation_history: Full list of message dicts, each with
                              "role" ("user" | "assistant") and "content".

    Returns:
        Formatted transcript string, or a placeholder if history is empty.

    Examples:
        >>> _format_history([
        ...     {"role": "user",      "content": "What is pipelining?"},
        ...     {"role": "assistant", "content": "Pipelining is..."},
        ... ])
        "User: What is pipelining?\\nAssistant: Pipelining is..."

        >>> _format_history([])
        "No prior conversation."
    """
    if not conversation_history:
        return "No prior conversation."

    # Window: last _MAX_HISTORY_TURNS exchanges (pairs of user + assistant)
    window = conversation_history[-(_MAX_HISTORY_TURNS * 2) :]

    lines = []
    for msg in window:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _format_chunks(chunks: List[Document]) -> str:
    """
    Formats retrieved Document chunks into a numbered context string for
    injection into the LLM prompt.

    Page labeling rules:
        - A page label ("— [N]") is appended only to the LAST chunk in a
          consecutive run of chunks sharing the same page number.
        - If the same page appears again later in a non-consecutive position,
          it is labeled again — because it represents a different retrieval
          context the LLM should anchor separately.
        - Chunks with no page_number in metadata are labeled "— Page ?" only
          if they are the last in a consecutive run of page-unknown chunks.

    This preserves retrieval order so the LLM sees the most relevant chunks
    first, while giving the student accurate page anchors without cluttering
    every line.

    Args:
        chunks: List of Document objects returned by get_semantic_chunks(),
                in retrieval (relevance) order.

    Returns:
        Formatted context string with page labels at run boundaries.
        Returns empty string if chunks is empty — caller handles no-chunks
        path separately.

    Example:
        chunks pages in order: [7, 7, 4, 12, 12, 4]

        [Chunk 1] Section: Pipeline Hazards
        Raw content of chunk 1...

        [Chunk 2] Section: Pipeline Hazards — [7]
        Raw content of chunk 2...

        [Chunk 3] Section: RISC Architecture — [4]
        Raw content of chunk 3...

        [Chunk 4] Section: Cache Memory
        Raw content of chunk 4...

        [Chunk 5] Section: Cache Memory — [12]
        Raw content of chunk 5...

        [Chunk 6] Section: RISC Architecture — [4]
        Raw content of chunk 6...
    """
    if not chunks:
        return ""

    parts = []
    for i, doc in enumerate(chunks):
        section = doc.metadata.get("section_heading", "Unknown Section")
        cur_page = doc.metadata.get("page_number", None)
        
        # Check if this chunk is the last in a consecutive same-page run.
        # A run ends when the next chunk has a different page, or this is
        # the final chunk in the list.
        is_last_in_run = (
            i == len(chunks) - 1
            or chunks[i + 1].metadata.get("page_number", None) != cur_page
        )

        if is_last_in_run:
            label = f"{section} — [{cur_page if cur_page is not None else '?'}]"
        else:
            label = section

        parts.append(f"[Chunk {i + 1}] Section: {label}\n{doc.page_content.strip()}")

    return "\n\n".join(parts)


def _extract_source_pages(chunks: List[Document]) -> List[int]:
    """
    Extracts page numbers from chunk metadata in retrieval order,
    deduplicating while preserving the order of first appearance.

    Retrieval order is preserved intentionally — the retriever ranks chunks
    by relevance, so the first page in this list is the most relevant one.
    Sorting is deliberately avoided to keep that signal intact for the
    frontend (e.g. "Sources: p.7, p.4, p.12" tells the student where the
    bulk of the answer came from).

    Non-consecutive duplicate pages (e.g. p.4 appearing at positions 3 and 6)
    are collapsed to a single entry since source_pages is a flat reference
    list — the page labeling in _format_chunks already handles the per-chunk
    context distinction for the LLM.

    Chunks with no page_number in metadata are silently skipped — they still
    appear in the formatted context via _format_chunks labeled as "Page ?",
    but cannot be referenced meaningfully in source_pages.

    Args:
        chunks: List of Document objects returned by get_semantic_chunks(),
                in retrieval (relevance) order.

    Returns:
        Deduplicated list of page numbers as integers, in retrieval order.
        Returns empty list if no chunks have page_number metadata.

    Example:
        chunks pages in order: [7, 7, 4, 12, 12, 4]
        returns:               [7, 4, 12]
        # p.4 appears twice but is deduplicated to its first-seen position
    """
    seen  : set       = set()
    pages : List[int] = []

    for doc in chunks:
        page = doc.metadata.get("page_number", None)
        if page is None or page in seen:
            continue
        seen.add(page)
        pages.append(int(page))

    return pages


def _reformulate_question(
    llm: BaseChatModel,
    question: str,
    conversation_history: List[Dict],
) -> tuple[str, Any]:
    """
    Rewrites a follow-up question as a standalone retrieval query using the
    conversation history as context.

    This is a lightweight pre-call that resolves pronouns and references
    (e.g. "What are its hazards?" → "What are the hazards of pipelining?")
    so the retriever gets a semantically complete query.

    Only called when conversation_history is non-empty. When history is
    empty the original question is used as-is and this function is skipped.

    Token usage from this call is accumulated by the caller into the same
    total_input_tokens / total_output_tokens counters so @token_guard
    deducts correctly.

    Args:
        llm:                  Initialised LLM instance.
        question:             The student's raw follow-up question.
        conversation_history: Full conversation history list (not windowed —
                              the prompt template windows internally).

    Returns:
        Tuple of (reformulated_question_string, llm_response_object).
        The response object is returned so the caller can extract
        usage_metadata for token accounting.

    Raises:
        RuntimeError: If the LLM call fails. Caller catches this and falls
                      back to the original question — non-fatal.

    Examples:
        Input:
            history:  [{"role": "user", "content": "What is pipelining?"},
                       {"role": "assistant", "content": "Pipelining is..."}]
            question: "What causes stalls in it?"
        Output:
            ("What causes stalls in pipelining?", <response object>)

        Input:
            history:  [...]
            question: "Explain RISC architecture"
        Output:
            ("Explain RISC architecture", <response object>)
            # Self-contained questions returned unchanged
    """
    history_text = _format_history(conversation_history)

    prompt = REFORMULATION_PROMPT.format(
        history=history_text,
        question=question,
    )

    try:
        response = llm.invoke(prompt)
        reformulated = response.content.strip()

        if not reformulated:
            raise ValueError("LLM returned empty reformulation.")

        print(f"[qa_chain] ✓ Question reformulated: '{reformulated[:80]}'")
        return reformulated, response

    except Exception as e:
        raise RuntimeError(
            f"Reformulation pre-call failed: {e}. "
            f"Original question will be used for retrieval."
        ) from e


def _call_qa_llm(
    llm: BaseChatModel,
    question: str,
    conversation_history: List[Dict],
    chunks: List[Document],
    learning_pace: str,
    current_topic: str,
) -> tuple[dict, list]:
    """
    Runs the main QA LLM call for a student question, returning a structured
    answer dict.

    Handles two retrieval states:
        - Chunks found:    Grounded answer from study material.
        - No chunks found: General knowledge answer with explicit disclaimer.

    Retry architecture (two nested loops — mirrors notes_chain._call_section_llm):

        Outer loop — rate-limit retries (up to _MAX_RATE_LIMIT_RETRIES):
            Only active when USE_GEMINI is True.
            RPM errors: backs off [15, 30, 60] s then retries.
            RPD errors: raises RateLimitExhaustedError(kind="rpd") immediately.

        Inner loop — parse-failure retries (up to _MAX_QA_RETRIES):
            JsonOutputParser primary → json_repair fallback.
            All parse-failure attempts accumulated for token accounting.

    Token usage accumulated across ALL attempts (both loops) and returned
    to the caller for aggregation into the synthetic response.

    Args:
        llm:                  Initialised LLM instance.
        question:             The student's question (original, not reformulated).
        conversation_history: Full history list — windowed internally.
        chunks:               Retrieved Document objects. Empty list triggers
                              the general-knowledge path.
        learning_pace:        "slow" | "average" | "fast"
        current_topic:        The topic the student is currently studying (e.g.
                              "Computer Architecture"). Injected into the grounding
                              instruction only on the no-chunks path so the LLM has
                              directional context when answering from general knowledge.

    Returns:
        Tuple of (parsed_dict, all_responses):

        parsed_dict example (grounded):
        {
            "answer":          "Pipelining hazards occur when...",
            "source_pages": [1, 2],
            "confidence":      "high"
        }

        parsed_dict example (no chunks):
        {
            "answer":          "Note: this is not from your study material.
                                From general knowledge: pipelining hazards...",
            "source_pages": [],
            "confidence":      "general"
        }

        all_responses: list of all LangChain response objects from every
                       attempt, for token accounting by the caller.

    Raises:
        ValueError:               If learning_pace is not recognised.
        RateLimitExhaustedError:  If RPD quota is hit, or RPM retries exhausted.
        RuntimeError:             If all parse retries are exhausted for a
                                  non-rate-limit reason.
    """
    if learning_pace not in QA_PROMPT_MAP:
        raise ValueError(
            f"Unrecognised learning_pace: '{learning_pace}'. "
            f"Valid values: {list(QA_PROMPT_MAP.keys())}"
        )

    # ── Build prompt components ───────────────────────────────────────────────
    has_chunks = bool(chunks)

    context_str = _format_chunks(chunks) if has_chunks else ""
    history_str = _format_history(conversation_history)
    grounding_instruction = (
        GROUNDING_INSTRUCTION_CONTEXT if has_chunks else GROUNDING_INSTRUCTION_GENERAL.format(current_topic=current_topic)
    )

    prompt_template = QA_PROMPT_MAP[learning_pace]

    prompt = prompt_template.format(
        grounding_instruction=grounding_instruction,
        context=context_str,
        history=history_str,
        question=question,
    )

    parser = JsonOutputParser()
    all_responses = []
    last_error = None

    # ── Outer loop: rate-limit retries ───────────────────────────────────────
    max_rate_attempts = _MAX_RATE_LIMIT_RETRIES if USE_GEMINI else 1

    for rate_attempt in range(1, max_rate_attempts + 1):
        # ── Inner loop: parse-failure retries ────────────────────────────────
        for attempt in range(1, _MAX_QA_RETRIES + 1):
            raw = ""

            try:
                response = llm.invoke(prompt)
                all_responses.append(response)
                raw = response.content.strip()

                # Primary parse: JsonOutputParser (fence stripping + json.loads)
                try:
                    parsed = parser.parse(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f"Expected dict, got {type(parsed).__name__}: "
                            f"{str(parsed)[:200]}"
                        )
                    if attempt > 1:
                        print(f"[qa_chain] ✓ Parse retry {attempt} succeeded.")
                    return parsed, all_responses

                # Fallback: json_repair handles literal newlines, missing
                # delimiters, truncated strings, etc.
                except Exception:
                    print(
                        f"[qa_chain] ⚠ JsonOutputParser failed "
                        f"(attempt {attempt}) — attempting json_repair."
                    )
                    repaired = repair_json(raw)
                    parsed = json.loads(repaired)
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f"json_repair produced {type(parsed).__name__} "
                            f"instead of dict: {str(parsed)[:200]}"
                        )
                    if attempt > 1:
                        print(f"[qa_chain] ✓ json_repair retry {attempt} succeeded.")
                    return parsed, all_responses

            except Exception as e:
                # ── Rate-limit detection ──────────────────────────────────────
                if USE_GEMINI:
                    kind = _classify_rate_error(e)

                    if kind == "rpd":
                        print("[qa_chain] ✗ Daily quota exhausted.")
                        raise RateLimitExhaustedError(
                            "Daily Gemini quota exhausted during QA call.",
                            kind="rpd",
                        )

                    if kind == "rpm":
                        print(
                            f"[qa_chain] ⚠ RPM limit hit "
                            f"(rate_attempt {rate_attempt}/{max_rate_attempts})."
                        )
                        last_error = e
                        break  # exit inner loop → outer loop applies backoff

                # Non-rate-limit parse failure — standard inner retry
                last_error = e
                if attempt < _MAX_QA_RETRIES:
                    print(
                        f"[qa_chain] ⚠ Parse attempt {attempt}/{_MAX_QA_RETRIES} "
                        f"failed: {e}. Retrying..."
                    )
                else:
                    print(
                        f"[qa_chain] ✗ All {_MAX_QA_RETRIES} parse attempts "
                        f"exhausted. Last error: {e}"
                    )

        # ── RPM backoff ───────────────────────────────────────────────────────
        if rate_attempt < max_rate_attempts:
            wait = _RPM_BACKOFF_SECONDS[rate_attempt - 1]
            print(
                f"[qa_chain] ⏳ Backing off {wait}s before rate-limit "
                f"retry {rate_attempt + 1}/{max_rate_attempts}."
            )
            time.sleep(wait)
        else:
            raise RateLimitExhaustedError(
                f"RPM retries exhausted after {max_rate_attempts} attempts. "
                f"Last error: {last_error}",
                kind="rpm",
            )

    # Parse retries exhausted (non-rate-limit path)
    raise RuntimeError(
        f"QA LLM call failed after {_MAX_QA_RETRIES} parse attempts. "
        f"Last raw output: '{raw[:200]}' | Last error: {last_error}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════


@token_guard
def run_qa_chain(
    student_id: str,
    question: str,
    conversation_history: List[Dict],
    from_condensed_notes: bool,
    current_topic: str,
    store: Chroma,
    llm: BaseChatModel,
    learning_pace: str,
) -> Any:
    """
    Answer a student question using retrieved context from either the original
    ingested PDF or the student's condensed notes, with full conversation
    history awareness.

    This is a read-only chain — it writes nothing to disk or Chroma. All
    state management (history persistence, session tracking) is the router's
    responsibility (see Router Responsibilities below).

    ── Source selection ─────────────────────────────────────────────────────────
        from_condensed_notes = False  →  retrieves from the original ingested PDF
                               (default student experience).
        from_condensed_notes = True   →  retrieves only from condensed notes chunks,
                               identified by metadata filter {"condensed": True}.
                               Requires notes to have been generated and
                               re-ingested via notes_chain Step 4 first.

    ── Conversation awareness ───────────────────────────────────────────────────
        If conversation_history is non-empty, a reformulation pre-call rewrites
        the question as a standalone retrieval query (e.g. "What are its hazards?"
        → "What are the hazards of pipelining?"). The reformulated query is used
        only for retrieval — the original question is always what the LLM answers.

        The LLM prompt includes a sliding window of the last _MAX_HISTORY_TURNS
        exchanges (default 4 pairs = 8 messages). Older messages are dropped from
        the prompt only — they are always preserved in the file on disk.

    ── No-chunks path ───────────────────────────────────────────────────────────
        If retrieval returns no chunks, the LLM still runs. It answers from
        general knowledge and explicitly tells the student the response is not
        sourced from their study material. confidence is set to "general" in
        the response so the frontend can render a disclaimer.

    ── Rate limiting ────────────────────────────────────────────────────────────
        Controlled by the module-level USE_GEMINI flag:
            USE_GEMINI = False  →  local Ollama, no delays, rate-limit handling off.
            USE_GEMINI = True   →  5 s delay between reformulation and main call.
                                   RPM errors retried with [15, 30, 60] s backoff.
                                   RPD errors raise RateLimitExhaustedError and
                                   propagate to the router.

    ── Token tracking ───────────────────────────────────────────────────────────
        Tokens from the reformulation pre-call and the main LLM call are
        accumulated and returned in usage_metadata so @token_guard deducts
        correctly from the student's balance.

    ── Router responsibilities ───────────────────────────────────────────────────
        The chain is fully stateless. All session and history management lives
        in the router. The expected router flow is:

        Request body:  POST /chat  { question, use_notes, session_id? }

        1. If no session_id in request:
               session_id = uuid4()              # new session
               create data/chat_history/{student_id}_{session_id}.json  → []

        2. Load full history from:
               data/chat_history/{student_id}_{session_id}.json

        3. Pass last (_MAX_HISTORY_TURNS * 2) messages to run_qa_chain
           as conversation_history.

        4. Call run_qa_chain(...) → result.

        5. Append to the history file:
               {"role": "user",      "content": question}
               {"role": "assistant", "content": answer}

        6. Return to frontend:
               { answer, source_sections, confidence, session_id,
                 full_display_history }

        Frontend stores session_id in localStorage so it survives page refresh.
        On refresh, the frontend sends the stored session_id → router loads the
        file → full history is restored for display.

        Future Supabase swap: replace file read with SELECT and file write with
        INSERT on the chat_history table. The chain itself does not change.

    Args:
        student_id:           Unique identifier for the student. Required first
                              by @token_guard — do not reorder.
        question:             The student's raw question. Max 500 chars.
        conversation_history: Windowed slice of prior exchanges, each a dict
                              with "role" and "content" keys. Loaded and
                              windowed by the router before this call.
                              Pass empty list [] for the first message.
        from_notes:           If True, retrieves from condensed notes only
                              (metadata filter condensed=True).
                              If False, retrieves from original PDF material.
        current_topic:        The topic the student is currently studying (e.g.
                              "Computer Architecture"). Passed through to the LLM
                              only when retrieval returns no chunks, giving it
                              directional context for a general knowledge answer.
        store:                Student-scoped Chroma vector store.
        llm:                  Initialised LangChain LLM instance.
        learning_pace:        "slow" | "average" | "fast". Controls prompt
                              verbosity and explanation depth.

    Returns:
        Synthetic response object with:
            .content        → JSON string of the answer dict:
                {
                    "answer":          "Pipelining hazards occur when...",
                    "source_sections": ["Pipeline Hazards", "RISC Architecture"],
                    "confidence":      "high" | "medium" | "low" | "general"
                }
            .usage_metadata → Accumulated token usage:
                {
                    "input_tokens":  320,
                    "output_tokens": 180,
                    "total_tokens":  500
                }
    
    confidence field:
        LLM self-assessed measure of how well the retrieved context supported
        the answer. Reflects retrieval quality, not factual correctness.

        "high"    — context directly and fully addressed the question.
        "medium"  — context was related but only partially covered the question;
                    answer may involve some inference.
        "low"     — context was sparse or loosely related; answer is best effort.
        "general" — no chunks retrieved; answer sourced entirely from the LLM's
                    general knowledge, not the student's study material.

        The frontend should use this to render appropriate trust indicators
        alongside the answer (e.g. a warning when confidence is "low" or "general").

    Raises:
        ValueError:              If question is empty, exceeds 500 chars,
                                 learning_pace is invalid, or
                                 conversation_history is malformed.
        RateLimitExhaustedError: If daily Gemini quota is exhausted.
                                 Router should return a 429 response with
                                 a user-friendly message.
        RuntimeError:            If all LLM parse retries are exhausted.

    Example call from router:
        history_window = full_history[-(_MAX_HISTORY_TURNS * 2):]
        result = run_qa_chain(
            student_id           = current_user["id"],
            question             = "What causes pipeline stalls?",
            conversation_history = history_window,
            from_notes           = False,
            current_topic        = "Computer Architecture",
            store                = store,
            llm                  = llm,
            learning_pace        = profile["learning_pace"],
        )
        answer_dict = json.loads(result.content)
        answer      = answer_dict["answer"]

    Example result.content:
        '{
            "answer":          "Pipeline stalls occur when an instruction...",
            "source_sections": ["Pipeline Hazards"],
            "confidence":      "high"
        }'

    Example result.usage_metadata:
        {
            "input_tokens":  320,
            "output_tokens": 180,
            "total_tokens":  500
        }
    """
    # ── Step 0: Input validation ──────────────────────────────────────────────
    if not question or not question.strip():
        raise ValueError("question cannot be empty or whitespace.")

    if len(question) > 500:
        raise ValueError(
            f"question exceeds 500 character limit "
            f"(got {len(question)} chars). Truncate before calling run_qa_chain()."
        )

    if learning_pace not in QA_PROMPT_MAP:
        raise ValueError(
            f"Invalid learning_pace: '{learning_pace}'. "
            f"Valid values: {list(QA_PROMPT_MAP.keys())}"
        )

    if not isinstance(conversation_history, list):
        raise ValueError(
            f"conversation_history must be a list of dicts, "
            f"got {type(conversation_history).__name__}."
        )

    for i, msg in enumerate(conversation_history):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError(
                f"conversation_history[{i}] must be a dict with 'role' and "
                f"'content' keys. Got: {msg}"
            )

    print(f"\n[qa_chain] ═══ Starting QA chain ═══")
    print(f"[qa_chain] Mode: {'Gemini API' if USE_GEMINI else 'Local Ollama'}")
    print(
        f"[qa_chain] Student: {student_id} | Pace: {learning_pace} | "
        f"Source: {'notes' if from_condensed_notes else 'original PDF'}"
    )
    print(f"[qa_chain] Question: '{question[:80]}'")
    print(
        f"[qa_chain] History turns in context: "
        f"{min(len(conversation_history), _MAX_HISTORY_TURNS * 2) // 2}"
    )

    total_input_tokens = 0
    total_output_tokens = 0

    # ── Step 0.5: Question reformulation (only when history is non-empty) ─────
    retrieval_query = question  # default: use original question

    if conversation_history:
        try:
            retrieval_query, reformulation_response = _reformulate_question(
                llm=llm,
                question=question,
                conversation_history=conversation_history,
            )
            ref_usage = getattr(reformulation_response, "usage_metadata", {}) or {}
            total_input_tokens += ref_usage.get("input_tokens", 0)
            total_output_tokens += ref_usage.get("output_tokens", 0)

        except RuntimeError as e:
            # Non-fatal — fall back to original question for retrieval
            print(f"[qa_chain] ⚠ Reformulation failed ({e}) — using original question.")
            retrieval_query = question

        # Inter-call delay when hitting Gemini to stay under RPM ceiling
        if USE_GEMINI:
            print(f"[qa_chain] ⏳ Throttle: sleeping {_REQUEST_DELAY_SECONDS}s.")
            time.sleep(_REQUEST_DELAY_SECONDS)

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    try:
        chunks = get_semantic_chunks(
            store=store,
            query=retrieval_query,
            score_threshold=0.4,
            condensed=from_condensed_notes,
        )
    except Exception as e:
        raise RuntimeError(
            f"Retrieval failed for query '{retrieval_query[:80]}': {e}"
        ) from e

    if chunks:
        print(
            f"[qa_chain] ✓ Retrieved {len(chunks)} chunk(s) "
            f"from {'notes' if from_condensed_notes else 'original PDF'}."
        )
    else:
        print(f"[qa_chain] ⚠ No chunks found — LLM will answer from general knowledge.")

    # ── Step 2 + 3: Build prompt + LLM call with retry ───────────────────────
    try:
        parsed, all_responses = _call_qa_llm(
            llm=llm,
            question=question,
            conversation_history=conversation_history,
            chunks=chunks,
            learning_pace=learning_pace,
            current_topic=current_topic
        )

        # Inject source_sections from metadata (ground truth) if chunks were found.
        # Overrides whatever the LLM put there — metadata is more reliable.
        if chunks:
            parsed["source_pages"] = _extract_source_pages(chunks)

    except RateLimitExhaustedError:
        # Propagate — router handles user-facing error response
        raise

    except RuntimeError as e:
        raise RuntimeError(
            f"QA chain LLM call failed for question '{question[:80]}': {e}"
        ) from e

    # Accumulate token usage from all LLM attempts
    for resp in all_responses:
        resp_usage = getattr(resp, "usage_metadata", None)
        if resp_usage is None:
            # Ollama locally doesn't always report usage.
            continue
        total_input_tokens += resp_usage.get("input_tokens", 0)
        total_output_tokens += resp_usage.get("output_tokens", 0)


    # ── Step 4: Build synthetic response for token_guard ─────────────────────
    total_tokens = total_input_tokens + total_output_tokens

    print(f"\n[qa_chain] ═══ QA chain complete ═══")
    print(
        f"[qa_chain] Confidence: {parsed.get('confidence', 'unknown')} | "
        f"Sources: {parsed.get('source_sections', [])}"
    )
    print(
        f"[qa_chain] Total tokens — "
        f"in: {total_input_tokens:,} | out: {total_output_tokens:,} | "
        f"total: {total_tokens:,}"
    )

    class _SyntheticResponse:
        """
        Lightweight response wrapper returned to @token_guard.
        content holds the answer dict as a JSON string.
        usage_metadata carries accumulated token counts from both the
        reformulation pre-call and the main LLM call.
        """

        def __init__(self, content: str, input_t: int, output_t: int, total_t: int):
            self.content = content
            self.usage_metadata = {
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": total_t,
            }

    return _SyntheticResponse(
        content=json.dumps(parsed),
        input_t=total_input_tokens,
        output_t=total_output_tokens,
        total_t=total_tokens,
    )
