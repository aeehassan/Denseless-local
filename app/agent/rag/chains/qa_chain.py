"""
qa_chain.py
===========
Chat-aware, rate-limit-aware QA chain for CogniLearn.

Memory architecture:
    STM — ConversationSummaryBufferMemory (within-session context compression)
    LTM — LangMem semantic memory (cross-session concept + misconception retention)

Entry points:
    list_conversations(student_id)          → list of saved conversation names
    run_qa_chain(...)                       → main QA entry point (@token_guard)
    view_ltm(student_id)                    → human-readable LTM dump string

Storage layout:
    data/chat_history/{student_id}/{convo_name}.json   ← full history per conversation
    data/ltm/{student_id}/memories.db                  ← LangMem SQLite persistence

Rate limiting mirrors all other chains:
    USE_GEMINI = False  →  local Ollama, no delays, rate-limit handling off.
    USE_GEMINI = True   →  Gemini API, RPM backoff active.

NOTE on LangMem imports:
    Assumed package: pip install langmem
    If your installed version differs, adjust the two imports marked [LANGMEM IMPORT]
    below. All LangMem calls are isolated to _init_ltm_store(), _query_ltm(),
    and _update_ltm() so fixes are localised.
"""

import json
import textwrap
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from json_repair import repair_json
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.memory import ConversationSummaryBufferMemory  # STM

# ── [LANGMEM IMPORT] ──────────────────────────────────────────────────────────
# Adjust these two lines if your langmem version exposes a different API.
from langmem import create_memory_manager  # LTM manager
from langgraph.store.memory import InMemoryStore  # LangGraph's store
# ─────────────────────────────────────────────────────────────────────────────

from app.agent.rag.retrieval.retriever import get_semantic_chunks
from app.services.token_service import token_guard
from app.agent.rag.prompts import (
    REFORMULATION_PROMPT,
    QA_PROMPT_MAP,
    GROUNDING_INSTRUCTION_CONTEXT,
    GROUNDING_INSTRUCTION_GENERAL,
)

logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

USE_GEMINI: bool = False

_REQUEST_DELAY_SECONDS: int = 5
_RPM_BACKOFF_SECONDS: list[int] = [15, 30, 60]
_MAX_RATE_LIMIT_RETRIES: int = 3
_MAX_QA_RETRIES: int = 3


# Max token buffer before ConversationSummaryBufferMemory starts summarising.
_STM_MAX_TOKEN_LIMIT: int = 600

CHAT_HISTORY_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "data" / "chat_history"
)
LTM_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "ltm"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTION
# ─────────────────────────────────────────────────────────────────────────────


class RateLimitExhaustedError(RuntimeError):
    """
    Raised when all rate-limit retries are exhausted.

    Attributes:
        kind (str): "rpm" if requests-per-minute limit hit,
                    "rpd" if daily quota exceeded.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


# ─────────────────────────────────────────────────────────────────────────────
# _SyntheticResponse
# ─────────────────────────────────────────────────────────────────────────────


class _SyntheticResponse:
    """
    Wraps qa_chain output for token_guard compatibility.

    Attributes:
        content (dict): Keys — answer, source_pages, convo_name, grounded, ltm_usage.
        usage_metadata (dict): Accumulated token counts for QA + STM + reformulation.
                               LangMem's internal calls are reported separately in
                               content["ltm_usage"] for the router to deduct manually.
    """

    def __init__(self, content: dict, usage_metadata: dict):
        self.content = content
        self.usage_metadata = usage_metadata


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS — rate limit + token utils
# ─────────────────────────────────────────────────────────────────────────────


def _classify_rate_error(e: Exception) -> str:
    """
    Classifies a rate-limit exception by inspecting its message string.

    Returns:
        "rpd"   — daily quota exceeded.
        "rpm"   — requests-per-minute limit hit.
        "other" — unrecognised.
    """
    msg = str(e).lower()
    if "daily" in msg or "quota" in msg:
        return "rpd"
    if "429" in msg or "rate" in msg or "resource_exhausted" in msg:
        return "rpm"
    return "other"


def _accumulate_tokens(base: dict, new: dict) -> dict:
    """
    Merges two usage_metadata dicts by summing integer token counts.
    Non-integer values (e.g. nested dicts like input_token_details) are skipped.

    Example:
        >>> _accumulate_tokens({"input_tokens": 300}, {"input_tokens": 210, "input_token_details": {"cache_read": 0}})
        {"input_tokens": 510}
    """
    result = dict(base)
    for key, value in new.items():
        if not isinstance(value, int):
            continue
        result[key] = result.get(key, 0) + value
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS — formatting
# ─────────────────────────────────────────────────────────────────────────────


def _format_history(conversation_history: List[Dict]) -> str:
    """
    Formats a conversation history list into a plain string for prompt injection.

    Args:
        conversation_history: List of {"role": "human"|"assistant", "content": str}.

    Returns:
        Multi-line string with labelled turns.

    Example:
        >>> _format_history([{"role": "human", "content": "What is recursion?"}])
        "Student: What is recursion?"
    """
    lines = []
    for turn in conversation_history:
        role = "Student" if turn["role"] == "human" else "Assistant"
        content = turn.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_chunks(chunks: List[Document]) -> str:
    """
    Formats retrieved chunks into a numbered context block for the QA prompt.

    Args:
        chunks: List of LangChain Document objects.

    Returns:
        Numbered string of chunk contents, or empty string if no chunks.

    Example:
        >>> _format_chunks([Document(page_content="Merge sort divides arrays.")])
        "1. Merge sort divides arrays."
    """
    if not chunks:
        return ""
    return "\n".join(f"{i + 1}. {doc.page_content}" for i, doc in enumerate(chunks))


def _extract_source_pages(chunks: List[Document]) -> List[int]:
    """
    Extracts and deduplicates page numbers from chunk metadata.

    Args:
        chunks: List of LangChain Document objects with metadata.

    Returns:
        Sorted list of unique page numbers. Empty list if none found.

    Example:
        >>> _extract_source_pages([Document(metadata={"page_number": 4})])
        [4]
    """
    pages = set()
    for doc in chunks:
        page = doc.metadata.get("page_number")
        if isinstance(page, int):
            pages.add(page)
    return sorted(pages)


def _sanitise_convo_name(convo_name: str) -> str:
    """
    Strips characters unsafe for filenames from the conversation name.

    Args:
        convo_name: Raw conversation name from the caller.

    Returns:
        Sanitised string safe for use as a .json filename.

    Example:
        >>> _sanitise_convo_name("Week 3: Sorting/Searching")
        "Week 3_ Sorting_Searching"
    """
    unsafe = r'\/:*?"<>|'
    result = convo_name
    for char in unsafe:
        result = result.replace(char, "_")
    return result.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS — STM
# ─────────────────────────────────────────────────────────────────────────────


def _update_stm_summary(
    old_summary: str,
    question: str,
    answer: str,
    llm: BaseChatModel,
) -> str:
    """
    Produces an updated running summary by folding the latest exchange
    into the existing summary using the LLM.

    If there is no prior summary, the first exchange is summarised from scratch.

    Args:
        old_summary: The current running summary string. Empty on first turn.
        question:    The student's question for this turn.
        answer:      The assistant's response for this turn.
        llm:         Language model used for summarisation.

    Returns:
        Updated summary string representing the full conversation so far.

    Raises:
        RuntimeError: If summarisation fails unexpectedly.

    Example:
        >>> updated = _update_stm_summary(
        ...     "Student asked about merge sort.",
        ...     "What about quicksort?",
        ...     "Quicksort uses a pivot...",
        ...     llm
        ... )
        "Student asked about merge sort. They then asked about quicksort —
         the assistant explained pivot selection and average-case O(n log n) complexity."
    """
    _UPDATE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are maintaining a running summary of a tutoring conversation between a student and an AI assistant.

            You will receive:
            - The existing summary of the conversation so far (may be empty if this is the first turn)
            - The latest exchange (student question + assistant answer)

            Your job: produce an updated summary that captures the full conversation so far, including this new exchange.

            Rules:
            - Write in third person ("The student asked...", "The assistant explained...")
            - Be concise but complete — preserve key concepts, misconceptions, and breakthroughs
            - Do not lose important context from the old summary
            - Do not add information that wasn't in the conversation

            Respond with ONLY the updated summary string. No preamble, no formatting.""",
            ),
            (
                "human",
                """Existing Summary:
            {old_summary}

            Latest Exchange:
            Student: {question}
            Assistant: {answer}""",
            ),
        ]
    )

    chain = _UPDATE_SUMMARY_PROMPT | llm

    try:
        response = chain.invoke(
            {
                "old_summary": old_summary or "No prior conversation.",
                "question": question,
                "answer": answer,
            }
        )
        updated_summary = response.content.strip()
        print(f"[qa_chain] STM summary updated ({len(updated_summary)} chars).")
        return updated_summary

    except Exception as e:
        raise RuntimeError(
            f"STM summary update failed. Error type: {type(e).__name__}. Error: {e}"
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS — LTM
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS — LTM (corrected for langmem 0.0.30)
# ─────────────────────────────────────────────────────────────────────────────


def _ltm_path(student_id: str) -> Path:
    """Returns the JSON persistence path for a student's LTM store."""
    path = LTM_DIR / student_id
    path.mkdir(parents=True, exist_ok=True)
    return path / "memories.json"


def _load_ltm_store(student_id: str) -> InMemoryStore:
    """
    Loads a student's LTM memories from disk into a fresh InMemoryStore.

    InMemoryStore is the runtime store. JSON on disk is the persistence layer.
    On first call (no file yet) returns an empty store.

    Args:
        student_id: Unique student identifier.

    Returns:
        InMemoryStore pre-populated with the student's saved memories.

    Raises:
        RuntimeError: If the JSON file exists but cannot be parsed.

    Example:
        >>> store = _load_ltm_store("student_42")
        [qa_chain] LTM store loaded for student 'student_42' (3 memories).
    """
    store = InMemoryStore()
    json_path = _ltm_path(student_id)

    if not json_path.exists():
        print(
            f"[qa_chain] No LTM file found — starting fresh for student '{student_id}'."
        )
        return store

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            memories = json.load(
                f
            )  # list of {"namespace": [...], "key": str, "value": str}

        for mem in memories:
            store.put(
                namespace=tuple(mem["namespace"]),
                key=mem["key"],
                value={"content": mem["value"]},
            )
        print(
            f"[qa_chain] LTM store loaded for student '{student_id}' ({len(memories)} memories)."
        )
        return store

    except Exception as e:
        raise RuntimeError(
            f"Failed to load LTM store from {json_path}. Error: {e}"
        ) from e


def _save_ltm_store(student_id: str, store: InMemoryStore) -> None:
    """
    Serializes all memories from the InMemoryStore back to JSON on disk.

    Args:
        student_id: Unique student identifier.
        store:      The InMemoryStore instance after LTM update.

    Raises:
        OSError: If the file cannot be written.

    Example:
        >>> _save_ltm_store("student_42", store)
        [qa_chain] LTM store saved for student 'student_42' (4 memories).
    """
    namespace = ("ltm", student_id)
    items = store.search(namespace)  # returns list of Item objects

    memories = [
        {
            "namespace": list(item.namespace),
            "key": item.key,
            "value": item.value.get("content", str(item.value)),
        }
        for item in items
    ]

    json_path = _ltm_path(student_id)

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=4)
        print(
            f"[qa_chain] LTM store saved for student '{student_id}' ({len(memories)} memories)."
        )
    except OSError as e:
        raise OSError(f"Failed to save LTM store to {json_path}. OS error: {e}") from e


def _query_ltm(
    store: InMemoryStore,
    question: str,
    student_id: str,
) -> str:
    """
    Searches the InMemoryStore for memories relevant to the current question.

    Uses basic keyword search across stored memory content since InMemoryStore
    does not support vector similarity without an embedding index configured.

    Args:
        store:      Pre-loaded InMemoryStore for the student.
        question:   The student's current question.
        student_id: Used for namespace scoping.

    Returns:
        Numbered string of relevant memories, or "" if none found.

    Example:
        >>> ctx = _query_ltm(store, "What is Big-O notation?", "student_42")
        "1. Student previously confused O(n) with O(n²) when describing bubble sort."
    """
    namespace = ("ltm", student_id)

    try:
        items = store.search(namespace)

        if not items:
            print(f"[qa_chain] LTM: store is empty for student '{student_id}'.")
            return ""

        # Basic relevance filter — check if any question keywords appear in memory content
        question_words = set(question.lower().split())
        relevant = []

        for item in items:
            content = item.value.get("content", "")
            if any(word in content.lower() for word in question_words if len(word) > 3):
                relevant.append(content)

        if not relevant:
            print(f"[qa_chain] LTM: no relevant memories for this question.")
            return ""

        lines = [f"{i + 1}. {mem}" for i, mem in enumerate(relevant)]
        print(f"[qa_chain] LTM: {len(relevant)} relevant memory/memories retrieved.")
        return "\n".join(lines)

    except Exception as e:
        print(f"[qa_chain] LTM query failed (non-fatal). Error: {e}")
        return ""


def _update_ltm(
    store: InMemoryStore,
    llm: BaseChatModel,
    question: str,
    answer: str,
    student_id: str,
) -> dict:
    """
    Passes the current exchange to create_memory_manager for extraction,
    then writes any extracted memories into the InMemoryStore and persists to disk.

    Args:
        store:      Pre-loaded InMemoryStore for the student.
        llm:        Language model used by langmem for memory extraction.
        question:   The student's question for this turn.
        answer:     The assistant's response for this turn.
        student_id: Used for namespace scoping.

    Returns:
        Empty dict — langmem 0.0.30 does not expose usage_metadata.

    Example:
        >>> usage = _update_ltm(store, llm, "Why is quicksort slow?", "...", "student_42")
        [qa_chain] LTM: 1 new memory/memories extracted and saved.
    """
    namespace = ("ltm", student_id)

    try:
        manager = create_memory_manager(
            llm,
            instructions="""You are tracking long-term facts about a STUDENT for a personalised tutoring system.

                Extract and store ONLY information about the student themselves:
                - Misconceptions or incorrect beliefs the student has demonstrated 
                - Learning preferences the student has expressed (e.g. dislike of analogies)
                - Persistent confusion patterns across the conversation

                DO NOT store:
                - Factual content from the assistant's explanations
                - Definitions, descriptions, or summaries of topics
                - Any information that is about the subject matter rather than the student
                - Inferences or assumptions about the student not directly evidenced in their message

                If the exchange contains nothing genuinely about the student's learning state, extract nothing.""",
        )

        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        results = manager.invoke({"messages": conversation})

        print(f"[qa_chain] LTM raw manager output: {results}")  # ← add this temporarily

        if not results:
            print(f"[qa_chain] LTM: no memories extracted from this exchange.")
            return {}

        import uuid

        new_count = 0

        for item in results:
            # ExtractedMemory(id=..., content=Memory(content='...'))
            raw = item.content if hasattr(item, "content") else item
            content = raw.content if hasattr(raw, "content") else str(raw)

            if content and content.strip():
                store.put(
                    namespace=namespace,
                    key=item.id if hasattr(item, "id") else str(uuid.uuid4()),
                    value={"content": content},
                )
                new_count += 1

        if new_count > 0:
            _save_ltm_store(student_id, store)
            print(
                f"[qa_chain] LTM: {new_count} new memory/memories extracted and saved."
            )
        else:
            print(f"[qa_chain] LTM: manager ran but extracted nothing worth storing.")

        return {}

    except Exception as e:
        print(f"[qa_chain] LTM update failed (non-fatal). Error: {e}")
        raise
        # return {}


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS — question reformulation + QA LLM call
# ─────────────────────────────────────────────────────────────────────────────


def _reformulate_question(
    llm: BaseChatModel,
    question: str,
    stm_summary: str,
) -> Tuple[str, dict]:
    """
    Reformulates the student's question into a standalone, context-aware string
    using the STM summary as conversation context.

    If stm_summary is empty the question is returned unchanged without an LLM call.

    Args:
        llm:         Language model instance.
        question:    The student's raw question.
        stm_summary: Compressed conversation context from STM.

    Returns:
        Tuple of (reformulated_question_string, usage_metadata_dict).

    Raises:
        RateLimitExhaustedError: If rate limit retries are exhausted.
        RuntimeError:            If reformulation fails after all retries.

    Example:
        >>> q, usage = _reformulate_question(llm, "What about its complexity?", "We discussed merge sort.")
        >>> q
        "What is the time complexity of merge sort?"
    """
    if not stm_summary:
        return question, {}

    reformulation_template = ChatPromptTemplate.from_template(REFORMULATION_PROMPT)
    chain = reformulation_template | llm
    rpm_attempt = 0

    while rpm_attempt <= _MAX_RATE_LIMIT_RETRIES:
        try:
            response = chain.invoke({"question": question, "history": stm_summary})
            usage_metadata = response.usage_metadata or {}
            reformulated = response.content.strip().strip('"')

            if not reformulated:
                print(
                    "[qa_chain] Reformulation returned empty string — using original question."
                )
                return question, {
                    k: v for k, v in usage_metadata.items() if isinstance(v, int)
                }

            print(f"[qa_chain] Question reformulated.")
            return reformulated, {
                k: v for k, v in usage_metadata.items() if isinstance(v, int)
            }

        except Exception as e:
            kind = _classify_rate_error(e)

            if kind == "rpd":
                raise RateLimitExhaustedError(
                    kind="rpd",
                    message=f"Daily quota exhausted during question reformulation. Error: {e}",
                ) from e

            if kind == "rpm":
                if rpm_attempt >= _MAX_RATE_LIMIT_RETRIES:
                    raise RateLimitExhaustedError(
                        kind="rpm",
                        message=f"RPM retries exhausted during reformulation. Error: {e}",
                    ) from e
                backoff = _RPM_BACKOFF_SECONDS[
                    min(rpm_attempt, len(_RPM_BACKOFF_SECONDS) - 1)
                ]
                print(
                    f"[qa_chain] RPM hit during reformulation. Backing off {backoff}s."
                )
                time.sleep(backoff)
                rpm_attempt += 1
                continue

            # Non-rate-limit error — fall back to original question
            print(
                f"[qa_chain] Reformulation failed (non-fatal): {e}. Using original question."
            )
            return question, {}

    return question, {}


def _call_qa_llm(
    llm: BaseChatModel,
    question: str,
    stm_summary: str,
    ltm_context: str,
    chunks: List[Document],
    learning_pace: str,
    current_topic: str,
) -> Tuple[dict, dict]:
    """
    Calls the main QA LLM with the full enriched prompt context.

    Injects STM summary and LTM context as labelled sections above the source
    material block. Both sections are omitted from the prompt if empty.

    Args:
        llm:           Language model instance.
        question:      Reformulated (or original) question string.
        stm_summary:   Compressed conversation context. Empty string → omitted.
        ltm_context:   Relevant past memories string. Empty string → omitted.
        chunks:        Retrieved semantic chunks.
        learning_pace: "slow" | "average" | "fast" — controls explanation depth.
        current_topic: Topic label for the QA prompt.

    Returns:
        Tuple of (result_dict, usage_metadata_dict).
        result_dict keys: "answer" (str), "grounded" (bool).

    Raises:
        RateLimitExhaustedError: If rate limit retries are exhausted.
        RuntimeError:            If parsing fails after all retries.

    Example:
        >>> result, usage = _call_qa_llm(llm, "What is merge sort?", "", "", chunks, "average", "Sorting")
        >>> result["grounded"]
        True
    """
    parser = JsonOutputParser()
    prompt_string = QA_PROMPT_MAP.get(learning_pace, QA_PROMPT_MAP["average"])
    # The below turns the prompt into a Runnable
    prompt_template = ChatPromptTemplate.from_template(prompt_string)
    chain = prompt_template | llm

    # Build optional memory context block to prepend to the prompt
    memory_context_parts = []
    if stm_summary:
        memory_context_parts.append(
            f"[Conversation Summary — what has been discussed so far]\n{stm_summary}"
        )
    if ltm_context:
        memory_context_parts.append(
            f"[Prior Knowledge — relevant concepts this student has encountered before]\n{ltm_context}"
        )
    memory_context_block = "\n\n".join(memory_context_parts)

    grounding_instruction = (
        GROUNDING_INSTRUCTION_CONTEXT if chunks else GROUNDING_INSTRUCTION_GENERAL
    )
    formatted_chunks = _format_chunks(chunks)
    print(formatted_chunks)

    rpm_attempt = 0

    while rpm_attempt <= _MAX_RATE_LIMIT_RETRIES:
        parse_attempt = 0
        raw_response = None

        while parse_attempt < _MAX_QA_RETRIES:
            try:
                raw_response = chain.invoke(
                    {
                        "question": question,
                        "context": formatted_chunks,
                        "topic": current_topic,
                        "grounding_instruction": grounding_instruction,
                        "memory_context": memory_context_block,
                    }
                )
                usage_metadata = {
                    k: v
                    for k, v in (raw_response.usage_metadata or {}).items()
                    if isinstance(v, int)
                }

                # Primary parse
                try:
                    result = parser.parse(raw_response.content)
                    print(
                        f"[qa_chain] QA LLM call succeeded. Grounded: {result.get('grounded')}"
                    )
                    return result, usage_metadata

                except Exception as parse_err:
                    print(
                        f"[qa_chain] QA parse failed (attempt {parse_attempt + 1}/{_MAX_QA_RETRIES}). "
                        f"Trying json_repair. Error: {parse_err}"
                    )
                    try:
                        repaired = repair_json(raw_response.content)
                        result = json.loads(repaired)
                        print(f"[qa_chain] json_repair succeeded.")
                        return result, usage_metadata
                    except Exception as repair_err:
                        print(
                            f"[qa_chain] json_repair failed (attempt {parse_attempt + 1}/{_MAX_QA_RETRIES}). "
                            f"Error: {repair_err}"
                        )
                        parse_attempt += 1

            except Exception as e:
                kind = _classify_rate_error(e)

                if kind == "rpd":
                    raise RateLimitExhaustedError(
                        kind="rpd",
                        message=f"Daily quota exhausted during QA LLM call. Error: {e}",
                    ) from e

                if kind == "rpm":
                    if rpm_attempt >= _MAX_RATE_LIMIT_RETRIES:
                        raise RateLimitExhaustedError(
                            kind="rpm",
                            message=f"RPM retries exhausted during QA LLM call. Error: {e}",
                        ) from e
                    backoff = _RPM_BACKOFF_SECONDS[
                        min(rpm_attempt, len(_RPM_BACKOFF_SECONDS) - 1)
                    ]
                    print(
                        f"[qa_chain] RPM hit. Backing off {backoff}s (attempt {rpm_attempt + 1})."
                    )
                    time.sleep(backoff)
                    rpm_attempt += 1
                    break  # restart outer loop

                raise RuntimeError(
                    f"Unexpected error during QA LLM call. "
                    f"Error type: {type(e).__name__}. Error: {e}"
                ) from e

        else:
            raise RuntimeError(
                f"Failed to parse QA response after {_MAX_QA_RETRIES} parse attempts. "
                f"Raw response: {raw_response.content if raw_response else 'None'}"
            )

    raise RateLimitExhaustedError(
        kind="rpm", message="RPM retries exhausted during QA LLM call."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS — history I/O
# ─────────────────────────────────────────────────────────────────────────────


def _load_stm(student_id: str, convo_name: str) -> dict:
    """
    Loads the running STM summary for a conversation from disk.

    Returns a dict with keys "summary" (str) and "turn_count" (int).
    Returns a blank state if the file does not exist (new conversation).

    Args:
        student_id: Unique student identifier.
        convo_name: Sanitised conversation name.

    Returns:
        Dict: {"summary": str, "turn_count": int}

    Raises:
        ValueError: If the file exists but contains invalid JSON.

    Example:
        >>> state = _load_stm("student_42", "sorting_session_1")
        [qa_chain] STM loaded for 'sorting_session_1' (3 turns summarised).
    """
    path = CHAT_HISTORY_DIR / student_id / f"{convo_name}.json"

    if not path.exists():
        print(f"[qa_chain] New conversation '{convo_name}' — STM initialised empty.")
        return {"summary": "", "turn_count": 0}

    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        print(
            f"[qa_chain] STM loaded for '{convo_name}' "
            f"({state.get('turn_count', 0)} turns summarised)."
        )
        return state
    except json.JSONDecodeError as e:
        raise ValueError(
            f"STM file for '{convo_name}' contains invalid JSON. "
            f"Path: {path}. Error: {e}"
        ) from e


def _save_stm(student_id: str, convo_name: str, state: dict) -> None:
    """
    Writes the updated STM summary state back to disk.

    Args:
        student_id: Unique student identifier.
        convo_name: Sanitised conversation name.
        state:      Dict with "summary" and "turn_count" keys.

    Raises:
        OSError: If the file cannot be written.

    Example:
        >>> _save_stm("student_42", "sorting_session_1", {"summary": "...", "turn_count": 4})
        [qa_chain] STM saved for 'sorting_session_1' (4 turns summarised).
    """
    path = CHAT_HISTORY_DIR / student_id / f"{convo_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        print(
            f"[qa_chain] STM saved for '{convo_name}' "
            f"({state.get('turn_count', 0)} turns summarised)."
        )
    except OSError as e:
        raise OSError(
            f"Failed to write STM for '{convo_name}'. Path: {path}. OS error: {e}"
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT 1 — list_conversations
# ─────────────────────────────────────────────────────────────────────────────


def list_conversations(student_id: str) -> List[str]:
    """
    Returns the list of saved conversation names for a student.

    No LLM call. No @token_guard.

    Args:
        student_id: Unique student identifier.

    Returns:
        Sorted list of conversation name strings (filenames without .json extension).
        Empty list if no conversations exist yet.

    Raises:
        ValueError: If student_id is not a non-empty string.

    Example:
        >>> list_conversations("student_42")
        ["algorithms_week3", "sorting_deep_dive", "trees_intro"]
    """
    if not student_id or not isinstance(student_id, str):
        raise ValueError(
            f"'student_id' must be a non-empty string. Got: {repr(student_id)}"
        )

    convo_dir = CHAT_HISTORY_DIR / student_id

    if not convo_dir.exists():
        print(f"[qa_chain] No conversations found for student '{student_id}'.")
        return []

    names = sorted(p.stem for p in convo_dir.iterdir() if p.suffix == ".json")
    print(f"[qa_chain] Found {len(names)} conversation(s) for student '{student_id}'.")
    return names


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT 2 — run_qa_chain  (@token_guard)
# ─────────────────────────────────────────────────────────────────────────────


@token_guard
def run_qa_chain(
    student_id: str,
    question: str,
    convo_name: str,
    from_condensed_notes: bool,
    store: Chroma,
    llm: BaseChatModel,
    learning_pace: str,
    current_topic: str,
    course: str = None,
) -> Any:
    """
    Main QA entry point. Handles the full STM + LTM + retrieval + generation pipeline.

    STM provides within-session context compression via ConversationSummaryBufferMemory.
    LTM provides cross-session semantic memory via LangMem (per-response update,
    LangMem decides relevance internally).

    Args:
        student_id:           Unique student identifier.
        question:             The student's raw question string.
        convo_name:           Name for the conversation thread. Used as the filename.
                              Unsafe characters are sanitised automatically.
        from_condensed_notes: If True, retrieves from the condensed notes collection.
                              If False, retrieves from the original ingested PDF collection.
        store:                Chroma vector store instance.
        llm:                  Language model instance.
        learning_pace:        "slow" | "average" | "fast" — controls explanation depth.
        current_topic:        Topic label injected into the QA prompt.
        course:               Title of the course in the vector store.

    Returns:
        _SyntheticResponse:
            content = {
                "answer":       str,    # LLM response to the student
                "source_pages": list,   # page numbers from retrieved chunks
                "convo_name":   str,    # sanitised conversation name
                "grounded":     bool,   # True if answer sourced from chunks
                "ltm_usage":    dict,   # LangMem's token usage (router deducts this)
            }
            usage_metadata = accumulated token counts (QA + STM + reformulation)

    Raises:
        ValueError:              On invalid inputs.
        RateLimitExhaustedError: If rate limit cannot be recovered from.
        RuntimeError:            On unexpected LLM or parsing failures.

    Example:
        >>> response = run_qa_chain(
        ...     student_id="student_42",
        ...     question="Why is merge sort preferred over bubble sort?",
        ...     convo_name="sorting_deep_dive",
        ...     from_condensed_notes=False,
        ...     current_topic="Sorting Algorithms",
        ...     store=chroma_store,
        ...     llm=llm,
        ...     learning_pace="average",
        ... )
        >>> response.content["answer"]
        "Merge sort runs in O(n log n) in all cases, while bubble sort degrades to O(n²)..."
        >>> response.content["grounded"]
        True
    """

    # ── Input validation ──────────────────────────────────────────────────────

    if not student_id or not isinstance(student_id, str):
        raise ValueError(
            f"'student_id' must be a non-empty string. Got: {repr(student_id)}"
        )

    if not question or not isinstance(question, str):
        raise ValueError(
            f"'question' must be a non-empty string. Got: {repr(question)}"
        )

    if not convo_name or not isinstance(convo_name, str):
        raise ValueError(
            f"'convo_name' must be a non-empty string. Got: {repr(convo_name)}"
        )

    valid_paces = {"slow", "average", "fast"}
    if learning_pace not in valid_paces:
        raise ValueError(
            f"'learning_pace' must be one of {valid_paces}. Got: {repr(learning_pace)}"
        )

    convo_name = _sanitise_convo_name(convo_name)
    print(
        f"[qa_chain] Starting QA — student: '{student_id}' | "
        f"topic: '{current_topic}' | convo: '{convo_name}' | pace: '{learning_pace}'"
    )

    accumulated_usage: dict = {}

    # ── Step 1 — Load conversation history ────────────────────────────────────

    history = _load_stm(student_id, convo_name)

    # ── Step 2 — Load STM summary from disk

    stm_state = _load_stm(student_id, convo_name)
    stm_summary = stm_state.get("summary", "")

    if stm_summary:
        print(f"[qa_chain] STM context loaded ({len(stm_summary)} chars).")
    else:
        print(f"[qa_chain] No prior summary — first turn in this conversation.")

    # ── Step 3 — Query LTM for relevant memories ──────────────────────────────

    ltm_store = _load_ltm_store(student_id)
    ltm_context = _query_ltm(ltm_store, question, student_id)

    # ── Step 4 — Reformulate question ─────────────────────────────────────────

    if stm_summary:
        if USE_GEMINI:
            time.sleep(_REQUEST_DELAY_SECONDS)
        reformulated_question, reform_usage = _reformulate_question(
            llm, question, stm_summary
        )
        accumulated_usage = _accumulate_tokens(accumulated_usage, reform_usage)
    else:
        reformulated_question = question
        print("[qa_chain] No STM context — reformulation skipped.")

    # ── Step 5 — Retrieve semantic chunks ─────────────────────────────────────

    try:
        chunks = get_semantic_chunks(
            store=store,
            query=reformulated_question,
            course=course,
            score_threshold=0.4,
            topic=current_topic,
            condensed=from_condensed_notes,
        )
        print(f"[qa_chain] Retrieved {len(chunks)} chunk(s).")
    except Exception as e:
        print(
            f"[qa_chain] Retrieval failed (non-fatal, proceeding with empty context). Error: {e}"
        )
        chunks = []

    source_pages = _extract_source_pages(chunks)

    # ── Step 6 — Call QA LLM ──────────────────────────────────────────────────

    if USE_GEMINI:
        time.sleep(_REQUEST_DELAY_SECONDS)

    result, qa_usage = _call_qa_llm(
        llm=llm,
        question=reformulated_question,
        stm_summary=stm_summary,
        ltm_context=ltm_context,
        chunks=chunks,
        learning_pace=learning_pace,
        current_topic=current_topic,
    )
    accumulated_usage = _accumulate_tokens(accumulated_usage, qa_usage)

    answer = result.get("answer", "").strip()
    answer = textwrap.fill(answer, width=80)
    grounded = bool(result.get("grounded", False))

    # ── Step 7 — Update summary and save ────────────────────────────────────

    updated_summary = _update_stm_summary(stm_summary, question, answer, llm)
    _save_stm(
        student_id,
        convo_name,
        {
            "summary": updated_summary,
            "turn_count": stm_state.get("turn_count", 0) + 1,
        },
    )

    # ── Step 8 — Update LTM (per-response, LangMem decides relevance) ─────────

    ltm_usage = _update_ltm(ltm_store, llm, question, answer, student_id)

    # ── Step 9 — Return ───────────────────────────────────────────────────────

    print(f"[qa_chain] QA chain complete for student '{student_id}'.")

    return _SyntheticResponse(
        content={
            "question": question,
            "answer": answer,
            "source_pages": source_pages,
            "convo_name": convo_name,
            "grounded": grounded,
            "ltm_usage": ltm_usage,  # router deducts this separately from token budget
        },
        usage_metadata=accumulated_usage,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT 3 — view_ltm
# ─────────────────────────────────────────────────────────────────────────────


def view_ltm(student_id: str) -> str:
    """
    Returns all stored long-term memories for a student as a human-readable string.

    No LLM call. No @token_guard.

    Args:
        student_id: Unique student identifier.

    Returns:
        Indexed string of all stored memories, one per line.
        Returns a 'no memories' message if the store is empty or does not exist.

    Raises:
        ValueError:  If student_id is not a non-empty string.
        RuntimeError: If the LTM store cannot be loaded.

    Example:
        >>> print(view_ltm("student_42"))
        Long-Term Memories for student_42:
        ──────────────────────────────────
        1. Student confused O(n) with O(n²) when describing bubble sort (resolved in session 3).
        2. Student understands merge sort's divide-and-conquer structure well.
        3. Student asked about quicksort pivot selection — showed curiosity about optimisation.
    """
    if not student_id or not isinstance(student_id, str):
        raise ValueError(
            f"'student_id' must be a non-empty string. Got: {repr(student_id)}"
        )

    json_path = _ltm_path(student_id)

    if not json_path.exists():
        return "No long-term memories stored yet for student."

    try:
        store = _load_ltm_store(student_id)
        items = store.search(("ltm", student_id))

        if not items:
            return "No long-term memories stored yet for student."

        lines = ["Long-Term Memories:", "─" * 40]
        for i, item in enumerate(items):
            content = item.value.get("content", str(item.value))
            lines.append(f"{i + 1}. {content}")

        return "\n".join(lines)

    except Exception as e:
        raise RuntimeError(
            f"Failed to load LTM for student '{student_id}'. Error: {e}"
        ) from e
