"""
quiz_chain.py
─────────────────────────────────────────────────────────────────────────────
Generates a 10-question, three-level assessment quiz grounded in retrieved
PDF chunks for a given course and topic.

Assessment structure (fixed, pace-independent):
    Surface-Level    3 questions  — recognition and recall
    Conceptual       4 questions  — relationships and processes
    Deep-Level       3 questions  — application and inference

Each question is returned with a rubric-style model_answer string grounded
in the retrieved material. student_answer, score, and explanation are
scaffolded as null — they are filled by the router (student submission) and
eval_chain (grading) respectively.

After generation, the quiz is saved to:
    data/quizzes/{student_id}_{topic_slug}_{timestamp}.json

This mirrors the notes_chain pattern where notes are saved to data/notes/.
For production, disk persistence is swapped for Supabase — chain unchanged.

Rate-limit architecture mirrors notes_chain.py and qa_chain.py:
    USE_GEMINI = False  →  local Ollama, no delays, outer retry loop inactive
    USE_GEMINI = True   →  Gemini API, inter-call delays + RPM backoff active
"""

import json
import time
import logging
import re

from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict
from app.agent.rag.prompts import QUIZ_PROMPT

from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import Chroma
from json_repair import repair_json

from app.services.token_service import token_guard
from app.agent.rag.retrieval.retriever import get_topic_chunks, get_semantic_chunks

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# RATE-LIMIT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

USE_GEMINI: bool = False

_REQUEST_DELAY_SECONDS  = 5
_RPM_BACKOFF_SECONDS    = [15, 30, 60]
_MAX_RATE_LIMIT_RETRIES = 3
_MAX_QUIZ_RETRIES       = 3

# ─────────────────────────────────────────────────────────────────────────────
# ASSESSMENT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_TOTAL_QUESTIONS  = 10
_SURFACE_COUNT    = 3
_CONCEPTUAL_COUNT = 4
_DEEP_COUNT       = 3

# ─────────────────────────────────────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────────────────────────────────────

_QUIZZES_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "quizzes"


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class RateLimitExhaustedError(RuntimeError):
    """
    Raised when all rate-limit retry attempts are exhausted.

    Attributes:
        kind (str): "rpm" — per-minute limit hit repeatedly.
                    "rpd" — daily quota exhausted, unrecoverable.

    Example:
        raise RateLimitExhaustedError("rpd", "Daily Gemini quota exhausted.")
    """

    def __init__(self, kind: str, message: str):
        if kind not in ("rpm", "rpd"):
            raise ValueError(
                f"Invalid kind '{kind}'. Must be 'rpm' or 'rpd'."
            )
        self.kind = kind
        super().__init__(message)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _classify_rate_error(e: Exception) -> str:
    """
    Classifies an exception as a rate-limit type or unrelated error.

    Inspects the string representation of the exception for known Gemini
    rate-limit signals. Returns one of three categories:
        "rpm"   — per-minute quota hit, recoverable with backoff
        "rpd"   — daily quota exhausted, unrecoverable
        "other" — unrelated error, should not trigger rate-limit handling

    Args:
        e (Exception): Any exception caught during an LLM call.

    Returns:
        str: "rpm", "rpd", or "other".

    Example:
        >>> _classify_rate_error(Exception("429 RESOURCE_EXHAUSTED daily"))
        "rpd"
        >>> _classify_rate_error(Exception("429 Too Many Requests"))
        "rpm"
        >>> _classify_rate_error(ValueError("Invalid JSON"))
        "other"
    """
    msg = str(e).lower()
    if "daily" in msg or "quota" in msg:
        return "rpd"
    if "429" in msg or "rate" in msg or "resource_exhausted" in msg:
        return "rpm"
    return "other"


def _slugify(text: str) -> str:
    """
    Converts a topic string into a filesystem-safe slug.

    Lowercases the string, replaces spaces and non-alphanumeric characters
    with underscores, and collapses consecutive underscores.

    Args:
        text (str): Raw topic string, e.g. "Sorting Algorithms".

    Returns:
        str: Slug string, e.g. "sorting_algorithms".

    Example:
        >>> _slugify("Newton's Laws of Motion!")
        "newtons_laws_of_motion"
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", "_", text)      # spaces → underscores
    text = re.sub(r"_+", "_", text)       # collapse consecutive underscores
    return text


def _format_chunks(chunks: List[Document]) -> str:
    """
    Formats retrieved chunks into a numbered context block for prompt injection.

    Handles both broad retrieval (documents grouped by section in a dictionary) 
    and adaptive retrieval (a flat list of targeted documents). Each chunk is 
    prefixed with a continuous, sequential index regardless of the input structure. 
    Page metadata is not injected here — source_pages are extracted separately 
    from metadata during the post-processing step.

    Args:
        chunks (List[Document] | Dict[str, List[Document]]): 
            The retrieved text chunks. 
            - Dict: Broad retrieval mapped as {section_title: [Documents]}.
            - List: Adaptive retrieval containing a flat sequence of Documents.

    Returns:
        str: Formatted string block combining all chunk content, ready for prompt injection.

    Example:
        Input (List): 
            [Document(page_content="ATP is...", metadata={"page": 4})]
        Input (Dict): 
            {"Cell Energy": [Document(page_content="ATP is...", metadata={"page": 4})]}
            
        Output (for both): 
            "[1] ATP is..."
    """
    if not chunks:
        return ""

    lines = []
    i = 1
    
    # .items() lets you access both the section title and the list of documents
    # Branch 1: Broad Retrieval (Dictionary)
    if isinstance(chunks, dict):
        for section_title, doc_list in chunks.items():
            for doc in doc_list:
                lines.append(f"[{i}] {doc.page_content.strip()}")
                i += 1
                
    # Branch 2: Adaptive Retrieval (List)
    elif isinstance(chunks, list):
        for doc in chunks:
            lines.append(f"[{i}] {doc.page_content.strip()}")
            i += 1

    return "\n\n".join(lines)


def _get_weak_area_chunks(
    store:      Chroma,
    course:     str,
    topic:      str,
    weak_areas: list[str],
) -> list:
    """
    Fetches chunks from Chroma filtered by specific weak sections.

    Used when the student has prior quiz history and the adaptive system
    targets only their weak areas rather than the full topic. Each section
    in weak_areas is queried independently and results are combined.

    Args:
        store      (Chroma):     Initialised Chroma vector store.
        course     (str):        Course name — used as metadata filter.
        topic      (str):        Topic name — used as metadata filter.
        weak_areas (list[str]):  Section names from the learner profile
                                 e.g. ["Algorithm Analysis", "Sorting Methods"]

    Returns:
        list: Combined list of LangChain Document objects across all sections.
              Empty list if all section queries fail.

    Example:
        chunks = _get_weak_area_chunks(
            store, "Computer Science", "Sorting Algorithms",
            ["Algorithm Analysis", "Complexity Theory"]
        )
        # Returns chunks filtered to only those two sections.
    """
    all_chunks = []

    for section in weak_areas:
        try:
            results = store.similarity_search(
                query  = topic,   # broad signal — metadata filter does the work
                k      = 20,
                filter = {
                    "$and": [
                        {"course":  {"$eq": course}},
                        {"topic":   {"$eq": topic}},
                        {"section": {"$eq": section}},
                    ]
                },
            )
            print(
                f"[quiz_chain] Section '{section}': {len(results)} chunk(s) fetched."
            )
            all_chunks.extend(results)

        except Exception as e:
            print(
                f"[quiz_chain] WARNING: Failed to fetch chunks for section "
                f"'{section}': {e}. Skipping."
            )

    return all_chunks

def _extract_source_pages(chunks: List[Document] | Dict[str, List[Document]]) -> list[int]:
    """
    Extracts unique page numbers from chunk metadata.

    LLM-generated source_pages are not trusted. This function overrides 
    them with ground-truth metadata values. Handles both Dictionary 
    (broad retrieval) and List (adaptive/semantic retrieval) structures.

    Args:
        chunks (List[Document] | Dict[str, List[Document]]): Retrieved chunks.

    Returns:
        list[int]: Deduplicated and sorted page numbers.
    """
    seen  = set()
    pages = []
    
    # Branch 1: Dictionary structure (Broad Retrieval)
    if isinstance(chunks, dict):
        for doc_list in chunks.values():
            for doc in doc_list:
                page = doc.metadata.get("page_number")
                if page is not None and page not in seen:
                    seen.add(page)
                    pages.append(int(page))
                    
    # Branch 2: Flat list structure (Adaptive / Semantic Search)
    elif isinstance(chunks, list):
        for doc in chunks:
            page = doc.metadata.get("page_number")
            if page is not None and page not in seen:
                seen.add(page)
                pages.append(int(page))
                
    return sorted(pages)

def _validate_and_repair_quiz(raw: Any) -> dict:
    """
    Validates quiz output structure and warns on count or distribution mismatch.

    Does not scaffold null fields or map sources — those happen in
    _map_sources_to_questions after per-question semantic search.

    Args:
        raw (Any): Parsed JSON dict from LLM output.

    Returns:
        dict: Quiz dict with questions list intact.

    Raises:
        ValueError: If the top-level 'questions' key is missing entirely.

    Example:
        Input:  {"questions": [{"level": "surface", "question": "...", ...}]}
        Output: Same dict, unchanged — warnings printed if counts are off.
    """
    if "questions" not in raw:
        raise ValueError(
            "LLM output missing top-level 'questions' key. "
            f"Got keys: {list(raw.keys())}"
        )

    questions = raw["questions"]
    total     = len(questions)

    if total != _TOTAL_QUESTIONS:
        print(
            f"[quiz_chain] WARNING: Expected {_TOTAL_QUESTIONS} questions, "
            f"got {total}. Proceeding with {total}."
        )

    counts   = {"surface": 0, "conceptual": 0, "deep": 0}
    expected = {
        "surface":    _SURFACE_COUNT,
        "conceptual": _CONCEPTUAL_COUNT,
        "deep":       _DEEP_COUNT,
    }

    for q in questions:
        level = q.get("level", "").lower()
        if level in counts:
            counts[level] += 1

    for level, exp_count in expected.items():
        if counts[level] != exp_count:
            print(
                f"[quiz_chain] WARNING: Expected {exp_count} '{level}' questions, "
                f"got {counts[level]}."
            )

    return {"questions": questions}


def _map_sources_to_questions(
    questions: list[dict],
    store:     Chroma,
    course:    str,
    topic:     str,
) -> list[dict]:
    """
    Maps precise source attribution to each question via per-question semantic search.

    For each question, concatenates the question text and model_answer into a
    search query and calls get_semantic_chunks() with k=3 to find the exact
    chunks the LLM drew from. Page numbers and section titles are extracted
    from chunk metadata and injected into the question dict.

    Also scaffolds student_answer, score, and explanation as null on every
    question. This is the single place where all question fields are finalised.

    Why question + model_answer as the query:
        The model_answer is grounded in the source vocabulary. Using it
        alongside the question produces a richer, more targeted retrieval
        signal than the question alone — especially for surface-level questions
        where the answer is a direct restatement of a chunk.

    source_pages: sorted, deduplicated list of page numbers across all k chunks.
    section:      title from the metadata of the first (most relevant) returned
                  chunk. Injected only if the field exists — omitted silently
                  if not present rather than injecting null.

    Args:
        questions (list[dict]): Questions list from _validate_and_repair_quiz.
        store     (Chroma):     Initialised Chroma vector store.
        course    (str):        Course name — passed to retriever for filtering.
        topic     (str):        Topic name — passed to retriever for filtering.

    Returns:
        list[dict]: Same questions list with source_pages, section (where
                    available), student_answer, score, explanation injected.

    Example:
        Input question:
            {
                "level":        "surface",
                "question":     "What is the time complexity of Quick Sort?",
                "model_answer": "O(n log n) on average.",
                "source_pages": [],
            }
        Output question:
            {
                "level":          "surface",
                "question":       "What is the time complexity of Quick Sort?",
                "model_answer":   "O(n log n) on average.",
                "source_pages":   [4],
                "section":        "Algorithm Analysis",
                "student_answer": null,
                "score":          null,
                "explanation":    null,
            }
    """
    for i, q in enumerate(questions):
        query = f"{q.get('question', '')} {q.get('model_answer', '')}".strip()

        try:
            chunks = get_semantic_chunks(query=query, store=store, top_k=3, score_threshold=0.4, course=course, topic=topic)
        except Exception as e:
            print(
                f"[quiz_chain] WARNING: Source mapping failed for question {i + 1}: {e}. "
                "Falling back to empty source_pages."
            )
            chunks = []

        # Extract and sort page numbers across all returned chunks
        seen_pages = set()
        pages      = []
        for chunk in chunks:
            page = chunk.metadata.get("page_number")
            if page is not None and page not in seen_pages:
                seen_pages.add(page)
                pages.append(int(page))

        q["source_pages"] = sorted(pages)

        # Inject section from the first (most relevant) chunk only if present
        if chunks:
            section = chunks[0].metadata.get("section")
            if section:
                q["section"] = section
            else:
                q["section"] = ""

        # Scaffold submission and grading fields
        q["student_answer"] = None
        q["score"]          = None
        q["explanation"]    = None

        print(
            f"[quiz_chain] Q{i + 1} mapped — "
            f"pages: {q['source_pages']}, "
            f"section: {q.get('section', 'not found')}"
        )

    return questions


def _save_quiz(
    quiz:       dict,
    student_id: str,
    topic:      str,
) -> Path:
    """
    Saves the generated quiz to disk under data/quizzes/.

    File naming convention:
        {student_id}_{topic_slug}_{YYYYMMDD_HHMMSS}.json

    Creates the quizzes directory if it does not exist. Mirrors the pattern
    used by notes_chain for saving to data/notes/.

    Args:
        quiz       (dict): Processed quiz dict with all questions.
        student_id (str):  Student identifier for namespacing.
        topic      (str):  Topic string — slugified for the filename.

    Returns:
        Path: Absolute path of the saved file.

    Raises:
        IOError: If the file cannot be written.

    Example:
        path = _save_quiz(quiz_dict, "student_42", "Sorting Algorithms")
        # path → data/quizzes/student_42_sorting_algorithms_20250503_142301.json
    """
    try:
        _QUIZZES_DIR.mkdir(parents=True, exist_ok=True)

        topic_slug = _slugify(topic)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = f"{student_id}_{topic_slug}_{timestamp}.json"
        filepath   = _QUIZZES_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(quiz, f, indent=2, ensure_ascii=False)

        print(f"[quiz_chain] Quiz saved to: {filepath}")
        return filepath

    except Exception as e:
        raise IOError(
            f"Failed to save quiz for student '{student_id}', "
            f"topic '{topic}': {e}"
        ) from e


class _SyntheticResponse:
    """
    Wraps the quiz result and usage metadata into a token_guard-compatible object.

    Mirrors the structure used in notes_chain and qa_chain so token_guard can
    extract usage_metadata consistently across all chains.

    Attributes:
        content        (dict): Processed quiz dict with all questions.
        usage_metadata (dict): Token usage from the LLM call.
        saved_path     (Path): Path where the quiz JSON was saved to disk.

    Example:
        response = _SyntheticResponse(
            content        = {"questions": [...]},
            usage_metadata = {"input_tokens": 800, "output_tokens": 400},
            saved_path     = Path("data/quizzes/student_42_sorting_20250503.json"),
        )
    """

    def __init__(self, content: dict, usage_metadata: dict, saved_path: Path):
        self.content        = content
        self.usage_metadata = usage_metadata
        self.saved_path     = saved_path


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL LLM CALL
# ─────────────────────────────────────────────────────────────────────────────

def _call_quiz_llm(
    llm:    BaseChatModel,
    prompt: str,
) -> tuple[dict, dict]:
    """
    Invokes the LLM with nested retry logic for parse failures and rate limits.

    Retry architecture (two nested loops):
        Outer loop — rate-limit retries (runs exactly once when USE_GEMINI=False)
        Inner loop — parse-failure retries (JsonOutputParser → json_repair)

    On RPD error: raises RateLimitExhaustedError immediately — unrecoverable.
    On RPM error: breaks inner loop, outer loop applies exponential backoff.
    On parse failure: json_repair attempted as fallback before inner retry.

    Args:
        llm    (BaseChatModel): Initialised LangChain chat model.
        prompt (str):           Fully rendered prompt string.

    Returns:
        tuple[dict, dict]: (parsed_json_dict, usage_metadata_dict)

    Raises:
        RateLimitExhaustedError: Daily quota exhausted (kind="rpd") or all
                                  RPM retry attempts exhausted (kind="rpm").
        RuntimeError:            All parse retries failed with non-rate errors.

    Example:
        parsed, usage = _call_quiz_llm(llm, rendered_prompt)
        # parsed → {"questions": [{...}, ...]}
        # usage  → {"input_tokens": 900, "output_tokens": 450, "total_tokens": 1350}
    """
    parser               = JsonOutputParser()
    usage_metadata: dict = {}
    last_error           = None

    rate_limit_attempts = _MAX_RATE_LIMIT_RETRIES if USE_GEMINI else 1

    for rate_attempt in range(rate_limit_attempts):

        if USE_GEMINI and rate_attempt > 0:
            backoff = _RPM_BACKOFF_SECONDS[
                min(rate_attempt - 1, len(_RPM_BACKOFF_SECONDS) - 1)
            ]
            print(
                f"[quiz_chain] RPM backoff — waiting {backoff}s "
                f"(attempt {rate_attempt + 1}/{rate_limit_attempts})"
            )
            time.sleep(backoff)

        for parse_attempt in range(_MAX_QUIZ_RETRIES):
            try:
                print(
                    f"[quiz_chain] LLM call — "
                    f"rate attempt {rate_attempt + 1}/{rate_limit_attempts}, "
                    f"parse attempt {parse_attempt + 1}/{_MAX_QUIZ_RETRIES}"
                )

                response       = llm.invoke(prompt)
                usage_metadata = getattr(response, "usage_metadata", {}) or {}
                raw_text       = response.content

                try:
                    parsed = parser.parse(raw_text)
                    print("[quiz_chain] JsonOutputParser succeeded.")
                    return parsed, usage_metadata

                except Exception as parse_err:
                    print(
                        f"[quiz_chain] JsonOutputParser failed: {parse_err}. "
                        "Attempting json_repair fallback."
                    )
                    repaired = repair_json(raw_text)
                    parsed   = json.loads(repaired)
                    print("[quiz_chain] json_repair fallback succeeded.")
                    return parsed, usage_metadata

            except Exception as e:
                rate_kind = _classify_rate_error(e)

                if rate_kind == "rpd":
                    raise RateLimitExhaustedError(
                        "rpd",
                        f"Daily Gemini quota exhausted during quiz generation: {e}"
                    ) from e

                if rate_kind == "rpm":
                    print(
                        f"[quiz_chain] RPM rate limit hit: {e}. "
                        "Breaking to outer loop for backoff."
                    )
                    last_error = e
                    break

                last_error = e
                print(
                    f"[quiz_chain] LLM/parse error "
                    f"(parse attempt {parse_attempt + 1}/{_MAX_QUIZ_RETRIES}): {e}"
                )

        else:
            raise RuntimeError(
                f"All {_MAX_QUIZ_RETRIES} parse attempts failed. "
                f"Last error: {last_error}"
            ) from last_error

    raise RateLimitExhaustedError(
        "rpm",
        f"RPM rate limit unresolved after {rate_limit_attempts} attempts. "
        f"Last error: {last_error}"
    ) from last_error


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

@token_guard
def run_quiz_chain(
    student_id: str,
    course:     str,
    topic:      str,
    weak_areas: list[str],      # empty list = broad retrieval (pre-test / no history)
    store:      Chroma,
    llm:        BaseChatModel,
) -> Any:
    """
    Generates a 10-question, three-level assessment quiz grounded in retrieved
    course material for the given student, course, and topic.

    Assessment structure (fixed, pace-independent):
        3 Surface-Level  — recognition and recall
        4 Conceptual     — relationships and processes
        3 Deep-Level     — application and inference

    Questions are open-ended (no MCQ). Each question includes a rubric-style
    model_answer that serves as the marking scheme for eval_chain.

    student_answer, score, and explanation are scaffolded as null:
        student_answer → filled by the router after student submission
        score          → filled by eval_chain (float 0.0–1.0)
        explanation    → filled by eval_chain

    source_pages on every question are overridden from chunk metadata.
    LLM-generated page numbers are not trusted (hallucination confirmed).

    The generated quiz is saved to:
        data/quizzes/{student_id}_{topic_slug}_{timestamp}.json

    This chain is stateless beyond disk save. Attaching student answers,
    marking the revision date as complete, and returning results to the
    frontend are all the router's responsibility.

    Args:
        student_id (str):          Unique student identifier.
                                   Used by token_guard for quota tracking.
        course     (str):          Course name, e.g. "Computer Science".
                                   Injected into the prompt for context.
        topic      (str):          Topic within the course, e.g. "Sorting Algorithms".
                                   Used as the retrieval query via get_topic_chunks().
        weak_areas (list[str]):    Weak section names from the learner profile.
                                   Empty list → broad retrieval via get_topic_chunks().
                                   Non-empty → filtered retrieval via _get_weak_area_chunks()
                                   targeting only sections the student previously failed.
        store      (Chroma):       Initialised Chroma vector store scoped to
                                   this student's collection.
        llm        (BaseChatModel):Initialised LangChain chat model. Ollama
                                   locally; Gemini 2.5 in production.

    Returns:
        _SyntheticResponse: Object with three attributes:
            .content        (dict) — processed quiz with 10 questions
            .usage_metadata (dict) — token usage from the LLM call
            .saved_path     (Path) — path of the saved quiz JSON file

    Raises:
        ValueError:              If student_id, course, or topic are empty,
                                 or if no chunks are found for the topic.
        RateLimitExhaustedError: If Gemini quota is exhausted and retries
                                 are unrecoverable.
        RuntimeError:            If retrieval fails or all LLM/parse retries fail.
        IOError:                 If the quiz file cannot be saved to disk.

    Example:
        response = run_quiz_chain(
            student_id = "student_42",
            course     = "Computer Science",
            topic      = "Sorting Algorithms",
            store      = chroma_store,
            llm        = ollama_llm,
        )

        # response.content:
        {
            "questions": [
                {
                    "level":          "surface",
                    "question":       "What is the time complexity of Quick Sort in the average case?",
                    "model_answer":   "O(n log n) — award mark for correct complexity; \
accept 'n log n' without Big-O notation.",
                    "student_answer": null,
                    "score":          null,
                    "explanation":    null,
                    "source_pages":   [4, 7]
                },
                {
                    "level":          "conceptual",
                    "question":       "Explain how Merge Sort achieves stable sorting.",
                    "model_answer":   "Answer should reference that Merge Sort preserves \
the relative order of equal elements during the merge step; award mark for any \
correct description of stability through the merging process.",
                    "student_answer": null,
                    "score":          null,
                    "explanation":    null,
                    "source_pages":   [4, 7]
                },
                {
                    "level":          "deep",
                    "question":       "Predict how the choice of pivot affects Quick Sort \
performance on an already-sorted list.",
                    "model_answer":   "A poor pivot causes unbalanced partitioning leading \
to O(n²) worst-case; award mark for any reference to degraded time complexity \
or unbalanced partitions on sorted input.",
                    "student_answer": null,
                    "score":          null,
                    "explanation":    null,
                    "source_pages":   [4, 7]
                }
                # ... 7 more questions
            ]
        }

        # response.usage_metadata:
        {"input_tokens": 920, "output_tokens": 480, "total_tokens": 1400}

        # response.saved_path:
        PosixPath("data/quizzes/student_42_sorting_algorithms_20250503_142301.json")
    """

    # ── STEP 0: Input validation ──────────────────────────────────────────────
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError(
            "student_id must be a non-empty string. "
            f"Got: {repr(student_id)}"
        )
    if not isinstance(course, str) or not course.strip():
        raise ValueError(
            "course must be a non-empty string, e.g. 'Computer Science'. "
            f"Got: {repr(course)}"
        )
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(
            "topic must be a non-empty string, e.g. 'Sorting Algorithms'. "
            f"Got: {repr(topic)}"
        )

    print(
        f"[quiz_chain] Starting — "
        f"student: '{student_id}', course: '{course}', topic: '{topic}'"
    )

    # ── STEP 1: Retrieve chunks (adaptive) ───────────────────────────────────
    try:
        if not weak_areas:
            print(f"[quiz_chain] Broad retrieval — no weak areas on record.")
            chunks = get_topic_chunks(store, topic, course)
        else:
            print(
                f"[quiz_chain] Adaptive retrieval — "
                f"targeting {len(weak_areas)} weak section(s): {weak_areas}"
            )
            chunks = _get_weak_area_chunks(store, course, topic, weak_areas)

    except Exception as e:
        raise RuntimeError(
            f"Retrieval failed for topic '{topic}' in course '{course}': {e}"
        ) from e

    if not chunks:
        raise ValueError(
            f"No material found for topic '{topic}' in course '{course}'. "
            "Ensure the relevant PDF has been ingested and the topic name "
            "matches the material content."
        )

    print(f"[quiz_chain] Retrieved {len(chunks)} chunk(s) for topic '{topic}'.")

    # ── STEP 2: Build prompt ──────────────────────────────────────────────────
    formatted_chunks = _format_chunks(chunks)
    source_pages     = _extract_source_pages(chunks)

    prompt = QUIZ_PROMPT.format(
        course     = course.strip(),
        topic      = topic.strip(),
        chunks     = formatted_chunks,
        total      = _TOTAL_QUESTIONS,
        surface    = _SURFACE_COUNT,
        conceptual = _CONCEPTUAL_COUNT,
        deep       = _DEEP_COUNT,
    )

    print(
        f"[quiz_chain] Prompt built — "
        f"chunks: {len(chunks)}, source pages: {source_pages}"
    )

    # ── STEP 3: LLM call ─────────────────────────────────────────────────────
    try:
        parsed, usage_metadata = _call_quiz_llm(llm, prompt)
    except RateLimitExhaustedError:
        raise
    except RuntimeError:
        raise

    # ── STEP 4: Post-process & source mapping ─────────────────────────────────
    try:
        quiz = _validate_and_repair_quiz(parsed)
    except ValueError as e:
        raise ValueError(
            f"Quiz post-processing failed — unexpected LLM output structure: {e}"
        ) from e

    print("[quiz_chain] Running per-question source mapping...")
    quiz["questions"] = _map_sources_to_questions(
        questions = quiz["questions"],
        store     = store,
        course    = course,
        topic     = topic,
    )

    total_generated = len(quiz["questions"])
    print(
        f"[quiz_chain] Complete — "
        f"questions: {total_generated}/{_TOTAL_QUESTIONS}, "
        f"tokens: {usage_metadata.get('total_tokens', 'N/A')}"
    )

    # ── STEP 5: Save to disk ──────────────────────────────────────────────────
    saved_path = _save_quiz(quiz, student_id, topic)

    # ── STEP 6: Return ────────────────────────────────────────────────────────
    return _SyntheticResponse(
        content        = quiz,
        usage_metadata = usage_metadata,
        saved_path     = saved_path,
    )