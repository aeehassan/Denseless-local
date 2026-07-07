"""
eval_chain.py
─────────────────────────────────────────────────────────────────────────────
Evaluation chain for CogniLearn.

Grades a submitted quiz against rubric-style marking schemes, classifies
section mastery using the weakest-link rule, generates personalised session
feedback, and updates the learner profile.

Does NOT touch the vector store.
"""

import json
import time
import logging
from datetime import date
from pathlib import Path
from typing import Any

from json_repair import repair_json
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.services.token_service import token_guard

logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

USE_GEMINI: bool = False  # False → local Ollama, no delays, retry loop runs once
# True  → Gemini API, inter-call delays + RPM backoff active

_REQUEST_DELAY_SECONDS = 5
_RPM_BACKOFF_SECONDS = [15, 30, 60]
_MAX_RATE_LIMIT_RETRIES = 3

PASS_THRESHOLD = 0.7
PROFILES_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "profiles"


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
        self.kind = kind  # "rpm" | "rpd"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


class _SyntheticResponse:
    """
    Wraps eval_chain output for token_guard compatibility.

    token_guard reads usage_metadata to deduct from the student's token budget.

    Attributes:
        content        (dict): Fully graded quiz dict.
        usage_metadata (dict): Accumulated token counts across all LLM calls.
    """

    def __init__(self, content: dict, usage_metadata: dict):
        self.content = content
        self.usage_metadata = usage_metadata


def _classify_rate_error(e: Exception) -> str:
    """
    Classifies a rate-limit exception by inspecting its message string.

    Args:
        e: The caught exception.

    Returns:
        "rpd"   — daily quota exceeded ("daily" or "quota" in message).
        "rpm"   — requests-per-minute limit hit ("429", "rate", or
                  "resource_exhausted" in message).
        "other" — unrecognised error type.
    """
    msg = str(e).lower()
    if "daily" in msg or "quota" in msg:
        return "rpd"
    if "429" in msg or "rate" in msg or "resource_exhausted" in msg:
        return "rpm"
    return "other"


def _accumulate_tokens(base: dict, new: dict) -> dict:
    """
    Merges two usage_metadata dicts by summing matching token count keys.

    Args:
        base: Running total accumulated from previous LLM calls.
        new:  Token counts returned from the latest LLM call.

    Returns:
        Dict with summed token counts across both inputs.

    Example:
        >>> _accumulate_tokens(
        ...     {"input_tokens": 300, "output_tokens": 120},
        ...     {"input_tokens": 210, "output_tokens": 85},
        ... )
        {"input_tokens": 510, "output_tokens": 205}
    """
    result = dict(base)
    for key, value in new.items():
        if not isinstance(value, int):  # skip nested dicts like input_token_details
            continue
        result[key] = result.get(key, 0) + value
    return result


def _load_profile(student_id: str) -> dict:
    """
    Loads a learner profile from disk.

    Args:
        student_id: Unique student identifier.

    Returns:
        Parsed profile dict.

    Raises:
        FileNotFoundError: If no profile file exists for this student.
        ValueError:        If the profile file contains invalid JSON.

    Example:
        >>> profile = _load_profile("student_42")
        >>> profile["student_id"]
        "student_42"
    """
    profile_path = PROFILES_DIR / f"{student_id}.json"

    if not profile_path.exists():
        raise FileNotFoundError(
            f"Profile not found for student '{student_id}'. "
            f"Expected path: {profile_path}"
        )

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Profile file for student '{student_id}' contains invalid JSON. "
            f"Parser error: {e}"
        ) from e


def _save_profile(student_id: str, profile: dict) -> None:
    """
    Writes an updated learner profile back to disk.

    Args:
        student_id: Unique student identifier.
        profile:    The mutated profile dict to persist.

    Raises:
        OSError: If the file cannot be written.

    Example:
        >>> _save_profile("student_42", updated_profile)
        [eval_chain] Profile saved for student 'student_42'.
    """
    profile_path = PROFILES_DIR / f"{student_id}.json"

    try:
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=4)
        print(f"[eval_chain] Profile saved for student '{student_id}'.")
    except OSError as e:
        raise OSError(
            f"Failed to write profile for student '{student_id}' "
            f"to {profile_path}. OS error: {e}"
        ) from e


def _save_quiz(quiz_path: Path, quiz: dict) -> None:
    """
    Writes the fully graded quiz dict back to its JSON file.

    Args:
        quiz_path: Absolute or relative path to the quiz JSON file.
        quiz:      The mutated quiz dict with scores and explanations filled.

    Raises:
        OSError: If the file cannot be written.

    Example:
        >>> _save_quiz(Path("data/quizzes/student_42_sorting_algorithms_1234.json"), graded_quiz)
        [eval_chain] Graded quiz saved to data/quizzes/student_42_sorting_algorithms_1234.json
    """
    try:
        with open(quiz_path, "w", encoding="utf-8") as f:
            json.dump(quiz, f, indent=4)
        print(f"[eval_chain] Graded quiz saved to {quiz_path}")
    except OSError as e:
        raise OSError(
            f"Failed to write graded quiz to {quiz_path}. OS error: {e}"
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_GRADING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict but fair academic examiner grading a student's open-ended answer.

You will receive:
- A question
- A marking scheme (model_answer) — this is a rubric string, not a model essay. It may \
contain inline cues for acceptable paraphrases and alternative answers.
- The student's answer

Your job:
1. Award a score between 0.0 and 1.0.
   - 1.0  = fully correct, or correctly expressed in the student's own words
   - 0.5  = partially correct (right idea, missing a key component)
   - 0.0  = incorrect, irrelevant, or blank
   - Use the full float range — partial credit is expected and important.
2. Write a one-sentence explanation telling the student why they received that score.
   - Base it directly on what the student wrote and the score you awarded.
   - Do NOT re-teach or elaborate. State what was right, partially right, or wrong.
   - Examples:
       "Correct — you identified the energy production role accurately."
       "Partial — you mentioned energy loss but missed the ATP production mechanism."
       "Incorrect — your answer describes photosynthesis, not cellular respiration."

CRITICAL RULES:
- Accept paraphrase. If the student expresses the correct idea in different words, \
award full or near-full credit.
- Honour every inline alternative stated in the marking scheme. \
If the scheme says "accept X or Y", treat both as fully valid.
- Do not penalise for spelling or grammar unless the answer is genuinely unintelligible.
- Do not be harsher than what the marking scheme intends.
- Never award 0.0 to an answer that contains a partially correct idea.

Respond ONLY with a valid JSON object. No preamble, no markdown, no extra text.

Format:
{{
    "score": <float 0.0–1.0>,
    "explanation": "<one-sentence grader note>"
}}""",
        ),
        (
            "human",
            """Question: {question}

Marking Scheme: {model_answer}

Student Answer: {student_answer}""",
        ),
    ]
)


_FEEDBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an academic coach providing personalised post-quiz feedback to a student.

You will receive:
- The topic of the quiz
- The student's total score out of 10
- Their strong sections (where they performed well)
- Their weak sections (where they struggled)

Write exactly five sentences of feedback structured strictly as follows:
1. Overall performance summary — state the score and what level of understanding it reflects.
2. What the student did well — name the strong sections specifically and acknowledge the achievement.
3. What needs attention — name the weak sections specifically, without discouraging the student.
4. A concrete, actionable suggestion for improving the weak areas quickly \
(e.g. specific study strategy, what to focus on).
5. A closing motivational note that is grounded in their actual result — \
not generic, tied to where they are right now.

Tone: Warm but direct. Academic but not cold. Encouraging without being hollow.
Format: Flowing prose only. No bullet points, no headers, no lists.

Respond ONLY with a valid JSON object. No preamble, no markdown, no extra text.

Format:
{{
    "feedback": "<five-sentence feedback string>"
}}""",
        ),
        (
            "human",
            """Topic: {topic}
Total Score: {total_score} / 10
Strong Sections: {strong_sections}
Weak Sections: {weak_sections}""",
        ),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — SINGLE QUESTION GRADER
# ─────────────────────────────────────────────────────────────────────────────


def _grade_question(
    question: str,
    model_answer: str,
    student_answer: str,
    llm: BaseChatModel,
) -> tuple[dict, dict]:
    """
    Grades a single question via the LLM grader using the double retry loop.

    Outer loop → rate-limit retries (RPM backoff or immediate RPD raise).
    Inner loop → parse-failure retries (JsonOutputParser → json_repair fallback).

    Args:
        question:       The question text.
        model_answer:   The rubric/marking scheme string from the quiz.
        student_answer: The student's submitted answer.
        llm:            The language model instance.

    Returns:
        Tuple of (result_dict, usage_metadata_dict).
        result_dict keys: "score" (float 0.0–1.0), "explanation" (str).

    Raises:
        RateLimitExhaustedError: If RPM retries are exhausted or RPD quota is hit.
        RuntimeError:            If the response cannot be parsed after all inner retries.

    Example:
        >>> result, usage = _grade_question(
        ...     question="What is O(n log n) complexity?",
        ...     model_answer="O(n log n) — award mark for correct complexity; \
accept 'n log n' without Big-O notation.",
        ...     student_answer="The time complexity is n log n.",
        ...     llm=llm,
        ... )
        >>> result
        {"score": 1.0, "explanation": "Correct — you identified n log n complexity accurately."}
    """
    parser = JsonOutputParser()
    chain = _GRADING_PROMPT | llm

    rpm_attempt = 0

    while rpm_attempt <= _MAX_RATE_LIMIT_RETRIES:
        parse_attempt = 0
        raw_response = None

        while parse_attempt < 3:
            try:
                raw_response = chain.invoke(
                    {
                        "question": question,
                        "model_answer": model_answer,
                        "student_answer": student_answer,
                    }
                )
                # print(f"[DEBUG] usage_metadata raw: {raw_response.usage_metadata}")
                usage_metadata = raw_response.usage_metadata or {}

                # Primary parse attempt
                try:
                    result = parser.parse(raw_response.content)
                    print(f"[eval_chain] Question graded. Score: {result.get('score')}")
                    return result, usage_metadata

                except Exception as parse_err:
                    print(
                        f"[eval_chain] JsonOutputParser failed (attempt {parse_attempt + 1}/3). "
                        f"Trying json_repair. Error: {parse_err}"
                    )
                    # Fallback parse attempt
                    try:
                        repaired = repair_json(raw_response.content)
                        result = json.loads(repaired)
                        print(
                            f"[eval_chain] json_repair succeeded. Score: {result.get('score')}"
                        )
                        return result, usage_metadata
                    except Exception as repair_err:
                        print(
                            f"[eval_chain] json_repair failed (attempt {parse_attempt + 1}/3). "
                            f"Error: {repair_err}"
                        )
                        parse_attempt += 1

            except Exception as e:
                kind = _classify_rate_error(e)

                if kind == "rpd":
                    raise RateLimitExhaustedError(
                        kind="rpd",
                        message=(
                            f"Daily quota exhausted during question grading. "
                            f"Original error: {e}"
                        ),
                    ) from e

                if kind == "rpm":
                    if rpm_attempt >= _MAX_RATE_LIMIT_RETRIES:
                        raise RateLimitExhaustedError(
                            kind="rpm",
                            message=(
                                f"RPM rate limit hit and all {_MAX_RATE_LIMIT_RETRIES} "
                                f"retries exhausted during question grading. "
                                f"Original error: {e}"
                            ),
                        ) from e

                    backoff = _RPM_BACKOFF_SECONDS[
                        min(rpm_attempt, len(_RPM_BACKOFF_SECONDS) - 1)
                    ]
                    print(
                        f"[eval_chain] RPM limit hit. Backing off {backoff}s "
                        f"(attempt {rpm_attempt + 1}/{_MAX_RATE_LIMIT_RETRIES})."
                    )
                    time.sleep(backoff)
                    rpm_attempt += 1
                    break  # restart outer loop

                raise RuntimeError(
                    f"Unexpected error during question grading. "
                    f"Error type: {type(e).__name__}. Error: {e}"
                ) from e

        else:
            # Inner loop exhausted without a successful parse or rate-limit break
            raise RuntimeError(
                f"Failed to parse grading response after 3 parse attempts. "
                f"Raw response: {raw_response.content if raw_response else 'None'}"
            )

    # Safety net — should not be reached under normal retry flow
    raise RateLimitExhaustedError(
        kind="rpm",
        message="RPM retries exhausted during question grading.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — SESSION FEEDBACK GENERATOR
# ─────────────────────────────────────────────────────────────────────────────


def _generate_session_feedback(
    topic: str,
    total_score: float,
    new_strong_sections: set,
    new_weak_sections: set,
    llm: BaseChatModel,
) -> tuple[str, dict]:
    """
    Generates a five-sentence personalised feedback string for the student.

    Single LLM call (not per-question). Same double retry loop applies.

    Args:
        topic:               The quiz topic.
        total_score:         Sum of all question scores (float, max 10.0).
        new_strong_sections: Sections where the student scored >= PASS_THRESHOLD
                             on all questions (after weakest-link resolution).
        new_weak_sections:   Sections where at least one question scored < PASS_THRESHOLD.
        llm:                 The language model instance.

    Returns:
        Tuple of (feedback_string, usage_metadata_dict).

    Raises:
        RateLimitExhaustedError: If RPM retries are exhausted or RPD quota is hit.
        RuntimeError:            If response cannot be parsed after all inner retries.

    Example:
        >>> feedback, usage = _generate_session_feedback(
        ...     topic="Sorting Algorithms",
        ...     total_score=6.5,
        ...     new_strong_sections={"Merge Sort Mechanics"},
        ...     new_weak_sections={"Algorithm Analysis", "Complexity Theory"},
        ...     llm=llm,
        ... )
        >>> feedback
        "You scored 6.5 out of 10, reflecting a developing but incomplete grasp of \
Sorting Algorithms. Your understanding of Merge Sort Mechanics stood out as a genuine \
strength, showing you have a solid foundation in that area. Algorithm Analysis and \
Complexity Theory are where you lost most marks and need focused attention before your \
next session. To close those gaps quickly, go back to your notes and trace through \
algorithm steps manually, focusing on how to express time complexity in Big-O notation. \
You are clearly building momentum — sharpening your analytical skills here will bring \
the rest of your understanding together."
    """
    parser = JsonOutputParser()
    chain = _FEEDBACK_PROMPT | llm

    # Format sets as readable strings for the prompt
    strong_str = (
        ", ".join(sorted(new_strong_sections))
        if new_strong_sections
        else "none identified"
    )
    weak_str = (
        ", ".join(sorted(new_weak_sections)) if new_weak_sections else "none identified"
    )

    rpm_attempt = 0

    while rpm_attempt <= _MAX_RATE_LIMIT_RETRIES:
        parse_attempt = 0
        raw_response = None

        while parse_attempt < 3:
            try:
                raw_response = chain.invoke(
                    {
                        "topic": topic,
                        "total_score": total_score,
                        "strong_sections": strong_str,
                        "weak_sections": weak_str,
                    }
                )
                usage_metadata = raw_response.usage_metadata or {}

                # Primary parse attempt
                try:
                    result = parser.parse(raw_response.content)
                    feedback = result["feedback"]
                    print(f"[eval_chain] Session feedback generated.")
                    return feedback, usage_metadata

                except Exception as parse_err:
                    print(
                        f"[eval_chain] Feedback parse failed (attempt {parse_attempt + 1}/3). "
                        f"Trying json_repair. Error: {parse_err}"
                    )
                    # Fallback parse attempt
                    try:
                        repaired = repair_json(raw_response.content)
                        result = json.loads(repaired)
                        feedback = result["feedback"]
                        print(f"[eval_chain] Feedback json_repair succeeded.")
                        return feedback, usage_metadata
                    except Exception as repair_err:
                        print(
                            f"[eval_chain] Feedback json_repair failed "
                            f"(attempt {parse_attempt + 1}/3). Error: {repair_err}"
                        )
                        parse_attempt += 1

            except Exception as e:
                kind = _classify_rate_error(e)

                if kind == "rpd":
                    raise RateLimitExhaustedError(
                        kind="rpd",
                        message=(
                            f"Daily quota exhausted during feedback generation. "
                            f"Original error: {e}"
                        ),
                    ) from e

                if kind == "rpm":
                    if rpm_attempt >= _MAX_RATE_LIMIT_RETRIES:
                        raise RateLimitExhaustedError(
                            kind="rpm",
                            message=(
                                f"RPM retries exhausted during feedback generation. "
                                f"Original error: {e}"
                            ),
                        ) from e

                    backoff = _RPM_BACKOFF_SECONDS[
                        min(rpm_attempt, len(_RPM_BACKOFF_SECONDS) - 1)
                    ]
                    print(
                        f"[eval_chain] RPM limit hit during feedback. Backing off {backoff}s "
                        f"(attempt {rpm_attempt + 1}/{_MAX_RATE_LIMIT_RETRIES})."
                    )
                    time.sleep(backoff)
                    rpm_attempt += 1
                    break  # restart outer loop

                raise RuntimeError(
                    f"Unexpected error during feedback generation. "
                    f"Error type: {type(e).__name__}. Error: {e}"
                ) from e

        else:
            raise RuntimeError(
                f"Failed to parse feedback response after 3 parse attempts. "
                f"Raw response: {raw_response.content if raw_response else 'None'}"
            )

    # Safety net — should not be reached under normal retry flow
    raise RateLimitExhaustedError(
        kind="rpm",
        message="RPM retries exhausted during feedback generation.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


@token_guard
def run_eval_chain(
    student_id: str,
    topic: str,
    quiz_phase: str,
    quiz_path: str | Path,
    llm: BaseChatModel,
    simulated_date: str = None,
) -> Any:
    """
    Grades a submitted quiz, updates the learner profile, and returns
    the fully annotated quiz dict.

    Implements the full seven-step eval algorithm:
        Step 1 — Grade each question individually via LLM (score + explanation).
        Step 2 — Compute total_score as sum of all question scores (max 10.0).
        Step 3 — Classify sections into weak/strong using PASS_THRESHOLD = 0.7.
        Step 4 — Apply weakest-link rule (weak sections cannot be strong).
        Step 5 — Generate five-sentence session feedback via single LLM call.
        Step 6 — Update learner profile:
                     a. Overwrite weak_areas / strong_areas for the topic.
                     b. Route score into comprehension/retention by quiz_phase.
                     c. Mark today's revision_dates entry as completed.
        Step 7 — Save profile and return _SyntheticResponse.

    Args:
        student_id: Unique student identifier.
        topic:      The topic the quiz covers.
        quiz_phase: "pre_test" | "post_test" | "revision"
        quiz_path:  path leading to full quiz dict with student_answer filled on every question.
        llm:        Language model instance.
        simulated_date: The date I plan to do an eval for

    Returns:
        _SyntheticResponse:
            content        → fully graded quiz dict (score + explanation on every question)
            usage_metadata → accumulated token counts across all LLM calls

    Raises:
        ValueError:              On invalid inputs or missing required quiz keys.
        FileNotFoundError:       If the learner profile does not exist on disk.
        RateLimitExhaustedError: If the LLM rate limit cannot be recovered from.
        RuntimeError:            On unexpected LLM or parsing failures.

    Example:
        >>> response = run_eval_chain(
        ...     student_id="student_42",
        ...     topic="Sorting Algorithms",
        ...     quiz_phase="post_test",
        ...     quiz=submitted_quiz_dict,
        ...     llm=llm,
        ... )
        >>> response.content["questions"][0]["score"]
        0.8
        >>> response.content["feedback"]
        "..."
        >>> response.content["questions"][0]["explanation"]
        "Correct — you identified merge sort's divide-and-conquer structure accurately."
        >>> response.usage_metadata
        {"input_tokens": 4320, "output_tokens": 910}
    """

    # ── Input validation ──────────────────────────────────────────────────────

    if not student_id or not isinstance(student_id, str):
        raise ValueError(
            f"'student_id' must be a non-empty string. Got: {repr(student_id)}"
        )

    if not topic or not isinstance(topic, str):
        raise ValueError(f"'topic' must be a non-empty string. Got: {repr(topic)}")

    valid_phases = {"pre_test", "post_test", "revision"}
    if quiz_phase not in valid_phases:
        raise ValueError(
            f"'quiz_phase' must be one of {valid_phases}. Got: {repr(quiz_phase)}"
        )

    quiz_path = Path(quiz_path)

    if not quiz_path.exists():
        raise FileNotFoundError(
            f"Quiz file not found at '{quiz_path}'. "
            f"Expected a valid path to a saved quiz JSON."
        )

    if quiz_path.suffix != ".json":
        raise ValueError(
            f"'quiz_path' must point to a .json file. Got: '{quiz_path.suffix}'"
        )

    # ── Load quiz from path (replaces the quiz dict parameter) ────────────

    try:
        with open(quiz_path, "r", encoding="utf-8") as f:
            quiz = json.load(f)
        print(f"[eval_chain] Quiz loaded from {quiz_path}")
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Quiz file at '{quiz_path}' contains invalid JSON. Parser error: {e}"
        ) from e

    if not isinstance(quiz, dict) or "questions" not in quiz:
        raise ValueError(
            f"'quiz' must be a dict containing a 'questions' key. "
            f"Got type: {type(quiz).__name__}"
        )

    questions = quiz["questions"]
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError(
            f"'quiz[\"questions\"]' must be a non-empty list. Got: {repr(questions)}"
        )

    required_keys = ("question", "model_answer", "student_answer", "section")
    for i, q in enumerate(questions):
        for key in required_keys:
            if key not in q:
                raise ValueError(
                    f"Question at index {i} is missing required key '{key}'. "
                    f"Keys present: {list(q.keys())}"
                )

    print(
        f"[eval_chain] Starting eval — "
        f"student: '{student_id}' | topic: '{topic}' | phase: '{quiz_phase}' | "
        f"questions: {len(questions)}"
    )

    # ── Step 1 — Grade each question ──────────────────────────────────────────

    accumulated_usage: dict = {}

    for i, question in enumerate(questions):
        print(f"[eval_chain] Grading question {i + 1}/{len(questions)}...")

        if USE_GEMINI:
            time.sleep(_REQUEST_DELAY_SECONDS)

        result, usage = _grade_question(
            question=question["question"],
            model_answer=question["model_answer"],
            student_answer=question["student_answer"] or "",
            llm=llm,
        )

        # Mutate score and explanation in place on the question dict
        question["score"] = float(result["score"])
        question["explanation"] = result["explanation"]

        accumulated_usage = _accumulate_tokens(accumulated_usage, usage)

    print(f"[eval_chain] All {len(questions)} questions graded.")

    # ── Step 2 — Compute total score ──────────────────────────────────────────

    total_score = round(sum(q["score"] for q in questions), 2)
    print(f"[eval_chain] Total score: {total_score} / {len(questions)}")

    # ── Step 3 — Classify sections ────────────────────────────────────────────

    new_weak_sections: set = set()
    new_strong_sections: set = set()

    for q in questions:
        section = q["section"]
        if q["score"] < PASS_THRESHOLD:
            new_weak_sections.add(section)
        else:
            new_strong_sections.add(section)

    print(
        f"[eval_chain] Pre-resolution — strong: {new_strong_sections} | weak: {new_weak_sections}"
    )

    # ── Step 4 — Weakest-link rule ────────────────────────────────────────────

    # One failed question in a section disqualifies it from being marked strong
    new_strong_sections = new_strong_sections - new_weak_sections

    print(
        f"[eval_chain] Post-resolution — strong: {new_strong_sections} | weak: {new_weak_sections}"
    )

    # ── Step 5 — Generate session feedback ────────────────────────────────────

    if USE_GEMINI:
        time.sleep(_REQUEST_DELAY_SECONDS)

    session_feedback, feedback_usage = _generate_session_feedback(
        topic=topic,
        total_score=total_score,
        new_strong_sections=new_strong_sections,
        new_weak_sections=new_weak_sections,
        llm=llm,
    )
    # Store feedback as part of the response object
    quiz["feedback"] = session_feedback
    accumulated_usage = _accumulate_tokens(accumulated_usage, feedback_usage)

    # ── Step 6 — Update learner profile ──────────────────────────────────────

    profile = _load_profile(student_id)
    today = date.today().isoformat()  # "YYYY-MM-DD"

    if simulated_date:
        today = simulated_date

    # 6a — Ensure topic + nested score containers exist
    profile.setdefault("scores", {})
    profile["scores"].setdefault("comprehension", {})
    profile["scores"].setdefault("retention", {})

    if topic not in profile.get("topics", {}):
        profile.setdefault("topics", {})[topic] = {
            "weak_areas": [],
            "strong_areas": [],
        }

    # Inject empty per-topic score arrays if this topic hasn't been seen before
    profile["scores"]["comprehension"].setdefault(topic, [])
    profile["scores"]["retention"].setdefault(topic, [])

    profile["topics"][topic]["weak_areas"] = list(new_weak_sections)
    profile["topics"][topic]["strong_areas"] = list(new_strong_sections)

    print(f"[eval_chain] Section classifications updated for topic '{topic}'.")

    # 6b — Route score into comprehension / retention arrays by quiz_phase
    if quiz_phase == "pre_test":
        profile["scores"]["comprehension"][topic].append(
            {"attempt": 0, "score": total_score, "date": today}
        )
        print(f"[eval_chain] Pre-test score → comprehension[{topic}] (attempt 0).")

    elif quiz_phase == "post_test":
        profile["scores"]["comprehension"][topic].append(
            {"attempt": 1, "score": total_score, "date": today}
        )
        profile["scores"]["retention"][topic].append(
            {"attempt": 0, "score": total_score, "date": today}
        )
        print(
            f"[eval_chain] Post-test score → comprehension[{topic}] (attempt {1}) "
            f"and retention[{topic}] (attempt 0)."
        )

    elif quiz_phase == "revision":
        retention_attempt = len(profile["scores"]["retention"][topic])
        profile["scores"]["retention"][topic].append(
            {"attempt": retention_attempt, "score": total_score, "date": today}
        )
        print(
            f"[eval_chain] Revision score → retention[{topic}] (attempt {retention_attempt})."
        )

    # 6c — Mark today's revision_dates entry as completed with feedback
    revision_dates_for_topic = profile.get("revision_dates", {}).get(topic, [])

    for entry in revision_dates_for_topic:
        if entry.get("date") == today:
            entry["feedback"] = session_feedback
            entry["status"] = "completed"
            print(f"[eval_chain] Revision date entry for {today} marked completed.")
            break

    # ── Step 7 — Save quiz, save profile, and return ──────────────────────

    _save_quiz(quiz_path, quiz)  # ← save graded quiz back to its file
    _save_profile(student_id, profile)

    print(f"[eval_chain] Eval chain complete for student '{student_id}'.")

    return _SyntheticResponse(
        content=quiz,
        usage_metadata=accumulated_usage,
    )
