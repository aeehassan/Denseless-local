"""
quiz_router.py
─────────────────────────────────────────────────────────────────────────────
Parent orchestrator for the adaptive quiz pipeline.

Handles phase detection, notes_reviewed gating, spaced repetition schedule
generation, and adaptive retrieval routing before delegating to run_quiz_chain.

Pipeline phases (detected from learner profile array state):

    Phase 1 — Pre-Test:
        scores.comprehension is empty.
        Broad retrieval. No history yet.

    Phase 2 — Post-Test:
        scores.comprehension has exactly 1 entry AND notes_reviewed = True.
        Broad retrieval. Measures learning gain after notes intervention.
        Blocked if notes_reviewed = False — student must read notes first.

    Phase 3 — Schedule Generation:
        scores.retention has exactly 1 entry AND revision_dates[topic] is empty.
        Spaced repetition schedule generated and saved now.
        First revision date is tomorrow — quiz not served immediately.

    Phase 4 — Revision Quiz:
        revision_dates[topic] exists AND today matches a pending date.
        Adaptive retrieval — targets weak sections only if any exist.

This module is a plain Python callable (no FastAPI) for local dev and testing.
When FastAPI is integrated, wrap handle_quiz_request() in a POST route handler
inside app/routers/quiz.py with minimal changes.

Dev usage:
    from app.services.quiz_router import handle_quiz_request

    # 1. Standard run (uses current system date)
    result = handle_quiz_request(
        student_id = "student_42",
        course     = "Computer Science",
        topic      = "Sorting Algorithms",
        store      = chroma_store,
        llm        = ollama_llm,
    )

    # 2. Time-travel testing (simulates a future date to test scheduled quizzes)
    result = handle_quiz_request(
        student_id      = "student_42",
        course          = "Computer Science",
        topic           = "Sorting Algorithms",
        store           = chroma_store,
        llm             = ollama_llm,
        simulated_today = "2026-05-11"  # Optional: ISO date string - YYYY-MM-DD (Year-Month-Day)
    )

    if result["status"] == "ok":
        quiz = result["quiz"]       # dict — 10 questions
    else:
        print(result["message"])    # e.g. "Read your notes first."
"""

import json

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_community.vectorstores import Chroma

from app.agent.rag.chains.quiz_chain import run_quiz_chain

# ─────────────────────────────────────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────────────────────────────────────

_PROFILES_DIR = Path(__file__).parent.parent.parent / "data" / "profiles"

# Spaced repetition intervals in days (anchored to post-test date)
_REVISION_INTERVALS_DAYS = [1, 3, 7, 14]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_profile(student_id: str) -> dict:
    """
    Loads the learner profile JSON from disk.

    Args:
        student_id (str): Unique student identifier.

    Returns:
        dict: Parsed learner profile.

    Raises:
        FileNotFoundError: If the profile file does not exist.
        ValueError:        If the file is not valid JSON.

    Example:
        profile = _load_profile("student_1042")
        # profile → {"topics": {}, "scores": {...}, "revision_dates": {}, ...}
    """
    path = _PROFILES_DIR / f"{student_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Learner profile not found for student '{student_id}'. "
            f"Expected at: {path}"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Learner profile for '{student_id}' is not valid JSON: {e}"
        ) from e


def _save_profile(student_id: str, profile: dict) -> None:
    """
    Writes the updated learner profile back to disk.

    Args:
        student_id (str):  Unique student identifier.
        profile    (dict): Updated profile dict.

    Raises:
        IOError: If the file cannot be written.

    Example:
        _save_profile("student_42", updated_profile)
        # Overwrites data/profiles/student_42.json
    """
    path = _PROFILES_DIR / f"{student_id}.json"

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        print(f"[quiz_router] Profile saved: {path}")
    except Exception as e:
        raise IOError(
            f"Failed to save profile for student '{student_id}': {e}"
        ) from e


def _ensure_topic(profile: dict, topic: str) -> dict:
    """
    Ensures the topic key exists in the profile with all required fields.

    Initialises the topic structure if this is the first time the student
    is taking a quiz on this topic. Mutates profile in place.

    Args:
        profile (dict): Learner profile dict.
        topic   (str):  Topic name.

    Returns:
        dict: The topic sub-dict (initialised or existing).

    Example:
        topic_data = _ensure_topic(profile, "Sorting Algorithms")
        # topic_data → {"weak_areas": [], "strong_areas": []}
    """
    if "topics" not in profile:
        profile["topics"] = {}

    if topic not in profile["topics"]:
        profile["topics"][topic] = {
            "weak_areas":  [],
            "strong_areas": [],
        }
        print(f"[quiz_router] New topic initialised in profile: '{topic}'")

    if "scores" not in profile:
        profile["scores"] = {"comprehension": [], "retention": []}

    if "revision_dates" not in profile:
        profile["revision_dates"] = {}

    if "notes_reviewed" not in profile:
        profile["notes_reviewed"] = False

    return profile["topics"][topic]


def _generate_revision_schedule(anchor: date) -> list[dict]:
    """
    Generates the spaced repetition revision schedule anchored to a given date.

    Intervals: +1 day, +3 days, +7 days, +14 days from anchor.
    Each entry initialised with status 'pending' and feedback null.

    Args:
        anchor (date): The date to anchor the schedule to (post-test date).

    Returns:
        list[dict]: Four revision date entries in chronological order.

    Example:
        schedule = _generate_revision_schedule(date(2025, 5, 3))
        # [
        #     {"date": "2025-05-04", "status": "pending", "feedback": null},
        #     {"date": "2025-05-06", "status": "pending", "feedback": null},
        #     {"date": "2025-05-10", "status": "pending", "feedback": null},
        #     {"date": "2025-05-17", "status": "pending", "feedback": null},
        # ]
    """
    schedule = []
    for interval in _REVISION_INTERVALS_DAYS:
        target = anchor + timedelta(days=interval)
        schedule.append({
            "date":     target.isoformat(),
            "status":   "pending",
            "feedback": None,
        })

    print(
        f"[quiz_router] Revision schedule generated: "
        f"{[e['date'] for e in schedule]}"
    )
    return schedule


def _detect_phase(
    profile: dict,
    topic:   str,
    current_date: date,
) -> tuple[str, str | None]:
    """
    Inspects learner profile array state to determine the current quiz phase.

    Phase detection order (evaluated top to bottom — first match wins):

        "pre_test"           comprehension is empty
        "blocked"            comprehension has 1 entry, notes_reviewed = False
        "post_test"          comprehension has 1 entry, notes_reviewed = True
        "schedule_pending"   retention has 1 entry, revision_dates[topic] empty
        "revision"           revision_dates[topic] exists, today matches pending date
        "no_quiz_due"        revision_dates[topic] exists, today does not match
        "series_complete"    all revision dates completed

    Args:
        profile (dict): Loaded learner profile.
        topic   (str):  Topic name being requested.
        current_date (date): ...

    Returns:
        tuple[str, str | None]:
            phase   — one of the phase strings above
            message — human-readable string for non-ok phases, None for ok phases

    Example:
        phase, msg = _detect_phase(profile, "Sorting Algorithms")
        # ("pre_test", None)
        # ("blocked", "Please complete your study notes before taking the post-test.")
        # ("no_quiz_due", "Your next revision is in 3 day(s), on 2025-05-07.")
    """
    comprehension  = profile.get("scores", {}).get("comprehension", [])
    retention      = profile.get("scores", {}).get("retention", [])
    notes_reviewed = profile.get("notes_reviewed", False)
    revision_dates = profile.get("revision_dates", {}).get(topic, [])
    today = current_date.isoformat()

    # Phase 1 — Pre-Test
    if not comprehension:
        return "pre_test", None

    # Phase 2 — Post-Test (gated by notes_reviewed)
    if len(comprehension) == 1:
        if not notes_reviewed:
            return (
                "blocked",
                "Please complete your study notes before taking the post-test. "
                "Once you have reviewed the generated notes, you will be able "
                "to take the post-test."
            )
        return "post_test", None

    # Phase 3 — Schedule generation trigger
    if len(retention) == 1 and not revision_dates:
        return "schedule_pending", None

    # Phase 4 — Revision
    if revision_dates:
        # Check for a pending date matching today
        for entry in revision_dates:
            if entry["date"] == today and entry["status"] == "pending":
                return "revision", None

        # Check if all dates are completed
        if all(e["status"] == "completed" for e in revision_dates):
            return (
                "series_complete",
                "You have completed all scheduled revision quizzes for this topic. "
                "Well done."
            )

        # Find next pending date
        for entry in revision_dates:
            if entry["status"] == "pending":
                target        = date.fromisoformat(entry["date"])
                today_date = current_date
                days_remaining = (target - today_date).days
                return (
                    "no_quiz_due",
                    f"No quiz is due today. Your next revision is in "
                    f"{days_remaining} day(s), on {entry['date']}."
                )

    # Fallback — should not reach here under normal flow
    return (
        "no_quiz_due",
        "No quiz is currently scheduled. Complete the pre-test to begin."
    )

def _is_revision_already_done(profile: dict, topic: str, current_date: date) -> bool:
    """
    Checks if a retention (revision) quiz has already been completed today
    for the specific topic.
    """
    retention_scores = profile.get("scores", {}).get("retention", [])
    if not retention_scores:
        return False

    today_str = current_date.isoformat()
    
    # Check the date of the very last retention attempt
    last_attempt_date = retention_scores[-1].get("date")
    
    return last_attempt_date == today_str

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def handle_quiz_request(
    student_id: str,
    course:     str,
    topic:      str,
    store:      Chroma,
    llm:        BaseChatModel,
    simulated_today: str | None = None,
) -> dict[str, Any]:
    """
    Parent orchestrator for the adaptive quiz pipeline.

    Loads the learner profile, detects the current phase, applies the
    notes_reviewed gate, generates the spaced repetition schedule when
    triggered, and delegates quiz generation to run_quiz_chain with the
    correct retrieval strategy.

    Returns a response dict in all cases — never raises to the caller.
    The caller inspects response["status"] to determine next action.

    Args:
        student_id (str):          Unique student identifier.
        course     (str):          Course name, e.g. "Computer Science".
        topic      (str):          Topic name, e.g. "Sorting Algorithms".
        store      (Chroma):       Initialised Chroma vector store.
        llm        (BaseChatModel):Initialised LangChain chat model.

    Returns:
        dict: Always contains "status" key. Possible shapes:

            Success:
                {
                    "status":     "ok",
                    "phase":      "pre_test | post_test | revision",
                    "quiz":       { ...10 questions... },
                    "saved_path": "data/quizzes/student_42_sorting_20250503.json"
                }

            Blocked (notes not reviewed):
                {
                    "status":  "blocked",
                    "message": "Please complete your study notes first."
                }

            Not due yet:
                {
                    "status":         "no_quiz_due",
                    "message":        "Your next revision is in 3 day(s)...",
                    "next_quiz_date": "2025-05-07"
                }

            Series complete:
                {
                    "status":  "series_complete",
                    "message": "You have completed all scheduled revision quizzes."
                }

            Error:
                {
                    "status":  "error",
                    "message": "Descriptive error message."
                }

    Example:
        result = handle_quiz_request(
            student_id = "student_42",
            course     = "Computer Science",
            topic      = "Sorting Algorithms",
            store      = chroma_store,
            llm        = ollama_llm,
        )

        if result["status"] == "ok":
            print(result["phase"])   # "pre_test"
            quiz = result["quiz"]    # 10 questions

        elif result["status"] == "blocked":
            print(result["message"])  # "Please read your notes first."

        elif result["status"] == "no_quiz_due":
            print(result["message"])  # "Next revision in 3 day(s)..."
    """

    # ── Input validation ──────────────────────────────────────────────────────
    for name, val in [("student_id", student_id), ("course", course), ("topic", topic)]:
        if not isinstance(val, str) or not val.strip():
            return {
                "status":  "error",
                "message": f"'{name}' must be a non-empty string. Got: {repr(val)}",
            }

    print(
        f"[quiz_router] Request — "
        f"student: '{student_id}', course: '{course}', topic: '{topic}'"
    )

    # ── Dev Clock Override ──
    current_date = date.today()
    if simulated_today:
        current_date = date.fromisoformat(simulated_today)
        print(f"[quiz_router] 🕒 DEV MODE: Simulating today as {current_date.isoformat()}")

    # ── Load profile ──────────────────────────────────────────────────────────
    try:
        profile = _load_profile(student_id)
    except (FileNotFoundError, ValueError) as e:
        return {"status": "error", "message": str(e)}

    # Ensure topic structure exists — initialises if first time
    topic_data = _ensure_topic(profile, topic)

    # ── Phase detection ───────────────────────────────────────────────────────
    phase, message = _detect_phase(profile, topic, current_date)

    print(f"[quiz_router] Phase detected: '{phase}'")

    # ── Daily Limit Gate (Revision Only) ──────────────────────────────────────
    if phase == "revision":
        if _is_revision_already_done(profile, topic, current_date):
            return {
                "status": "blocked",
                "message": (
                    "You have already completed your revision for today. "
                    "Great job! See you at your next scheduled date."
                )
            }

    # ── Non-quiz phases — return early ────────────────────────────────────────
    if phase == "blocked":
        return {"status": "blocked", "message": message}

    if phase == "no_quiz_due":
        # Extract next_quiz_date from revision_dates for convenience
        revision_dates = profile.get("revision_dates", {}).get(topic, [])
        next_date      = next(
            (e["date"] for e in revision_dates if e["status"] == "pending"),
            None,
        )
        return {
            "status":         "no_quiz_due",
            "message":        message,
            "next_quiz_date": next_date,
        }

    if phase == "series_complete":
        return {"status": "series_complete", "message": message}

    # ── Phase 3 — Schedule generation ────────────────────────────────────────
    if phase == "schedule_pending":
        print("[quiz_router] Phase 3 triggered — generating revision schedule.")
        schedule = _generate_revision_schedule(anchor=current_date)
        profile["revision_dates"][topic] = schedule

        try:
            _save_profile(student_id, profile)
        except IOError as e:
            return {"status": "error", "message": str(e)}

        # First revision date is tomorrow — no quiz served today
        return {
            "status":  "no_quiz_due",
            "message": (
                f"Great work completing your post-test. Your revision schedule "
                f"has been set. First revision: {schedule[0]['date']}."
            ),
            "next_quiz_date": schedule[0]["date"],
        }

    # ── Phases that serve a quiz: pre_test, post_test, revision ──────────────

    # Determine adaptive retrieval strategy
    weak_areas = topic_data.get("weak_areas", [])

    # Pre-test and post-test always use broad retrieval
    if phase in ("pre_test", "post_test"):
        weak_areas = []

    print(
        f"[quiz_router] Retrieval strategy — "
        f"{'broad' if not weak_areas else f'adaptive ({len(weak_areas)} section(s))'}"
    )

    # ── Call quiz chain ───────────────────────────────────────────────────────
    try:
        response = run_quiz_chain(
            student_id = student_id,
            course     = course,
            topic      = topic,
            weak_areas = weak_areas,
            store      = store,
            llm        = llm,
        )
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {
            "status":  "error",
            "message": f"Unexpected error during quiz generation: {e}",
        }

    print(f"[quiz_router] Quiz generated successfully for phase '{phase}'.")

    return {
        "status":     "ok",
        "phase":      phase,
        "quiz":       response.content,
        "saved_path": str(response.saved_path),
    }