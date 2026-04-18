
**Local implementation:**
- ChromaDB for vector store
- JSON files for user data and learner profile

**Final Learner Profile Structure:**

```json
{
  "user_id": "...",
  "learning_pace": "average",
  "tokens_remaining": 1000000,
  "tokens_used": 0,
  "token_history": [
    {
      "input_tokens": 312,
      "output_tokens": 688,
      "total_tokens": 1000,
      "chain": "run_qa_chain",
      "timestamp": "2026-04-14T10:23:01"
    },
    {
      "input_tokens": 890,
      "output_tokens": 1240,
      "total_tokens": 2130,
      "chain": "run_notes_chain",
      "timestamp": "2026-04-14T11:45:22"
    }
  ],
  "weak_topics": [],
  "strong_topics": [],
  "revision_dates": [],
  "topics": { ... }
}
```

---

**Documentation:**

---

# Learner Profile

Don't store token_history as a JSON column on Learner_Profiles.
Give it its own table instead:

Token_History
├── id              UUID    PRIMARY KEY
├── student_id      UUID    FOREIGN KEY → Learner_Profiles
├── chain           VARCHAR
├── input_tokens    INTEGER
├── output_tokens   INTEGER
├── total_tokens    INTEGER
├── timestamp       TIMESTAMPTZ

## Overview
The Learner Profile is a per-user JSON data structure that persists across sessions in Supabase. It is the central data structure driving CogniLearn's personalization engine — every adaptive decision made by the system, from how notes are generated to how quizzes are weighted, is derived from the state of this structure.

---

## Top-Level Keys

### `user_id`
A unique identifier linking this profile to the authenticated user in Supabase. Set once at account creation and never modified.

---

### `learning_pace`
A string value — `"fast"`, `"average"`, or `"slow"` — manually assigned at account creation based on the learner's CGPA bracket. Controls how the AI generates condensed notes by influencing prompt templates:
- **Fast** — assumes familiarity with prior concepts, minimal elaboration
- **Average** — balanced elaboration
- **Slow** — assumes prior concepts may have been forgotten, more detailed explanation

Never updated programmatically in this version.

---

### `weak_topics` and `strong_topics`
Top-level arrays listing topic names where the learner is globally weak or strong across all studied topics. A topic is classified as weak if its `weak_areas` outnumber its `strong_areas`, and strong if the reverse is true.

Both are **derived and stored** — recomputed and written to the profile after every quiz evaluation. Not utilized in the prototype since testers study a single topic, but the structure supports multi-topic use in future versions.

---

### `revision_dates`
A list of 5 ISO date strings representing scheduled revision sessions, generated from a fixed spaced repetition algorithm anchored to the date of the learner's first retention test.

**Spacing pattern from anchor date:**
- +1 day
- +2 days
- +1 week
- +2 weeks

**Generation trigger:** When the quiz generation function is called and both `revision_dates` and `scores.retention` are empty, it anchors to today's date, generates the 5 dates, and writes them to the profile. This fires exactly once — after the first retention test is completed.

**Example:** First retention test taken on `2025-04-11`:
```json
"revision_dates": [
  "2025-04-12",
  "2025-04-13",
  "2025-04-18",
  "2025-04-25"
]
```

---

### `topics`
A dictionary keyed by topic name (matching the `section` metadata field from the ingested PDF). Each topic entry tracks the learner's performance and progress for that specific topic.

---

## Topic-Level Keys

### `weak_areas` and `strong_areas`
Arrays of section names (matching `metadata["section"]` values from the vector store) identifying which sections of the topic the learner is weak or strong in.

**Updated by:** The quiz evaluation function after every quiz submission. The evaluation function marks the learner's answers, identifies which sections had incorrect answers (weak) and correct answers (strong), and returns them. The profile update function writes this output directly into these arrays, fully replacing their previous values to always reflect the learner's current state.

---

### `scores`

#### `comprehension`
Tracks pre and post-study comprehension test scores for this topic. Each entry is an object:
```json
{"attempt": 0, "score": 40, "date": "2025-04-01"}
```
- `attempt: 0` — pre-study comprehension (taken before reading the notes)
- `attempt: 1` — post-study comprehension (taken after reading the notes)

Used to measure how much the learner's understanding improved after studying. Maximum of 2 entries per topic in the current version, but the array structure supports additional attempts in future versions without schema changes.

**Updated by:** The quiz evaluation function after each comprehension test submission — appends a new entry with the next attempt index.

#### `retention`
Tracks spaced repetition quiz scores taken on scheduled revision dates. Each entry follows the same structure:
```json
{"attempt": 0, "score": 70, "date": "2025-04-10"}
```
Entries are appended after each revision session. Used to drive the spaced repetition scheduling — the date of `attempt: 0` anchors all `revision_dates`.

**Updated by:** The quiz evaluation function after each retention test submission — appends a new entry and, on the first attempt, triggers revision date generation.

### Token Economy:
    tokens_remaining    Tracks how many LLM tokens the student has left.
                        Initialised at 1,000,000 on signup. Decremented
                        after every chain call by the exact token count
                        returned in the model's response metadata.
                        Requests are blocked when this reaches 0.

    tokens_used         Cumulative total of tokens consumed by the student
                        across all chain calls (notes, quiz, eval, Q&A).
                        Used for cost tracking and research evaluation.

    Invariant:          tokens_remaining + tokens_used = 1,000,000 always.

### Important note
#### Supabase
token_service.py — the Supabase RPC note. When you implement deduct_tokens(), use a stored procedure rather than two separate UPDATE queries. Two separate queries can create a race condition if a student somehow triggers two chain calls simultaneously — the atomic RPC prevents that.

---

## Update Flow Summary

| Event | Keys Updated |
|---|---|
| Account creation | `user_id`, `learning_pace` |
| First quiz on a topic | `topics[topic]` entry created, `weak_areas`, `strong_areas` written |
| Any quiz submission | `weak_areas`, `strong_areas`, `weak_topics`, `strong_topics`, `scores.comprehension` or `scores.retention` |
| First retention test | `revision_dates` generated and written |