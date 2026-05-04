"""
app/agent/rag/retriever.py

Retrieval layer — sits between the vector store and the four chains.
Exposes two retrieval strategies suited to different chain purposes.

Retrieval Strategies:
  get_semantic_chunks()  → cosine similarity search driven by a query string
                           Used by: qa_chain, eval_chain

  get_topic_chunks()     → full store sweep grouped by section heading
                           Used by: notes_chain, quiz_chain

Page Limit:
  get_topic_chunks() is designed for PDFs up to 50 pages.
  Although Gemini 2.5 Pro (~2,400 pages) and Llama3.2:3b (~290 pages)
  can technically accommodate more, 50 pages is enforced as a practical
  UX boundary — beyond this, condensed notes lose coherence and quiz
  quality degrades.

Backend Agnostic:
  Both functions call query_store() from vector_store.py exclusively.
  They never call query_chroma() or query_supabase() directly.
  Switching backends only requires changes in vector_store.py.

Usage:
  from app.agent.rag.retrieval.retriever import get_semantic_chunks, get_topic_chunks

  # For qa_chain / eval_chain
  chunks = get_semantic_chunks(query="What is gradient descent?", store=store)

  # For notes_chain / quiz_chain
  topic_map = get_topic_chunks(store=store)
"""

from typing import Dict, List
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.agent.rag.ingestion.vector_store import query_store


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# BGE models require this prefix on query strings at retrieval time.
# Document chunks are embedded without a prefix during ingestion.
# Omitting this at query time reduces retrieval accuracy noticeably.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Hard cap on pages passed to notes_chain and quiz_chain.
_MAX_PAGES = 50

# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPER — SECTION CONTEXT EXPANSION
# ─────────────────────────────────────────────────────────────────────────────
 
 
def _expand_with_section_context(
    seed_chunks: List[Document],
    store: Chroma,
) -> List[Document]:
    """
    Expand a list of seed chunks with all sibling chunks that share
    the same section name in the vector store.
 
    This addresses the limitation of small unstructured chunks — instead
    of returning only the matched fragment, the full section context is
    returned so the LLM has enough material to generate a quality answer.
 
    Algorithm:
        1. Collect all unique section names from seed_chunks
        2. For each section, fetch every chunk tagged with that section
        3. Merge seed chunks + section neighbours into one pool
        4. Deduplicate by page_content — same content = same knowledge
           regardless of element ID
        5. Sort by (page_number, position) to preserve reading order
 
    This is a best-effort operation. If section expansion fails for any
    reason, the original seed_chunks are returned unchanged so the chain
    always gets something rather than crashing.
 
    Args:
        seed_chunks:  The initially retrieved chunks from query_store().
        store:        The student-scoped Chroma vector store.
 
    Returns:
        Expanded, deduplicated, reading-order sorted list of Documents.
    """
    if not seed_chunks:
        return seed_chunks
 
    # ── Step 1: Collect unique section names from seed chunks ─────────────────
    sections = {
        chunk.metadata.get("section")
        for chunk in seed_chunks
        if chunk.metadata.get("section")  # skip chunks with no section tag
    }
 
    if not sections:
        print("[Retriever] ⚠ No section metadata found — skipping expansion, returning seed chunks.")
        return seed_chunks
 
    print(f"[Retriever] Expanding context across {len(sections)} section(s): {sections}")
 
    # ── Step 2: Fetch all chunks per unique section ───────────────────────────
    expansion_pool: List[Document] = list(seed_chunks)  # start with seeds
 
    for section in sections:
        try:
            section_chunks = store.similarity_search(
                query=" ",
                k=10_000,
                filter={"section": section},
            )
            expansion_pool.extend(section_chunks)
            print(f"[Retriever]   → Section '{section[:50]}': {len(section_chunks)} chunk(s) fetched.")
 
        except Exception as e:
            # Non-fatal — log the failure and continue with other sections
            print(
                f"[Retriever] ⚠ Section expansion failed for '{section[:50]}': {e}. "
                f"Continuing with remaining sections."
            )
            continue
 
    # ── Step 3: Deduplicate by page_content ───────────────────────────────────
    # Same content = same knowledge regardless of element_id.
    # First occurrence wins — seeds appear first so they're always kept.
    seen_contents: set[str] = set()
    unique_chunks: List[Document] = []
 
    for chunk in expansion_pool:
        content = chunk.page_content.strip()
        if content and content not in seen_contents:
            seen_contents.add(content)
            unique_chunks.append(chunk)
 
    # ── Step 4: Sort by page_number to restore reading order ─────────────────
    # Gives the LLM context in the same order a student would read the PDF.
    unique_chunks.sort(
        key=lambda c: (
            c.metadata.get("doc_index"),
        )
    )
 
    print(
        f"[Retriever] → Expansion complete: {len(seed_chunks)} seed → "
        f"{len(unique_chunks)} unique chunk(s) after deduplication."
    )
 
    return unique_chunks


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — SEMANTIC RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────


def get_semantic_chunks(
    query: str,
    store: Chroma,
    top_k: int = 5,
    score_threshold: float = 0.5,
    course: str | None = None,
    topic: str | None = None,
    condensed: bool = False
) -> List[Document]:
    """
    Retrieve semantically relevant chunks for a query, enriched with
    full section context to compensate for small unstructured chunk sizes.
 
    Two-phase retrieval:
        Phase 1 — Semantic search:
            Query the vector store with a BGE-prefixed query string.
            Returns the top_k most relevant chunks above score_threshold.
            These are the "seed" chunks — precise but potentially too small
            to give the LLM enough context for a quality answer.
 
        Phase 2 — Section context expansion:
            For each seed chunk, all sibling chunks sharing the same
            section name are fetched and merged into the result.
            This ensures the LLM receives the full surrounding context
            of every matched chunk — not just the matched fragment.
            Duplicates are removed by page_content comparison.
            Final list is sorted by page_number (reading order).
 
    Score Threshold:
        The minimum confidence level (0.0–1.0) for a chunk to be
        considered relevant to the query. Think of it as a cut-off
        grade — chunks that score below it are discarded, chunks
        above it are returned. A passing chunk must be both
        mathematically similar (vector distance) and contextually
        relevant to the query.
 
        0.5 → balanced, recommended starting point
        0.1 → too lenient, returns irrelevant chunks
        0.9 → too strict, may return nothing at all
 
        Note: Tune this per chain through observation —
        eval_chain may need a higher threshold (0.65) than
        qa_chain (0.5) since eval requires more precise grounding.
 
    Used by:
        qa_chain   — student's free-form question drives the search
        eval_chain — AI's quiz question drives the search for ground truth
 
    Args:
        query:            The student's question or the AI's quiz question.
        store:            The student-scoped vector store.
        top_k:            Number of seed chunks from initial search (default 5).
                          Must be a positive integer. Final result count will
                          be higher due to section expansion.
        score_threshold:  Minimum relevance confidence (default 0.5).
                          Must be between 0.0 and 1.0 inclusive.
 
    Returns:
        Expanded, deduplicated list of Document objects in reading order
        (ascending page_number). Returns an empty list if no chunks pass
        the score threshold — never raises on empty results so the chain
        can handle it gracefully.
 
    Raises:
        ValueError:   If query is empty, store is None, or arguments
                      are out of valid range.
        RuntimeError: If the initial vector store query fails unexpectedly.
                      Section expansion failures are non-fatal and logged.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not query or not query.strip():
        raise ValueError("query cannot be empty or whitespace.")
 
    if store is None:
        raise ValueError(
            "store cannot be None. Ensure the vector store has been "
            "initialised via get_vector_store() before calling this function."
        )
 
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(
            f"top_k must be a positive integer. Got: {top_k}"
        )
 
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError(
            f"score_threshold must be between 0.0 and 1.0. Got: {score_threshold}"
        )
 
    # ── Phase 1: Semantic search ──────────────────────────────────────────────
    # Apply BGE prefix — critical for retrieval accuracy with bge models
    prefixed_query = f"{_BGE_QUERY_PREFIX}{query.strip()}"
 
    print(f"[Retriever] Phase 1 — Semantic search (top_k={top_k}, threshold={score_threshold}): '{query[:60]}'")
 
    try:
        seed_chunks = query_store(
            store=store,
            query=prefixed_query,
            top_k=top_k,
            score_threshold=score_threshold,
        )
    except Exception as e:
        raise RuntimeError(
            f"Vector store query failed in get_semantic_chunks(). "
            f"Query: '{query[:60]}' | Error: {e}"
        ) from e
 
    print(f"[Retriever] → {len(seed_chunks)} seed chunk(s) passed threshold.")
 
    if not seed_chunks:
        # No results above threshold — return early, nothing to expand
        print("[Retriever] ⚠ No chunks passed threshold. Consider lowering score_threshold.")
        return []
 
    # ── Phase 2: Section context expansion ───────────────────────────────────
    print("[Retriever] Phase 2 — Section context expansion...")
 
    enriched_chunks = _expand_with_section_context(
        seed_chunks=seed_chunks,
        store=store,
    )
 
    print(f"[Retriever] → Final result: {len(enriched_chunks)} chunk(s) returned to chain.")
    return enriched_chunks


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — TOPIC SWEEP RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_section_key(text: str) -> str:
    """
    Normalise a section name for deduplication comparison.
    Lowercases and collapses internal whitespace so variations
    like "Interconnection Structures" and "interconnection structures"
    are treated as the same section.

    Example:
        "Interconnection  Structures" → "interconnection structures"
        "RISC Architecture"           → "risc architecture"
    """
    return re.sub(r'\s+', ' ', text).strip().lower()

def get_topic_chunks(
    store: Chroma,
    topic: str,
    course: str,
) -> Dict[str, List[Document]]:
    """
    Retrieve all chunks from the store grouped by section heading.

    Performs a full sweep of the student's vector store and groups
    every chunk by its ``metadata["section"]`` field. All chunks
    per section are returned without any limit — this preserves the
    complete meaning of the student's material and ensures notes_chain
    and quiz_chain have full coverage of every section in the PDF.

    Context Window Safety:
        No per-section chunk limit is enforced here because the 50-page
        PDF cap applied at ingestion time already guarantees the total
        chunk volume stays within the context windows of both
        Gemini 2.5 Pro and Llama3.2:3b. Trimming chunks further would
        only degrade output quality without any safety benefit.

    Used by:
        notes_chain  — needs every chunk to produce complete condensed notes
        quiz_chain   — needs full coverage for a balanced assessment across
                       all topics in the PDF

    Args:
        store: The student-scoped vector store. Must not be None.
        topic: The specific topic chunks are being fetched for.
        course: The specific course that topic resides in

    Returns:
        Dict mapping section heading → list of all Document objects
        in that section, ordered by page_number (ascending).

        Example:
        {
            "Memory Management":          [doc1, doc2, doc3],
            "RISC Architecture":          [doc4, doc5, doc6, doc7],
            "Interconnection Structures": [doc8],
        }

        Sections are ordered by their first appearance in the PDF
        (ascending page_number of their first chunk).

    Raises:
        ValueError:   If store is None or the vector store is empty.
        RuntimeError: If the vector store query fails unexpectedly.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if store is None:
        raise ValueError(
            "store cannot be None. Ensure the vector store has been "
            "initialised via get_vector_store() before calling this function."
        )

    print("[Retriever] Topic sweep — fetching all chunks from store...")

    # ── Fetch every chunk in the student's collection ─────────────────────────
    # Large top_k ensures nothing is missed. No score threshold —
    # we want every chunk, not just the most similar ones.
    try:
        all_chunks = store.similarity_search(query=" ", k=10_000, filter={"$and": [{"course": course}, {"topic": topic}]})
    except Exception as e:
        raise RuntimeError(
            f"Vector store sweep failed in get_topic_chunks(). "
            f"Ensure the store is initialised and accessible. | Error: {e}"
        ) from e

    if not all_chunks:
        raise ValueError(
            "Vector store is empty. Ensure the PDF has been ingested "
            "via process_and_load_file() before calling get_topic_chunks()."
        )

    # ── Group chunks by section, merging duplicate section names ──────────────
    raw_groups: Dict[str, List[Document]] = {}
    key_to_canonical: Dict[str, str] = {}  # normalised key → first-seen original casing
    merge_count = 0

    for chunk in all_chunks:
        section = chunk.metadata.get("section", "General")
        norm_key = _normalise_section_key(section)

        if norm_key in key_to_canonical:
            # Seen this section before under a different casing — merge into
            # the first-seen canonical name
            canonical = key_to_canonical[norm_key]
            raw_groups[canonical].append(chunk)
            merge_count += 1
        else:
            # First time seeing this section — register original casing as canonical
            key_to_canonical[norm_key] = section
            raw_groups[section] = [chunk]

    if merge_count:
        print(f"[Retriever] → {merge_count} chunk(s) merged into existing sections (duplicate titles collapsed).")

    print(f"[Retriever] → {len(all_chunks)} chunk(s) across {len(raw_groups)} section(s).")

    # ── Sort sections by first page appearance ────────────────────────────────
    # Preserves natural reading order of the PDF in the output dict.
    def section_first_page(section_chunks: List[Document]) -> int:
        pages = [
            c.metadata.get("page_number", 0)
            for c in section_chunks
            if c.metadata.get("page_number") is not None
        ]
        return min(pages) if pages else 0

    # ── Sort chunks within each section by page number ────────────────────────
    topic_map: Dict[str, List[Document]] = {}

    for section, chunks in sorted(
        raw_groups.items(),
        key=lambda kv: section_first_page(kv[1])
    ):
        topic_map[section] = chunks

    print(f"[Retriever] → Topic map built: {len(topic_map)} section(s), all chunks retained.")

    return topic_map
