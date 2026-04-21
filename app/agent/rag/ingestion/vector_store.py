# ══════════════════════════════════════════════════════════════════════════════
# app/agent/rag/ingestion/vector_store.py
#
# Vector store abstraction layer for the CogniLearn RAG pipeline.
#
# Pipeline stage handled here:
#   Storage & Retrieval — Persists embedded document chunks and exposes
#   a similarity search interface for the retriever step.
#
# Multi-user design:
#   Every function is scoped to a student_id. Each student's documents
#   live in an isolated collection (Chroma) or filtered partition
#   (Supabase pgvector). No student can ever retrieve another's content.
#
# Active backend  : Chroma (local development + testing)
# Future backend  : Supabase pgvector (production)
#
# To switch backends, set the VECTOR_STORE environment variable:
#   VECTOR_STORE=chroma     (default)
#   VECTOR_STORE=supabase   (when ready for production)
#
# Usage:
#   from app.agent.rag.ingestion.vector_store import get_vector_store, add_documents_to_chroma
#
#   store = get_vector_store(student_id="abc123", embedder=embedder)
#   add_documents_to_chroma(store, embeddings, documents, course, topic, file_name)
# ══════════════════════════════════════════════════════════════════════════════

import os
import numpy as np
import uuid
from typing import List

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────────────────────────────────────


def _get_collection_name(student_id: str) -> str:
    """
    Derives a consistent, namespaced Chroma collection name from a student ID.

    This is an internal helper — nothing outside this module should ever
    construct a collection name manually. All functions go through this
    so the naming convention stays consistent across the entire codebase.

    In Chroma  : becomes the collection name  → "student_abc123"
    In Supabase: becomes a WHERE filter value → WHERE student_id = 'abc123'

    Args:
        student_id: The unique identifier for the student.

    Returns:
        A namespaced string safe to use as a Chroma collection name.
    """
    return f"student_{student_id}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CHROMA (active implementation)
# ─────────────────────────────────────────────────────────────────────────────


def get_chroma_store(
    student_id: str,
    embedder: HuggingFaceEmbeddings,
    persist_directory: str = "../chroma_db",
) -> Chroma:
    """
    Initialise or load an existing Chroma vector store for a specific student.

    Each student gets their own isolated Chroma collection derived from
    their ``student_id``. This means similarity searches are always
    scoped — a query against student A's store cannot return chunks
    belonging to student B.

    If the collection already exists on disk (e.g. the student previously
    uploaded a PDF), it is loaded rather than recreated — no duplicate
    chunks are added.

    Args:
        student_id:         Unique identifier for the student.
        embedder:           Initialised HuggingFaceEmbeddings instance.
        persist_directory:  Local path where Chroma persists to disk.
                            Defaults to ``../chroma_db`` (dev only).

    Returns:
        A ``Chroma`` instance scoped exclusively to this student.
    """
    try:
        collection_name = _get_collection_name(student_id)

        print(f"Initialising Chroma store — student: '{student_id}'")
        print(f"  → Collection : {collection_name}")
        print(f"  → Persist dir: {persist_directory}")

        store = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=persist_directory,
        )

        print(
            f"  → Chroma store ready. \nDocuments in collection: {store._collection.count()}"
        )
        return store
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise


def add_documents_to_chroma(
    store: Chroma,
    embeddings: np.ndarray,
    documents: List[Document],
    condensed: bool,
    course: str,
    topic: str,
    file_name: str,
) -> None:
    """
    Add a list of LangChain Document objects to a student's Chroma store.

    Before inserting, checks the existing collection for duplicate content.
    Documents whose page_content already exists in the store are silently
    skipped — only genuinely new documents are inserted. This prevents
    duplicate chunks from accumulating across notebook reruns or repeated
    ingestion calls on the same PDF.

    Deduplication is performed by hashing each document's page_content.
    Two documents with identical content are considered the same chunk
    regardless of their metadata or element_id.

    Documents and embeddings are filtered in sync — the final lists
    passed to Chroma are always guaranteed to be the same length.

    Args:
        store:      A student-scoped Chroma store (from get_chroma_store).
        embeddings: Corresponding embeddings for the documents as a
                    numpy array. Must be the same length as documents.
        documents:  List of chunked Document objects from ingestion.
        condensed:  Whether the documents are condensed or not.
        course:     The course these documents belong to.
        topic:      The topic these documents belong to.
        file_name:  The name of the source PDF file.

    Returns:
        None

    Raises:
        ValueError:   If documents and embeddings lengths do not match.
        RuntimeError: If fetching existing documents or inserting new
                      ones fails unexpectedly.
    """
    if not documents:
        print("  ⚠ No documents provided — skipping Chroma insertion.")
        return

    if len(documents) != len(embeddings):
        raise ValueError(
            f"Number of documents ({len(documents)}) must match "
            f"number of embeddings ({len(embeddings)})."
        )

    # ── Step 1: Fetch existing content hashes from the collection ────────────
    try:
        existing = store.get()
        existing_contents = existing.get("documents", [])

        existing_hashes: set[int] = {
            hash(content.strip()) for content in existing_contents if content
        }

        print(
            f"  [vector_store] {len(existing_hashes)} existing chunk(s) found in collection."
        )

    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch existing documents from Chroma store "
            f"during deduplication check. | Error: {e}"
        ) from e

    # ── Step 2: Filter documents AND embeddings in sync ───────────────────────
    # Zip together so both lists stay aligned after filtering.
    new_pairs = [
        (doc, emb)
        for doc, emb in zip(documents, embeddings)
        if hash(doc.page_content.strip()) not in existing_hashes
    ]

    skipped = len(documents) - len(new_pairs)

    if skipped > 0:
        print(f"  [vector_store] {skipped} duplicate chunk(s) skipped.")

    if not new_pairs:
        print(
            "  [vector_store] ✓ All documents already exist in store — nothing to add."
        )
        return

    # Unzip back into separate lists for Chroma insertion
    new_documents, new_embeddings = zip(*new_pairs)

    # ── Step 3: Build Chroma insertion payload ────────────────────────────────
    try:
        print(
            f"  [vector_store] Adding {len(new_documents)} new chunk(s) to Chroma store..."
        )

        ids = []
        metadatas = []
        documents_text = []
        embedding_list = []

        # Standardised source naming
        source = f"{course}_{topic}_{file_name}"

        for i, (doc, embedding) in enumerate(zip(new_documents, new_embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadatum = dict(doc.metadata)
            metadatum["course"] = course
            metadatum["topic"] = topic
            metadatum["condensed"] = condensed
            metadatum["source"] = source
            metadatum["doc_index"] = i
            metadatum["content_length"] = len(doc.page_content)
            metadatas.append(metadatum)

            documents_text.append(doc.page_content)
            embedding_list.append(embedding.tolist())

        store._collection.add(
            ids=ids,
            documents=documents_text,
            metadatas=metadatas,
            embeddings=embedding_list,
        )

        print(f"  [vector_store] ✓ Successfully added {len(new_documents)} chunk(s).")
        print(
            f"  [vector_store] Total documents in collection: {store._collection.count()}"
        )

    except Exception as e:
        raise RuntimeError(
            f"Failed to add documents to Chroma store. "
            f"Attempted to insert {len(new_documents)} chunk(s). | Error: {e}"
        ) from e


def query_chroma(
    store: Chroma,
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.5,
    course: str | None = None,
    topic: str | None = None,
) -> List[tuple[Document, float]]:
    """
    Run a semantic similarity search scoped to one student's Chroma store.
    It is also designed to perform filtering such that only the course and
    topic in question are returned.

    Because the store was initialised with the student's collection,
    this search is physically isolated — it cannot surface documents
    belonging to any other student.

    The query string is embedded at runtime using the same model used
    during ingestion, then ranked by cosine similarity. Only chunks
    that meet or exceed ``score_threshold`` are returned.

    Args:
        store:            A student-scoped, populated Chroma store.
        query:            The question or topic string to search for.
        top_k:            Number of most relevant chunks to return (default 5).
        score_threshold:  The minimum confidence level (0.0–1.0) for a chunk
                          to be considered relevant to the query.
                          Think of it as a cut-off grade — chunks that score
                          below it are discarded, chunks above it are returned.
                          A passing chunk must be both mathematically similar
                          (vector distance) and contextually relevant to the query.

                          0.5 → balanced, recommended starting point
                          0.1 → too lenient, returns irrelevant chunks
                          0.9 → too strict, may return nothing at all
        course:           The course to filter by.
        topic:            The topic to filter by.

    Returns:
        A list of up to ``top_k`` Document objects, most relevant first,
        each retaining its full metadata from ingestion.
    """
    print(
        f"Querying Chroma (top_k={top_k}, threshold={score_threshold}): '{query[:60]}...'"
    )

    try:
        # To enable filter search as well as full search
        filters = {}

        if course is not None:
            filters["course"] = course

        if topic is not None:
            filters["topic"] = topic

        # similarity_search_with_score returns (Document, score) tuples
        # Chroma uses L2 distance — lower score = more similar
        # We convert to a 0.0–1.0 relevance scale for a consistent interface
        results_with_scores = store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filters if filters else None,  # ← only pass filter when non-empty
        )

        # Remove duplicate chunks by content
        # Chroma sometimes returns identical or near-identical chunks
        # This block ensures each unique chunk appears only once
        seen_contents = set()
        unique_results = []

        for doc, score in results_with_scores:
            content = doc.page_content.strip()

            if content not in seen_contents:
                seen_contents.add(content)
                unique_results.append((doc, score))

        # Replace original results with deduplicated version
        results_with_scores = unique_results

        # Filter out chunks that fall below the score threshold
        filtered = [
            doc
            for doc, score in results_with_scores
            if (1 - score) >= score_threshold  # convert distance → similarity
        ]

        print(
            f"  → Retrieved {len(results_with_scores)} chunk(s), {len(filtered)} passed threshold."
        )
        return filtered
    except Exception as e:
        print(f"Error querying vector store: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SUPABASE / PGVECTOR (placeholders — not yet implemented)
#
# When implementing:
#   - student_id becomes a metadata column stored on every row
#   - Queries add WHERE student_id = ? to scope results per student
#   - Same isolation guarantee as Chroma, different mechanism
#   - Only these functions change — nothing in retriever.py or chains
# ─────────────────────────────────────────────────────────────────────────────


# TODO: implement when switching to Supabase
def get_supabase_store(
    student_id: str,
    embedder: HuggingFaceEmbeddings,
):
    """
    Initialise a Supabase pgvector store scoped to a specific student.

    Will use SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from environment.
    student_id will be stored as a metadata column on every row so that
    all queries can filter with WHERE student_id = ? for isolation.

    Args:
        student_id: Unique identifier for the student.
        embedder:   Initialised embedding model instance.
    """
    pass


# TODO: implement when switching to Supabase
def add_documents_to_supabase(
    student_id: str,
    documents: List[Document],
) -> None:
    """
    Add LangChain Document objects to the Supabase pgvector table.

    Will inject student_id into every document's metadata before
    insertion so the row is always scoped to the correct student.
    Mirrors the signature and behaviour of add_documents_to_chroma().

    Args:
        student_id: Unique identifier for the student.
        documents:  List of chunked Document objects from ingestion.
    """
    pass


# TODO: implement when switching to Supabase
def query_supabase(
    student_id: str,
    query: str,
    top_k: int = 5,
) -> List[Document]:
    """
    Run a similarity search against the Supabase pgvector store,
    filtered to a specific student's documents.

    Will support both:
      - Semantic retrieval  (cosine similarity on query vector)
      - Hybrid retrieval    (topic metadata filter + cosine similarity)

    Args:
        student_id: Unique identifier for the student.
        query:      The question or topic string to search for.
        top_k:      Number of most relevant chunks to return.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FACTORY
# ─────────────────────────────────────────────────────────────────────────────


def query_store(
    store: Chroma,
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.5,
    course: str | None = None,
    topic: str | None = None,
    backend: str | None = None,
) -> List[Document]:
    """
    Factory — routes a similarity search to the correct backend.

    This is the single query entry point retriever.py uses.
    It means retriever.py never imports query_chroma() or
    query_supabase() directly — only this function. When the
    backend switches to Supabase, only this function changes
    internally. retriever.py touches nothing.

    Args:
        store:            An initialised, student-scoped vector store.
        query:            The question or topic string to search for.
        top_k:            Number of most relevant chunks to return.
        score_threshold:  Minimum relevance score for a chunk to pass.
                          See query_chroma() for full definition.
        course:           The course to filter by.
        topic:            The topic to filter by.
        backend:          Override the env variable programmatically.
                          Useful in tests. Defaults to None (reads from env).

    Returns:
        A list of relevant Document objects with full metadata intact.

    Raises:
        ValueError:          If backend is not "chroma" or "supabase".
        NotImplementedError: If "supabase" is requested before implemented.
    """
    resolved_backend = (backend or os.getenv("VECTOR_STORE", "chroma")).lower().strip()

    if resolved_backend == "chroma":
        return query_chroma(
            store=store,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            course=course,
            topic=topic,
        )

    elif resolved_backend == "supabase":
        raise NotImplementedError(
            "Supabase query backend is not yet implemented. "
            "Set VECTOR_STORE=chroma for local development."
        )

    else:
        raise ValueError(
            f"Unrecognised vector store backend: '{resolved_backend}'. "
            f"Valid options are: 'chroma', 'supabase'."
        )


def get_vector_store(
    student_id: str,
    embedder: HuggingFaceEmbeddings,
    backend: str | None = None,
) -> Chroma:
    """
    Factory — returns the correct vector store scoped to a student.

    This is the single entry point the rest of the pipeline uses.
    Swapping from Chroma to Supabase in production requires only:
      1. Setting VECTOR_STORE=supabase in .env
      2. Implementing the three Supabase placeholder functions above
    No changes needed in retriever.py or any chain file.

    Args:
        student_id: Unique identifier for the student.
        embedder:   Initialised embedding model instance.
        backend:    Override the env variable programmatically.
                    Useful in tests. Defaults to None (reads from env).

    Returns:
        An initialised, student-scoped vector store instance.

    Raises:
        ValueError:          If backend is not "chroma" or "supabase".
        NotImplementedError: If "supabase" is requested before implemented.
    """
    resolved_backend = (backend or os.getenv("VECTOR_STORE", "chroma")).lower().strip()

    print(f"Vector store backend: '{resolved_backend}' | student: '{student_id}'")

    if resolved_backend == "chroma":
        return get_chroma_store(
            student_id=student_id,
            embedder=embedder,
        )

    elif resolved_backend == "supabase":
        # Placeholder — will call get_supabase_store() once implemented
        raise NotImplementedError(
            "Supabase pgvector backend is not yet implemented. "
            "Set VECTOR_STORE=chroma for local development."
        )

    else:
        raise ValueError(
            f"Unrecognised vector store backend: '{resolved_backend}'. "
            f"Valid options are: 'chroma', 'supabase'."
        )
