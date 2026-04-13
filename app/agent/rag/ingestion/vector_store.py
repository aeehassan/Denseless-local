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
#   from app.agent.rag.ingestion.vector_store import get_vector_store
#
#   store = get_vector_store(student_id="abc123", embedder=embedder)
# ══════════════════════════════════════════════════════════════════════════════

import os
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
    documents: List[Document],
) -> None:
    """
    Add a list of LangChain Document objects to a student's Chroma store.

    Each Document's ``page_content`` is embedded using the store's
    embedding function and persisted alongside its full ``metadata`` dict.
    This preserves section, category, page_number, and all other fields
    attached during the ingestion step.

    Skips silently if ``documents`` is empty to avoid unnecessary
    model calls.

    Args:
        store:      A student-scoped Chroma store (from get_chroma_store).
        documents:  List of chunked Document objects from ingestion.

    Returns:
        None
    """
    try:
        if not documents:
            print("  ⚠ No documents provided — skipping Chroma insertion.")
            return

        print(f"Adding {len(documents)} document chunk(s) to Chroma store...")
        store.add_documents(documents)
        print(f"  → Successfully added {len(documents)} chunk(s).")
    except Exception as e:
        print(f"Error adding documents to vector store: {e}")
        raise


def query_chroma(
    store: Chroma,
    query: str,
    top_k: int = 5,
) -> List[Document]:
    """
    Run a semantic similarity search scoped to one student's Chroma store.

    Because the store was initialised with the student's collection,
    this search is physically isolated — it cannot surface documents
    belonging to any other student.

    The query string is embedded at runtime using the same model used
    during ingestion, then ranked by cosine similarity against all
    stored vectors in this collection.

    Args:
        store:  A student-scoped, populated Chroma store.
        query:  The question or topic string to search for.
        top_k:  Number of most relevant chunks to return (default 5).

    Returns:
        A list of up to ``top_k`` Document objects, most relevant first,
        each retaining its full metadata from ingestion.
    """
    print(f"Querying Chroma (top_k={top_k}): '{query[:60]}...'")

    try:
        results = store.similarity_search(query=query, k=top_k)

        print(f"  → Retrieved {len(results)} chunk(s).")
        return results
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
