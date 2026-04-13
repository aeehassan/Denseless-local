"""
DenseLess — RAG Embeddings Module
===================================
Handles the instantiation of the text embedding model.

Pipeline stage handled here:
    Text Embedding — Converts text chunks (or user queries)
    into numerical vector representations using a HuggingFace
    sentence-transformers model.

Usage:
    from app.agent.rag.ingestion.embeddings import get_embedding_model, generate_embeddings

    embedder = get_embedding_model()
    vectors = generate_embeddings(docs, embedder)

==================================
Note
==================================
BGE models expect a prefix on query strings at retrieval time:
python

# When embedding a document chunk — no prefix needed
"Gradient descent is an optimisation algorithm..."

# When embedding a user's query — add this prefix
"Represent this sentence for searching relevant passages: What is gradient descent?"

This is only relevant in your retriever.py, not in embeddings.py
"""

from typing import List
import numpy as np
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────
# TEXT EMBEDDINGS
# ─────────────────────────────────────────────────────────────────


def get_embedding_model(
    model_name: str = "BAAI/bge-base-en-v1.5",  # The model performed well on academic and technical retrieval benchmarks
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """
    Initialize and return the sentence-transformers embedding model.

    By default, it uses 'all-MiniLM-L6-v2', which is a fast and
    lightweight model excellent for general-purpose semantic search.

    Args:
        model_name: The HuggingFace model repo ID to download/use.
        device:     'cpu' or 'cuda' (if GPU is available).

    Returns:
        An instance of ``HuggingFaceEmbeddings`` ready to be passed
        to a vector store or LangChain pipeline.
    """
    print(f"Loading embedding model: {model_name} (device={device})")

    # model_kwargs configures underlying torch settings
    model_kwargs = {"device": device}

    # encode_kwargs ensures embeddings are L2 normalized (critical for cosine similarity)
    encode_kwargs = {"normalize_embeddings": True}

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        print(
            f"  → Embedding model loaded successfully. Embedding dimension: {embeddings.client.get_sentence_embedding_dimension()}"
        )
        return embeddings

    except Exception as e:
        print(f"  ✗ Failed to load embedding model '{model_name}': {e}")
        # Reraise or handle gracefully depending on architectural needs
        raise RuntimeError(f"Embedding model initialization failed: {e}") from e


def generate_embeddings(
    documents: List[Document], embedder: HuggingFaceEmbeddings, batch_size: int = 32
) -> np.ndarray:
    """
    Generate vector embeddings for a list of LangChain Document objects.

    Iterates over the provided chunks in batches to display a progress bar,
    extracts their text content, and generates vectors using the given embedding model.

    Args:
        documents:  A list of Document chunks generated from the ingestion process.
        embedder:   An initialized HuggingFaceEmbeddings model instance.
        batch_size: Number of documents to embed at once (for progress bar chunking).

    Returns:
        A NumPy array of vector embeddings (shape: [num_documents, embedding_dim]).
    """
    if not documents:
        print("No documents provided to embed.")
        return np.array([])

    print(f"Generating embeddings for {len(documents)} document chunk(s)...")

    # Extract only the textual content from each Document object to embed
    texts = [doc.page_content for doc in documents]
    all_embeddings = []

    # Process with a progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding documents"):
        batch = texts[i : i + batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    # Convert list to a numpy array
    embeddings_array = np.array(all_embeddings)

    print(f"  → Successfully generated {len(embeddings_array)} embedding vectors.")
    print(f"  → Embeddings shape: {embeddings_array.shape}")

    return embeddings_array
