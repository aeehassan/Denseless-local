"""
DenseLess — RAG Data Ingestion Module
=======================================
Handles loading a single PDF and splitting it into semantically
meaningful chunks based on the document's own section structure.

Uses ``langchain_unstructured.UnstructuredLoader`` with the
``"by_title"`` chunking strategy so that headings / sections in
the PDF naturally define chunk boundaries — ideal for a study
app where preserving topic semantic structure.

Pipeline stages handled here:
    1. PDF loading  — extract text + structural metadata.
    2. Chunking     — split by document sections (titles/headings).

Usage:
    from app.agent.rag.ingestion.data_ingestion import process_file

    chunks = process_file("path/to/lecture.pdf")
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document

# ── Load environment variables from .env ─────────────────────────
load_dotenv()


# ─────────────────────────────────────────────────────────────────
# PDF LOADING + SECTION-AWARE CHUNKING
# ─────────────────────────────────────────────────────────────────


def process_and_load_file(
    file_path: str,
    max_characters: int = 1000,
    overlap: int = 200,
) -> List[Document]:
    """
    Load a single PDF and chunk it by document sections.
    It can only handle text based pdfs and not OCR ones.

    The ``"by_title"`` strategy groups consecutive elements
    (paragraphs, lists, etc.) under the same heading into a
    single chunk, preserving semantic context.

    Args:
        file_path:      Path to the PDF file.
        max_characters: Soft upper limit on chunk size (default 1000).
        overlap:        Character overlap between chunks (default 200).

    Returns:
        A list of chunked ``Document`` objects with rich metadata
        (category, page number, element_id, etc.).

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError:        If *file_path* is not a .pdf file.
        ValueError:        If the API key or URL is not found.
    """
    pdf = Path(file_path)

    if not pdf.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf}")

    if not pdf.match("*.pdf"):
        raise ValueError(f"File is not a PDF: {pdf.name}")

    print(f"Processing: {pdf.name}")

    # Retrieve credentials from environment variables
    api_key = os.getenv("UNSTRUCT_API_KEY")
    api_url = os.getenv("UNSTRUCT_API_URL")

    if not api_key or not api_url:
        raise ValueError(
            "Missing UNSTRUCTURED_API_KEY or UNSTRUCTURED_API_URL environment variables."
        )

    try:
        loader = UnstructuredLoader(
            str(pdf),
            api_key=api_key,
            url=api_url,
            chunking_strategy="by_title",
            max_characters=max_characters,
            overlap=overlap,
            include_orig_elements=False,
            strategy="hi_res",
        )

        chunks: List[Document] = loader.load()

        # Enrich metadata
        for chunk in chunks:
            chunk.metadata["source_file"] = pdf.name
            chunk.metadata["file_type"] = "pdf"

        print(f"  → Loaded & chunked into {len(chunks)} chunk(s)")
        return chunks

    except Exception as e:
        print(f"  ✗ Failed to process '{pdf.name}': {e}")
        return []
