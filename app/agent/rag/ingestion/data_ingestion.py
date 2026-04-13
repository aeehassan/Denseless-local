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
from unstructured.partition.api import partition_via_api
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
    Parse a text-based PDF into categorized LangChain Documents using
    unstructured's partition_pdf() directly, without any chunking strategy.

    Partitioning Approach:
        Uses raw partitioning (no ``chunking_strategy``) so that each element
        retains its true semantic category (Title, NarrativeText, ListItem, Table).
        This is intentional — preserving categories enables downstream chains to
        differentiate between headings, body content, lists, and tables, which is
        essential for generating structured condensed notes.

        Elements exceeding ``max_characters`` are split manually with ``overlap``
        characters carried over between splits to maintain continuity.

    Section Tracking:
        The most recently seen Title element is tracked as ``current_section``.
        Every subsequent element inherits this value in its metadata, allowing
        quiz and notes chains to filter or group chunks by topic section.

    Metadata per Document:
        - section:      Title of the section this chunk belongs to.
                        Drives weak-topic filtering in quiz and notes chains.
        - category:     Unstructured element type (Title, NarrativeText,
                        ListItem, Table). Used by notes chains to determine
                        how to format each chunk.
        - page_number:  Source page in the original PDF.
        - element_id:   Unique element identifier from unstructured.
        - filename:     Basename of the source PDF file.

    Filtered Out:
        FigureCaption, Header, Footer, PageBreak, and any other element types
        not in {Title, NarrativeText, ListItem, Table} — these add noise to
        retrieval without contributing to learning content. Empty text elements
        are also discarded after stripping.

    Limitations:
        Designed for text-based PDFs only (LaTeX, Word exports).
        Scanned or image-based PDFs require strategy="hi_res" with OCR
        and are not supported here.

    Args:
        file_path:      Absolute or relative path to the PDF file.
        max_characters: Hard upper limit on characters per chunk (default 1000).
                        Elements exceeding this are split manually.
        overlap:        Character overlap between consecutive splits of a single
                        oversized element (default 200). Has no effect on
                        elements already within max_characters.

    Returns:
        A list of LangChain ``Document`` objects, each representing one element
        (or a split of one element) with its semantic category and section
        context preserved in metadata.

    Raises:
        FileNotFoundError: If ``file_path`` does not point to an existing file.
        ValueError:        If ``file_path`` does not have a .pdf extension.
        ValueError:        If UNSTRUCTURED_API_KEY or UNSTRUCTURED_API_URL
                           are not set in the environment.
    """

    pdf = Path(file_path)

    # 1. Validate inputs
    if not pdf.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf}")

    if pdf.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf.name}")

    api_key = os.getenv("UNSTRUCT_API_KEY")
    api_url = os.getenv("UNSTRUCT_API_URL")

    if not api_key or not api_url:
        raise ValueError(
            "Missing UNSTRUCTURED_API_KEY or UNSTRUCTURED_API_URL environment variables."
        )

    print(f"Processing: {pdf.name}")

    # 2. Partition into raw elements — no chunking_strategy so categories are preserved
    elements = partition_via_api(
        filename=str(pdf),
        strategy="fast",  # extract text only, no ocr
        api_key=api_key,
        api_url=api_url,
    )

    # 3 & 4. Filter and track current section
    KEEP = {"Title", "NarrativeText", "ListItem", "Table"}
    current_section = ""
    documents: List[Document] = []

    for element in elements:
        category = element.category

        if category not in KEEP:
            continue

        if category == "Title":
            current_section = element.text

        text = element.text.strip()
        if not text or len(text) < 30:  # discard very short elements
            continue

        # 5. Split element text if it exceeds max_characters, with overlap
        chunks = []
        if len(text) <= max_characters:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                end = start + max_characters
                chunks.append(text[start:end])
                start = end - overlap  # carry overlap into next chunk

        for chunk_text in chunks:
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "section": current_section,
                        "category": category,
                        "page_number": element.metadata.page_number,
                        "element_id": element.id,
                        "filename": pdf.name,
                    },
                )
            )

    print(f"  → Parsed into {len(documents)} document(s)")
    return documents
