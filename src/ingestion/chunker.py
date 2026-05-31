"""
src/ingestion/chunker.py

Splits documents into overlapping chunks while preserving and enriching
metadata so every chunk carries enough context for citation generation.
"""

from typing import List
from loguru import logger
from src.langchain_compat import Document

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from configs.settings import settings


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split a list of Documents into smaller, overlapping chunks.

    Each output chunk gets additional metadata:
        chunk_id    – "{doc_id}-{index}" for stable referencing
        chunk_index – 0-based position within the parent document
        chunk_total – total chunks produced from that document

    Args:
        documents:     Raw documents (from document_loader).
        chunk_size:    Override from settings.
        chunk_overlap: Override from settings.

    Returns:
        List of chunked Documents, each citation-ready.
    """
    size    = chunk_size    or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        # Prefer splitting on paragraphs → sentences → words → chars
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,   # injects 'start_index' into metadata
    )

    # Group by doc_id so we can compute chunk_total per document
    from collections import defaultdict
    raw_chunks_by_doc: dict[str, list[Document]] = defaultdict(list)

    for doc in documents:
        chunks = splitter.split_documents([doc])
        doc_id = doc.metadata.get("doc_id", "unknown")
        raw_chunks_by_doc[doc_id].extend(chunks)

    enriched: List[Document] = []
    for doc_id, chunks in raw_chunks_by_doc.items():
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"]    = f"{doc_id}-{idx:04d}"
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_total"] = total
            enriched.append(chunk)

    logger.info(
        f"Chunking complete: {len(documents)} page(s) → "
        f"{len(enriched)} chunk(s) "
        f"(size={size}, overlap={overlap})"
    )
    return enriched
