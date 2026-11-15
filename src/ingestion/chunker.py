"""
src/ingestion/chunker.py

Financial-aware chunking — uses larger chunks for dense financial text
and preserves table/numerical content by avoiding mid-number splits.
"""

from typing import List
from loguru import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict

from configs.settings import settings


# Financial documents need larger chunks — tables and financial paragraphs
# lose meaning when split too small
FINANCIAL_CHUNK_SIZE    = 1200   # larger than default 800
FINANCIAL_CHUNK_OVERLAP = 200    # more overlap to preserve context across pages


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split documents into overlapping chunks.
    Automatically uses larger chunk sizes for financial document types.
    """
    # Financial filings (long, table-heavy) get larger chunks; plain notes use
    # the configured defaults. Decided per document type, not per batch.
    financial_types = {
        "annual_report", "quarterly_results", "investor_presentation",
        "regulatory_filing", "rbi_report", "market_report",
    }
    sample_doc_type = documents[0].metadata.get("doc_type", "") if documents else ""
    is_financial = sample_doc_type in financial_types

    if chunk_size is None:
        chunk_size = FINANCIAL_CHUNK_SIZE if is_financial else settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = FINANCIAL_CHUNK_OVERLAP if is_financial else settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # For financial docs: prefer splitting on section breaks, not mid-sentence
        separators=[
            "\n\n\n",   # section breaks
            "\n\n",     # paragraph breaks
            "\n",       # line breaks
            ". ",       # sentence ends
            " ",        # word boundaries
            "",         # last resort: character
        ],
        add_start_index=True,
    )

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
        f"Chunking complete: {len(documents)} page(s) → {len(enriched)} chunk(s) "
        f"(size={chunk_size}, overlap={chunk_overlap}, "
        f"financial_mode={is_financial})"
    )
    return enriched