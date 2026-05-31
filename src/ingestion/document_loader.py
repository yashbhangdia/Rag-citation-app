"""
src/ingestion/document_loader.py

Handles loading documents from disk (PDF, TXT, DOCX, MD).
Preserves rich metadata for citation generation.
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from src.langchain_compat import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)


SUPPORTED_EXTENSIONS = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": Docx2txtLoader,
    ".md":   UnstructuredMarkdownLoader,
}


def _file_hash(path: Path) -> str:
    """SHA-256 fingerprint of a file (first 16 hex chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_single_document(file_path: str | Path) -> List[Document]:
    """
    Load a single document and inject citation-ready metadata.

    Metadata schema per chunk:
        source        – original file path (str)
        file_name     – basename only
        file_type     – extension (.pdf, .txt, …)
        doc_id        – stable hash-based document ID
        page          – page number (PDF) or 1 for flat files
        total_pages   – total pages in document
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    loader_cls = SUPPORTED_EXTENSIONS[ext]
    loader = loader_cls(str(path))
    docs = loader.load()

    doc_id = _file_hash(path)
    total_pages = len(docs)

    enriched: List[Document] = []
    for i, doc in enumerate(docs):
        meta: Dict[str, Any] = {
            **doc.metadata,
            "source":      str(path),
            "file_name":   path.name,
            "file_type":   ext,
            "doc_id":      doc_id,
            "page":        doc.metadata.get("page", i) + 1,  # 1-indexed
            "total_pages": total_pages,
        }
        enriched.append(Document(page_content=doc.page_content, metadata=meta))

    logger.info(f"Loaded '{path.name}' → {len(enriched)} page(s)")
    return enriched


def load_directory(
    directory: str | Path,
    recursive: bool = True,
) -> List[Document]:
    """
    Walk a directory and load every supported document.

    Args:
        directory: Root folder to scan.
        recursive: If True, descend into sub-folders.

    Returns:
        Flat list of Documents with metadata populated.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    pattern = "**/*" if recursive else "*"
    all_docs: List[Document] = []
    skipped = 0

    for file_path in sorted(root.glob(pattern)):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue
        try:
            all_docs.extend(load_single_document(file_path))
        except Exception as exc:
            logger.warning(f"Skipping '{file_path.name}': {exc}")

    logger.info(
        f"Directory scan complete: {len(all_docs)} pages loaded, "
        f"{skipped} unsupported files skipped."
    )
    return all_docs
