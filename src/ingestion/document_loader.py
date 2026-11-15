"""
src/ingestion/document_loader.py
Enhanced for financial documents — extracts company, FY, doc_type from filename.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import pymupdf4llm

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)

__all__ = [
    "load_single_document", "load_directory", "load_pdf_with_vision",
    "COMPANY_PATTERNS", "detect_companies", "detect_fiscal_years",
]

SUPPORTED_EXTENSIONS = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": Docx2txtLoader,
    ".md":   UnstructuredMarkdownLoader,
}

# Financial document type detection from filename keywords
DOC_TYPE_PATTERNS = {
    "regulatory_filing":   ["sebi", "bse", "nse", "filing", "circular"],
    "rbi_report":          ["rbi", "monetary-policy", "reserve-bank"],
    "annual_report":       ["annual", "ar", "integrated-report"],
    "quarterly_results":   ["q1", "q2", "q3", "q4", "quarterly", "results"],
    "investor_presentation": ["presentation", "investor-deck", "agm"],
    "market_report":       ["market", "outlook", "research", "report"],
}

# Known company name mapping from filename keywords
COMPANY_PATTERNS = {
    "TCS":          ["tcs", "tata-consultancy"],
    "Infosys":      ["infosys", "infy"],
    "HDFC Bank":    ["hdfc-bank", "hdfcbank", "hdfc_bank"],
    "Reliance":     ["reliance", "ril"],
    "Wipro":        ["wipro"],
    "ICICI Bank":   ["icici"],
    "Zomato":       ["zomato"],
    "SEBI":         ["sebi"],
    "RBI":          ["rbi", "reserve-bank"],
}


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _has_keyword(tokens: List[str], keyword: str) -> bool:
    """Whole-token (or contiguous token sequence) match — 'ar' must not hit 'circular'."""
    kw = _tokens(keyword)
    n = len(kw)
    return any(tokens[i:i + n] == kw for i in range(len(tokens) - n + 1))


def _detect_financial_metadata(filename: str) -> Dict[str, str]:
    """
    Infer company name, document type, and fiscal year from filename.
    e.g. 'TCS_Annual_Report_FY2024.pdf' →
         {company: 'TCS', doc_type: 'annual_report', fiscal_year: 'FY2024'}
    """
    name_lower = filename.lower()
    tokens = _tokens(filename)
    meta = {
        "company":     "Unknown",
        "doc_type":    "financial_document",
        "fiscal_year": "Unknown",
    }

    for company, keywords in COMPANY_PATTERNS.items():
        if any(_has_keyword(tokens, kw) for kw in keywords):
            meta["company"] = company
            break

    for doc_type, keywords in DOC_TYPE_PATTERNS.items():
        if any(_has_keyword(tokens, kw) for kw in keywords):
            meta["doc_type"] = doc_type
            break

    fy = detect_fiscal_years(name_lower)
    if fy:
        meta["fiscal_year"] = fy[0]

    return meta


def detect_companies(text: str) -> List[str]:
    """Companies mentioned in free text (e.g. a user question)."""
    tokens = _tokens(text)
    return [
        c for c, kws in COMPANY_PATTERNS.items()
        if any(_has_keyword(tokens, kw) for kw in kws)
    ]


def detect_fiscal_years(text: str) -> List[str]:
    """Fiscal years mentioned in text, normalised to the filename style (FY2024-25)."""
    out = []
    for m in re.finditer(r"(?:fy\s?)?(20\d{2})\s?[-/]\s?(\d{2,4})|fy\s?(20\d{2}|\d{2})", text.lower()):
        if m.group(1):
            end = m.group(2)[-2:]
            out.append(f"FY{m.group(1)}-{end}")
        else:
            y = m.group(3)
            out.append(f"FY20{y}" if len(y) == 2 else f"FY{y}")
    return list(dict.fromkeys(out))


def load_single_document(file_path: str | Path) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Route PDFs through enhanced loader
    if path.suffix.lower() == ".pdf":
        return load_pdf_with_vision(path)
    
    # All other formats use existing loaders
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'.")
    
    loader = SUPPORTED_EXTENSIONS[ext](str(path))
    docs = loader.load()

    doc_id = _file_hash(path)
    total_pages = len(docs)

    # Extract financial metadata from filename
    fin_meta = _detect_financial_metadata(path.name)

    enriched: List[Document] = []
    for i, doc in enumerate(docs):
        meta: Dict[str, Any] = {
            **doc.metadata,
            "source":      str(path),
            "file_name":   path.name,
            "file_type":   ext,
            "doc_id":      doc_id,
            "page":        doc.metadata.get("page", i) + 1,
            "total_pages": total_pages,
            # Financial-specific metadata
            "company":     fin_meta["company"],
            "doc_type":    fin_meta["doc_type"],
            "fiscal_year": fin_meta["fiscal_year"],
        }
        enriched.append(Document(page_content=doc.page_content, metadata=meta))

    logger.info(
        f"Loaded '{path.name}' → {len(enriched)} page(s) "
        f"[{fin_meta['company']} | {fin_meta['doc_type']} | {fin_meta['fiscal_year']}]"
    )
    return enriched


def load_directory(directory: str | Path, recursive: bool = True) -> List[Document]:
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

def load_pdf_with_vision(file_path: str | Path) -> List[Document]:
    """
    Load a PDF with pymupdf4llm (tables / multi-column aware).

    Uses ``page_chunks=True`` so every Document carries the *real* PDF page
    number (1-indexed) — citations depend on this being right.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    fin_meta = _detect_financial_metadata(path.name)
    doc_id = _file_hash(path)
    total_pages = len(page_chunks)

    docs: List[Document] = []
    for i, pc in enumerate(page_chunks):
        text = (pc.get("text") or "").strip()
        if not text:
            continue
        page_no = (pc.get("metadata") or {}).get("page_number", i + 1)
        docs.append(Document(
            page_content=text,
            metadata={
                "source":      str(path),
                "file_name":   path.name,
                "file_type":   ".pdf",
                "doc_id":      doc_id,
                "page":        int(page_no),
                "total_pages": total_pages,
                "company":     fin_meta["company"],
                "doc_type":    fin_meta["doc_type"],
                "fiscal_year": fin_meta["fiscal_year"],
            },
        ))

    logger.info(f"Loaded '{path.name}' via pymupdf4llm → {len(docs)}/{total_pages} pages")
    return docs
