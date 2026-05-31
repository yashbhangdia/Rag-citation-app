"""
src/utils/citation_builder.py

Builds structured citations from retrieved document chunks.
Handles parsing of [N] markers out of LLM responses.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict

from src.langchain_compat import Document


@dataclass
class Citation:
    index: int             # 1-based citation number used in text [1], [2] …
    chunk_id: str
    doc_id: str
    file_name: str
    page: int
    total_pages: int
    snippet: str           # first ~200 chars of the chunk
    source_path: str

    def to_dict(self) -> Dict:
        return {
            "index":       self.index,
            "chunk_id":    self.chunk_id,
            "doc_id":      self.doc_id,
            "file_name":   self.file_name,
            "page":        self.page,
            "total_pages": self.total_pages,
            "snippet":     self.snippet,
            "source_path": self.source_path,
        }

    def format_inline(self) -> str:
        return f"[{self.index}] {self.file_name} — p.{self.page}"

    def format_reference(self) -> str:
        return (
            f"[{self.index}] **{self.file_name}**, "
            f"Page {self.page}/{self.total_pages}\n"
            f"   _{self.snippet}_"
        )


@dataclass
class CitationResult:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    raw_answer: str = ""          # LLM output before post-processing

    @property
    def cited_indices(self) -> List[int]:
        """Return citation numbers actually referenced in the answer."""
        return list(map(int, re.findall(r"\[(\d+)\]", self.answer)))

    def used_citations(self) -> List[Citation]:
        used = set(self.cited_indices)
        return [c for c in self.citations if c.index in used]

    def format_full_response(self) -> str:
        """Combine answer + formatted references section."""
        refs = "\n".join(c.format_reference() for c in self.used_citations())
        if refs:
            return f"{self.answer}\n\n---\n**References**\n{refs}"
        return self.answer


# ── Builder ──────────────────────────────────────────────────────────────────

def build_citations(chunks: List[Document]) -> List[Citation]:
    """
    Convert a list of retrieved Document chunks into Citation objects.
    Each chunk gets a stable 1-based index matching its position in the list.
    """
    citations: List[Citation] = []
    for i, doc in enumerate(chunks, start=1):
        meta = doc.metadata
        citations.append(Citation(
            index=i,
            chunk_id=meta.get("chunk_id", f"chunk-{i}"),
            doc_id=meta.get("doc_id", "unknown"),
            file_name=meta.get("file_name", "unknown"),
            page=meta.get("page", 1),
            total_pages=meta.get("total_pages", 1),
            snippet=doc.page_content[:200].replace("\n", " ").strip() + "…",
            source_path=meta.get("source", ""),
        ))
    return citations


def build_context_block(chunks: List[Document]) -> str:
    """
    Build a numbered context string to inject into the LLM prompt.

    Format:
        [1] (report.pdf, p.3)
        <text of chunk>

        [2] (manual.pdf, p.7)
        <text of chunk>
    """
    parts: List[str] = []
    for i, doc in enumerate(chunks, start=1):
        meta = doc.metadata
        header = f"[{i}] ({meta.get('file_name','?')}, p.{meta.get('page','?')})"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def parse_citations_from_answer(
    answer: str,
    citations: List[Citation],
) -> CitationResult:
    """
    Given raw LLM answer and citation list, produce a CitationResult.
    Cleans up any stray formatting artifacts.
    """
    # Normalize citation markers: [1], [1,2], [ 1 ] all become [1]
    cleaned = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", answer)
    return CitationResult(
        answer=cleaned.strip(),
        citations=citations,
        raw_answer=answer,
    )
