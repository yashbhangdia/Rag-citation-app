"""
src/utils/citation_builder.py

Builds structured citations from retrieved document chunks.
Handles parsing of [N] markers out of LLM responses.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from src.langchain_compat import Document


@dataclass
class Citation:
    index: int             # 1-based citation number used in text [1], [2] …
    chunk_id: str
    doc_id: str
    file_name: str
    page: int
    total_pages: int
    snippet: str           # best-supporting sentence (after verify) or first ~200 chars
    source_path: str
    text: str = ""         # full chunk text (used for verification)
    issues: List[str] = field(default_factory=list)  # verification problems
    verified: Optional[bool] = None                   # None = not checked yet

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
            "verified":    self.verified,
            "issues":      list(self.issues),
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
    uncited_figures: List[str] = field(default_factory=list)  # sentences with numbers but no [N]
    dropped_markers: List[int] = field(default_factory=list)  # [N] outside the context range

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
            text=doc.page_content,
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
    # Normalize citation markers: [ 1 ] -> [1]; [1, 2] / [1,2,3] -> [1][2][3]
    def _expand(m: "re.Match") -> str:
        return "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(0)))

    cleaned = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", _expand, answer)
    return CitationResult(
        answer=cleaned.strip(),
        citations=citations,
        raw_answer=answer,
    )



# ── Verification ─────────────────────────────────────────────────────────────

_MARKER = re.compile(r"\[(\d+)\]")
_NUMBER = re.compile(r"(?<![A-Za-z\d])\d[\d,]*(?:\.\d+)?")  # not glued to letters: FY25, Q4
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z₹\[(\d])|\n+")
_WORD = re.compile(r"[a-z0-9]+")


def _norm_number(tok: str) -> str:
    return tok.replace(",", "").rstrip(".")


def _figures(text: str) -> List[str]:
    """Numbers worth verifying: skip [N] markers, single digits and bare years."""
    text = _MARKER.sub(" ", text)
    out = []
    for m in _NUMBER.findall(text):
        n = _norm_number(m)
        if not n:
            continue
        if "." not in n and (len(n) == 1 or 1900 <= int(n) <= 2100):
            continue
        out.append(n)
    return out


def _chunk_numbers(text: str) -> set:
    return {_norm_number(m) for m in _NUMBER.findall(text)}


def _overlap_score(a: str, b: str) -> float:
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def best_supporting_sentence(
    claim: str,
    chunk_text: str,
    scorer: Optional[Callable[[str, str], float]] = None,
) -> Tuple[str, float]:
    """Sentence of the chunk that best supports ``claim`` (scorer or word overlap)."""
    sents = [s.replace("\n", " ").strip() for s in split_sentences(chunk_text)]
    sents = [s for s in sents if len(s) > 20] or [chunk_text[:300].strip()]
    score = scorer or _overlap_score
    ranked = sorted(((score(claim, s), s) for s in sents), reverse=True)
    top_score, top = ranked[0]
    return (top[:300] + ("…" if len(top) > 300 else "")), float(top_score)


def verify_citations(
    result: CitationResult,
    scorer: Optional[Callable[[str, str], float]] = None,
    min_score: Optional[float] = None,
) -> CitationResult:
    """
    Check the answer's citations against the retrieved chunks.

    * [N] outside 1..len(citations) is removed from the answer.
    * Every figure in a cited sentence must literally appear in a cited chunk.
    * With a ``scorer`` (e.g. cross-encoder) and ``min_score``, the cited chunk
      must also be topically relevant to the sentence.
    * Sentences containing figures but no citation are listed in ``uncited_figures``.
    * Each used citation gets ``verified`` + ``issues``, and its ``snippet`` becomes
      the best-supporting sentence from the chunk.
    """
    by_idx = {c.index: c for c in result.citations}
    valid = set(by_idx)

    # 1. drop out-of-range markers
    dropped: List[int] = []

    def _fix(m: "re.Match") -> str:
        n = int(m.group(1))
        if n in valid:
            return m.group(0)
        dropped.append(n)
        return ""

    answer = _MARKER.sub(_fix, result.answer)
    answer = re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", answer))
    result.answer = answer.strip()
    result.dropped_markers = dropped

    for c in result.citations:
        c.issues, c.verified = [], None
    used_flags: Dict[int, bool] = {}
    uncited: List[str] = []

    # 2. per-sentence checks
    for sent in split_sentences(result.answer):
        idxs = [int(i) for i in _MARKER.findall(sent)]
        claim = _MARKER.sub("", sent).strip()
        if not idxs:
            if _figures(claim):
                uncited.append(claim)
            continue
        cited = [by_idx[i] for i in idxs if i in by_idx]
        union_nums = set().union(*(_chunk_numbers(c.text) for c in cited)) if cited else set()
        missing = [f for f in _figures(claim) if f not in union_nums]

        for c in cited:
            issues: List[str] = []
            best, score = best_supporting_sentence(claim, c.text, scorer)
            if min_score is not None and score < min_score:
                issues.append("cited passage looks unrelated to this statement")
            if missing:
                issues.append("figure(s) not found in cited passages: " + ", ".join(missing))
            if best and c.verified is not False:
                c.snippet = best
            c.issues.extend(i for i in issues if i not in c.issues)
            used_flags[c.index] = used_flags.get(c.index, True) and not issues
            c.verified = used_flags[c.index]

    result.uncited_figures = uncited
    return result
