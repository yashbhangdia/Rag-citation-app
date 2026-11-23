"""
tests/test_citations.py
Unit tests for citation building and parsing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.langchain_compat import Document
from src.utils.citation_builder import (
    build_citations,
    build_context_block,
    parse_citations_from_answer,
    CitationResult,
    Citation,
)


def make_doc(content: str, page: int = 1, file_name: str = "test.pdf") -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id":    f"abc-{page:04d}",
            "doc_id":      "abc123",
            "file_name":   file_name,
            "page":        page,
            "total_pages": 10,
            "source":      f"/data/{file_name}",
        }
    )


class TestBuildCitations:
    def test_basic(self):
        docs = [make_doc("Content one", page=1), make_doc("Content two", page=2)]
        citations = build_citations(docs)
        assert len(citations) == 2
        assert citations[0].index == 1
        assert citations[1].index == 2

    def test_file_name_preserved(self):
        docs = [make_doc("Hello", file_name="report.pdf")]
        citations = build_citations(docs)
        assert citations[0].file_name == "report.pdf"

    def test_snippet_truncated(self):
        long_text = "X" * 500
        docs = [make_doc(long_text)]
        citations = build_citations(docs)
        assert len(citations[0].snippet) <= 210  # 200 + "…"


class TestBuildContextBlock:
    def test_numbered_format(self):
        docs = [make_doc("Paragraph one."), make_doc("Paragraph two.")]
        block = build_context_block(docs)
        assert "[1]" in block
        assert "[2]" in block
        assert "Paragraph one." in block

    def test_includes_filename(self):
        docs = [make_doc("Text", file_name="manual.pdf")]
        block = build_context_block(docs)
        assert "manual.pdf" in block


class TestParseCitations:
    def test_parses_markers(self):
        docs = [make_doc("Source one"), make_doc("Source two")]
        citations = build_citations(docs)
        answer = "The sky is blue [1]. Water is wet [2]."
        result = parse_citations_from_answer(answer, citations)
        assert isinstance(result, CitationResult)
        assert result.cited_indices == [1, 2]

    def test_expands_grouped_markers(self):
        citations = build_citations([make_doc("a"), make_doc("b"), make_doc("c")])
        result = parse_citations_from_answer("Revenue rose [1, 2] and [ 3 ].", citations)
        assert result.answer == "Revenue rose [1][2] and [3]."
        assert result.cited_indices == [1, 2, 3]

    def test_used_citations_filters(self):
        docs = [make_doc("S1"), make_doc("S2"), make_doc("S3")]
        citations = build_citations(docs)
        answer = "Only first source cited [1]."
        result = parse_citations_from_answer(answer, citations)
        used = result.used_citations()
        assert len(used) == 1
        assert used[0].index == 1

    def test_format_full_response(self):
        docs = [make_doc("Evidence here")]
        citations = build_citations(docs)
        answer = "This is a fact [1]."
        result = parse_citations_from_answer(answer, citations)
        full = result.format_full_response()
        assert "References" in full
        assert "test.pdf" in full
