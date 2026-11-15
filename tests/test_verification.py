"""Citation verification: figures must exist in the cited chunk; bad markers are dropped."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.langchain_compat import Document
from src.utils.citation_builder import (
    build_citations, parse_citations_from_answer, verify_citations,
)
from src.utils.render import visible_text

CHUNK = "TCS reported revenue of ₹2,40,893 crore in FY25. Net profit was ₹48,797 crore."


def _result(answer):
    doc = Document(page_content=CHUNK, metadata={"file_name": "a.pdf", "page": 3, "total_pages": 9})
    return parse_citations_from_answer(answer, build_citations([doc]))


def test_correct_figure_verified():
    r = verify_citations(_result("Revenue was ₹2,40,893 crore [1]."))
    assert r.citations[0].verified is True
    assert r.citations[0].issues == []


def test_wrong_figure_flagged():
    r = verify_citations(_result("Profit was ₹50,000 crore [1]."))
    assert r.citations[0].verified is False
    assert "50000" in r.citations[0].issues[0]


def test_out_of_range_marker_dropped():
    r = verify_citations(_result("Revenue was ₹2,40,893 crore [1]. Something else [7]."))
    assert "[7]" not in r.answer
    assert r.dropped_markers == [7]


def test_uncited_figure_reported():
    r = verify_citations(_result("Revenue was ₹2,40,893 crore [1]. Margin was 24.5% last year."))
    assert any("24.5" in s for s in r.uncited_figures)


def test_years_and_small_ints_ignored():
    r = verify_citations(_result("In 2025 there were 3 segments [1]."))
    assert r.citations[0].verified is True


def test_scorer_threshold_flags_irrelevant_chunk():
    r = verify_citations(_result("Revenue was ₹2,40,893 crore [1]."), scorer=lambda c, t: -9.0, min_score=0.0)
    assert r.citations[0].verified is False


def test_snippet_is_supporting_sentence():
    r = verify_citations(_result("Net profit was ₹48,797 crore [1]."))
    assert "48,797" in r.citations[0].snippet


def test_visible_text_hides_think_blocks():
    assert visible_text("<think>hmm</think>Answer") == "Answer"
    assert visible_text("Answer <think>still thinking") == "Answer"


def test_fiscal_year_labels_are_not_figures():
    r = verify_citations(_result("Revenue was ₹2,40,893 crore in Q4 FY25 [1]."))
    assert r.citations[0].verified is True
