"""Hybrid fusion, metadata filters and filename/question detection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.langchain_compat import Document
from src.retrieval.retriever import RAGRetriever, build_where, reciprocal_rank_fusion
from src.ingestion.document_loader import (
    _detect_financial_metadata, detect_companies, detect_fiscal_years,
)


def d(cid):
    return Document(page_content=cid, metadata={"chunk_id": cid})


def test_rrf_prefers_docs_in_both_lists():
    fused = reciprocal_rank_fusion([[d("a"), d("b"), d("c")], [d("c"), d("a")]])
    assert [x.metadata["chunk_id"] for x in fused][:2] == ["a", "c"]
    assert len(fused) == 3


def test_build_where_single_company_only():
    assert build_where(["TCS"]) == {"company": "TCS"}
    assert build_where(["TCS", "Infosys"]) is None
    assert build_where([]) is None


def test_detect_companies_whole_token():
    assert detect_companies("TCS vs Infosys revenue") == ["TCS", "Infosys"]
    assert detect_companies("what happened in april") == []  # 'ril' inside 'april'


def test_detect_fiscal_years_normalised():
    assert "FY2025" in detect_fiscal_years("revenue in FY25")
    assert "FY2024-25" in detect_fiscal_years("results for 2024-25")


def test_filename_metadata():
    m = _detect_financial_metadata("SEBI_Circular_Mutual_Funds_FY2024.pdf")
    assert m["doc_type"] == "regulatory_filing"      # not 'annual_report' via 'ar_' in circul-ar_
    m = _detect_financial_metadata("TCS_ANNUAL_REPORT_FY2024-25.pdf")
    assert (m["company"], m["doc_type"], m["fiscal_year"]) == ("TCS", "annual_report", "FY2024-25")


class FakeVSM:
    def count(self): return 3


def test_matches_filter():
    r = RAGRetriever(FakeVSM())
    assert r._matches({"company": "TCS"}, {"company": "TCS"})
    assert not r._matches({"company": "Infosys"}, {"company": "TCS"})
    assert r._matches({"company": "TCS"}, None)
