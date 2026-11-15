"""PDF page numbers must be the real PDF pages (citations depend on it)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pymupdf
import pytest

from src.ingestion.document_loader import load_single_document


@pytest.fixture
def pdf_path(tmp_path):
    doc = pymupdf.open()
    for i in range(1, 5):
        page = doc.new_page()
        # include a '-----' rule that used to break the page splitter
        page.insert_text((72, 72), f"Page marker {i}\n-----\nTCS revenue line {i}")
    p = tmp_path / "TCS_Annual_Report_FY2024-25.pdf"
    doc.save(p)
    return p


def test_real_page_numbers(pdf_path):
    docs = load_single_document(pdf_path)
    assert [d.metadata["page"] for d in docs] == [1, 2, 3, 4]
    assert all(d.metadata["total_pages"] == 4 for d in docs)
    for d in docs:
        assert f"Page marker {d.metadata['page']}" in d.page_content


def test_metadata_from_filename(pdf_path):
    d = load_single_document(pdf_path)[0]
    assert (d.metadata["company"], d.metadata["fiscal_year"]) == ("TCS", "FY2024-25")
