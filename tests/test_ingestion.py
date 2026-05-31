"""
tests/test_ingestion.py
Basic smoke tests for document loading and chunking.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import os

from src.ingestion.document_loader import load_single_document
from src.ingestion.chunker import chunk_documents
from src.langchain_compat import Document


@pytest.fixture
def sample_txt_file():
    """Create a temporary text file for testing."""
    content = (
        "This is the first paragraph about artificial intelligence.\n\n"
        "Machine learning is a subset of AI that focuses on algorithms.\n\n"
        "Deep learning uses neural networks with many layers.\n\n"
        "Natural language processing helps computers understand human language.\n\n"
        "Computer vision allows machines to interpret visual information.\n"
    ) * 5  # repeat to get sufficient length
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


class TestDocumentLoader:
    def test_load_txt(self, sample_txt_file):
        docs = load_single_document(sample_txt_file)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_metadata_populated(self, sample_txt_file):
        docs = load_single_document(sample_txt_file)
        for doc in docs:
            assert "file_name" in doc.metadata
            assert "doc_id" in doc.metadata
            assert "file_type" in doc.metadata
            assert doc.metadata["file_type"] == ".txt"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_single_document("/nonexistent/path/file.txt")

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            with pytest.raises(ValueError):
                load_single_document(path)
        finally:
            os.unlink(path)


class TestChunker:
    def test_chunks_created(self, sample_txt_file):
        docs = load_single_document(sample_txt_file)
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=30)
        assert len(chunks) > 0

    def test_chunk_metadata(self, sample_txt_file):
        docs = load_single_document(sample_txt_file)
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=30)
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "chunk_total" in chunk.metadata

    def test_chunk_ids_unique(self, sample_txt_file):
        docs = load_single_document(sample_txt_file)
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=30)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
