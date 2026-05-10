"""
scripts/ingest.py

CLI script to ingest documents into the vector store.

Usage:
    # Ingest a single file
    python scripts/ingest.py --file data/documents/report.pdf

    # Ingest an entire directory
    python scripts/ingest.py --dir data/documents/

    # Wipe and re-index
    python scripts/ingest.py --dir data/documents/ --reset

    # Show what's already indexed
    python scripts/ingest.py --list
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.ingestion import load_single_document, load_directory, chunk_documents
from src.retrieval.vector_store import VectorStoreManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Path to a single document.")
    group.add_argument("--dir",  type=str, help="Path to a directory of documents.")
    group.add_argument("--list", action="store_true",
                       help="List all indexed documents.")

    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe the entire vector store before ingesting."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Override chunk size (default from .env)."
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=None,
        help="Override chunk overlap (default from .env)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vsm = VectorStoreManager()

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list:
        docs = vsm.list_documents()
        if not docs:
            print("No documents indexed yet.")
            return
        print(f"\n{'='*55}")
        print(f"  Indexed documents ({len(docs)} total):")
        print(f"{'='*55}")
        for d in docs:
            print(f"  [{d['doc_id']}] {d['file_name']}  ({d['file_type']})")
        print(f"  Total vectors in store: {vsm.count()}")
        return

    # ── Optional reset ────────────────────────────────────────────────────────
    if args.reset:
        confirm = input("⚠  This will delete ALL indexed documents. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            vsm.reset()
            vsm = VectorStoreManager()   # re-init after wipe
            logger.info("Vector store reset.")
        else:
            logger.info("Reset cancelled.")

    # ── Load documents ────────────────────────────────────────────────────────
    if args.file:
        raw_docs = load_single_document(args.file)
    elif args.dir:
        raw_docs = load_directory(args.dir)
    else:
        print("Provide --file, --dir, or --list. Use --help for usage.")
        sys.exit(1)

    if not raw_docs:
        logger.warning("No documents loaded. Check the path and file types.")
        sys.exit(0)

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunks = chunk_documents(
        raw_docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # ── Index ─────────────────────────────────────────────────────────────────
    vsm.add_documents(chunks)

    print(f"\n✅  Done! Indexed {len(chunks)} chunks from {len(raw_docs)} pages.")
    print(f"   Total vectors in store: {vsm.count()}")


if __name__ == "__main__":
    main()
