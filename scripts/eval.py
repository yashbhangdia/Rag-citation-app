"""
scripts/eval.py — retrieval + citation-verification eval.

Reports, per question in eval/questions.jsonl:
  hit@k    : does a top-k reranked chunk come from `file` on one of `expect_page`
             and contain `expect_text`?
With --generate it also runs the LLM and reports the share of cited statements
whose figures verify against the cited chunk.

    python scripts/eval.py            # retrieval only (no LLM needed)
    python scripts/eval.py --generate # full pipeline
Add your own questions to eval/questions.jsonl (page numbers are 1-indexed PDF pages).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.rag_pipeline import RAGPipeline
from src.retrieval import RAGRetriever, VectorStoreManager, get_reranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval/questions.jsonl")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--generate", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.file).read_text().splitlines() if l.strip()]
    vsm = VectorStoreManager()
    pipe = RAGPipeline(vector_store=vsm) if args.generate else None
    retriever, reranker = RAGRetriever(vsm), get_reranker()

    hits = verified = cited = 0
    for r in rows:
        cands = retriever.retrieve(r["q"], k=20, strategy="hybrid")
        top = reranker.rerank(r["q"], cands, top_n=args.k)
        hit = any(
            d.metadata.get("file_name") == r["file"]
            and d.metadata.get("page") in r["expect_page"]
            and r["expect_text"].lower() in d.page_content.lower()
            for d in top
        )
        hits += hit
        line = f"{'HIT ' if hit else 'MISS'} {r['q']}"
        if pipe:
            res = pipe.query(r["q"], top_k=args.k)
            used = res.used_citations()
            cited += len(used)
            verified += sum(1 for c in used if c.verified)
            line += f"  | cited={len(used)} verified={sum(1 for c in used if c.verified)}"
        print(line)

    print(f"\nhit@{args.k}: {hits}/{len(rows)} = {hits / max(len(rows), 1):.0%}")
    if pipe:
        print(f"verified citations: {verified}/{cited} = {verified / max(cited, 1):.0%}")


if __name__ == "__main__":
    main()
