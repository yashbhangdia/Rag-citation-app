# 📊 FinSight AI — cited answers from Indian financial filings

Ask questions about annual reports, quarterly results and SEBI/RBI circulars and get answers where **every figure is cited to a page and checked against the source text**. Free to run: open models on a Colab GPU, no paid API.

## How it works

```
question ──► (follow-up rewrite) ──► hybrid retrieval ──► cross-encoder rerank ──► LLM ──► citation verification
              uses chat history      dense + BM25 (RRF)     top-k chunks, [1]..[k]   Qwen2.5   ✅ / ⚠️ per citation
                                     + company filter
```

| Stage | What it does | Code |
|---|---|---|
| Ingestion | pymupdf4llm per-page extraction (real PDF page numbers), company / doc-type / FY from the filename, 1200-char financial chunks | `src/ingestion/` |
| Retrieval | Dense (bge-small) + BM25 fused with reciprocal-rank-fusion; a question naming one company is filtered to it, a comparison ("TCS vs Infosys") is retrieved per company | `src/retrieval/retriever.py` |
| Rerank | Cross-encoder keeps the best `k` chunks | `src/retrieval/reranker.py` |
| Generate | Pluggable free backends (below), model's chat template applied once | `src/generation/` |
| Verify | `[N]` outside the context is dropped; every figure in a cited sentence must literally appear in the cited chunk; relevance score checked; the citation snippet becomes the best-supporting sentence; figures with no citation are flagged | `src/utils/citation_builder.py` |

### LLM backends (`LLM_BACKEND`)
* `transformers` (default) — Qwen2.5-7B-Instruct, 4-bit, in-process on a CUDA GPU (Colab T4). Falls back to Qwen2.5-3B without CUDA. No key.
* `hf_api` — HuggingFace Inference Providers with a free token (`HUGGINGFACEHUB_API_TOKEN`); subject to free-tier limits.

## Run on Colab (recommended)
1. Push this repo to GitHub.
2. Open `notebooks/finsight_colab.ipynb` in Colab, set **Runtime → T4 GPU**, run all cells. It clones the repo, installs `requirements-colab.txt`, stores the index and PDFs on your Drive, ingests, and opens a Gradio chat with a public link.
3. Changed code locally? Push and re-run the first cell.

## Run locally
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # set LLM_BACKEND=hf_api and a HF token on a laptop without a GPU
python scripts/download_documents.py --all     # sample filings (PDFs are not committed)
python scripts/ingest.py --dir data/documents
streamlit run frontend/app.py   # or: python frontend/gradio_app.py
```

## Quality checks
```bash
pytest tests/                   # unit tests (verification, retrieval, page numbering, ingestion)
python scripts/eval.py          # retrieval hit@k on eval/questions.jsonl
python scripts/eval.py --generate   # + share of citations whose figures verify
```
`eval/questions.jsonl` is small — add questions (with the expected PDF page) for your own filings; it is only as good as that set.

## Known limits
* Verification checks that figures **appear in the cited chunk** and that it is topically relevant. It cannot prove the sentence is *semantically* entailed by the passage, so ⚠️ means "check", ✅ means "figures found", not "guaranteed correct".
* Scanned PDFs without a text layer are not OCR'd.
* Tables are extracted as markdown by pymupdf4llm; very complex multi-column tables can still lose structure.
* A 7B 4-bit model is weaker than frontier models at multi-step financial arithmetic — it is told to copy figures, not compute them.
* Fiscal-year filtering is left to the reranker (filename "FY2024-25" vs question "FY25" are different strings).
