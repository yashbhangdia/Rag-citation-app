# 📊 FinSight AI — cited answers from Indian financial filings

![CI](https://github.com/yashbhangdia/rag-citation-app/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

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
* `transformers` (default) — Qwen2.5-3B-Instruct (~6 GB download), 4-bit, in-process on a CUDA GPU (Colab T4). Falls back to Qwen2.5-1.5B without CUDA. No key. Set `LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct` for better answers (~15 GB download).
* `hf_api` — HuggingFace Inference Providers with a free token (`HUGGINGFACEHUB_API_TOKEN`); subject to free-tier limits.

## Run on Colab (recommended)
1. Push this repo to GitHub.
2. Open `notebooks/finsight_colab.ipynb` in Colab, set **Runtime → T4 GPU**, run all cells. It clones the repo, installs `requirements-colab.txt`, keeps the index and PDFs on Colab's temporary disk (set `USE_DRIVE = True` to persist them on Google Drive instead), ingests, and opens a Gradio chat with a public link.
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

### Docker
```bash
cp .env.example .env   # set HUGGINGFACEHUB_API_TOKEN (the container has no GPU, so it uses LLM_BACKEND=hf_api)
docker compose up --build   # http://localhost:8501
```

## Design decisions
* **Hybrid retrieval (dense + BM25, fused with RRF).** Financial questions hinge on exact tokens (figures, company names) that embeddings blur; BM25 recovers them.
* **Cross-encoder rerank after retrieval.** Retrieve 20 candidates cheaply, then score each (question, chunk) pair and keep the top `k`. The same scorer later checks that a cited chunk is relevant to the sentence citing it.
* **Real PDF page numbers.** Pages come from `pymupdf4llm(page_chunks=True)` metadata, never from splitting text, so `[N] file, p.X` points at the true page.
* **Verify, don't trust.** Out-of-range `[N]` markers are dropped, grouped markers like `[1, 2]` are expanded, and every figure in a cited sentence must literally appear in a cited chunk.
* **Free by default.** Open models (Qwen2.5) on a Colab GPU; the HF Inference API is optional.

## Project layout
```
configs/    settings (pydantic-settings, .env)
src/ingestion/   loaders + chunker       src/retrieval/   vector store, hybrid retriever, reranker
src/generation/  prompts, LLM backends, RAG pipeline       src/utils/   citation building + verification
frontend/   Streamlit and Gradio apps    scripts/   ingest, download filings, eval
eval/       question set                 tests/     pytest suite (run in CI)
```

## Quality checks
```bash
pytest tests/                   # unit tests (verification, retrieval, page numbering, ingestion)
python scripts/eval.py          # retrieval hit@k on eval/questions.jsonl
python scripts/eval.py --generate   # + share of citations whose figures verify
```
**Current result:** hit@5 = **18/30 (60%)** on 30 questions over the six sample filings (7,143 chunks). The match is strict (right file, right page *and* the expected figure), so figures that legitimately appear on several pages count as misses (e.g. HDFC Bank's attrition table on p.153 vs. the p.52 summary). Real weak spot: headline numbers in long reports can rank below narrative pages (Reliance). Improving this is the next step.

`eval/questions.jsonl` is small — add questions (with the expected PDF page) for your own filings; it is only as good as that set.

## Known limits
* Verification checks that figures **appear in the cited chunk** and that it is topically relevant. It cannot prove the sentence is *semantically* entailed by the passage, so ⚠️ means "check", ✅ means "figures found", not "guaranteed correct".
* Scanned PDFs without a text layer are not OCR'd.
* Tables are extracted as markdown by pymupdf4llm; very complex multi-column tables can still lose structure.
* A 3B 4-bit model is weaker than frontier models at multi-step financial arithmetic — it is told to copy figures, not compute them.
* Fiscal-year filtering is left to the reranker (filename "FY2024-25" vs question "FY25" are different strings).
