# 📚 RAG Citation Assistant

A production-ready **Retrieval-Augmented Generation** application that answers questions from your documents with precise inline citations. Built with HuggingFace (free), ChromaDB, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?style=flat-square&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Free_API-yellow?style=flat-square&logo=huggingface)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Local-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## ✨ Features

- 📄 **Multi-format ingestion** — PDF, TXT, DOCX, Markdown
- 🔍 **Semantic search** with MMR retrieval diversity
- 🤖 **Free LLM inference** via HuggingFace API (no credit card)
- 📎 **Inline citations** — every claim traced back to source + page
- 💬 **Streamlit chat UI** with streaming responses
- 🗄️ **Local ChromaDB** — persists between runs, no cloud needed
- 🚀 **One-command deploy** to Streamlit Cloud

---

## 🏗 Architecture

```
User Question
     │
     ▼
┌──────────────────────┐
│   Streamlit UI        │  frontend/app.py
└──────────┬───────────┘
           │
     ┌─────▼─────────────────────────────────────┐
     │              RAG Pipeline                  │  src/generation/rag_pipeline.py
     │                                            │
     │  1. Retrieve top-k chunks (MMR/cosine)    │
     │  2. Build numbered context block          │
     │  3. Format model-specific prompt          │
     │  4. Call HuggingFace Inference API        │
     │  5. Parse [N] citation markers            │
     └────┬──────────────────────────┬───────────┘
          │                          │
   ┌──────▼──────┐           ┌───────▼──────────┐
   │  ChromaDB   │           │ HuggingFace API  │
   │  (local)    │           │ zephyr-7b / phi3 │
   └─────────────┘           └──────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-citation-app.git
cd rag-citation-app

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your HuggingFace token
# Get a FREE token at: https://huggingface.co/settings/tokens
nano .env     # or open in your editor
```

Your `.env` should look like:
```env
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
LLM_MODEL=HuggingFaceH4/zephyr-7b-beta
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### 3. Ingest Documents

```bash
# Index the sample documents (included)
python scripts/ingest.py --dir data/documents/

# Or index your own files
python scripts/ingest.py --file /path/to/your/document.pdf

# Check what's indexed
python scripts/ingest.py --list
```

### 4. Run the App

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 and start asking questions!

---

## 📁 Project Structure

```
rag-citation-app/
├── frontend/
│   └── app.py                  # Streamlit UI
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py  # PDF, TXT, DOCX, MD loading
│   │   └── chunker.py          # Overlapping chunk strategy
│   ├── retrieval/
│   │   ├── vector_store.py     # ChromaDB manager
│   │   └── retriever.py        # MMR + similarity search
│   ├── generation/
│   │   ├── llm_client.py       # HuggingFace Inference API
│   │   ├── prompts.py          # Model-specific prompt templates
│   │   └── rag_pipeline.py     # End-to-end orchestration
│   └── utils/
│       └── citation_builder.py # Citation objects & formatting
├── configs/
│   └── settings.py             # Pydantic config with .env loading
├── scripts/
│   └── ingest.py               # CLI ingestion tool
├── tests/
│   ├── test_ingestion.py
│   └── test_citations.py
├── data/
│   └── documents/              # Put your documents here
├── .streamlit/
│   └── config.toml             # Streamlit theme
├── .env.example                # Environment template
├── requirements.txt
└── README.md
```

---

## 🤖 Free LLM Options

| Model | Quality | Speed | Needs Approval |
|-------|---------|-------|----------------|
| `HuggingFaceH4/zephyr-7b-beta` | ⭐⭐⭐⭐ | Medium | ❌ No |
| `microsoft/Phi-3-mini-4k-instruct` | ⭐⭐⭐ | Fast | ❌ No |
| `mistralai/Mistral-7B-Instruct-v0.3` | ⭐⭐⭐⭐⭐ | Medium | ✅ Yes |
| `meta-llama/Meta-Llama-3-8B-Instruct` | ⭐⭐⭐⭐⭐ | Medium | ✅ Yes |

Change the model in `.env`:
```env
LLM_MODEL=microsoft/Phi-3-mini-4k-instruct
```

---

## 🌐 Deploy for Free

### Option A: Streamlit Cloud (Easiest)

1. Push your repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path**: `frontend/app.py`
5. Add secrets under **Advanced settings**:
   ```toml
   HUGGINGFACE_API_TOKEN = "hf_xxxx"
   ```
6. Click **Deploy** — live in ~2 minutes!

> ⚠️ Note: Streamlit Cloud's free tier has 1GB RAM. Use a smaller embedding model (`all-MiniLM-L6-v2`) and pre-index documents locally, then commit the `data/chroma_db/` folder to the repo.

### Option B: Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** SDK
3. Push your code; add `HUGGINGFACE_API_TOKEN` in Space secrets

### Option C: Render.com

```bash
# render.yaml already handles everything
# Just connect your GitHub repo at render.com
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 🔧 Ingestion CLI Reference

```bash
# Single file
python scripts/ingest.py --file report.pdf

# Full directory (recursive)
python scripts/ingest.py --dir data/documents/

# Custom chunk size
python scripts/ingest.py --dir data/ --chunk-size 600 --chunk-overlap 100

# Reset and re-index everything
python scripts/ingest.py --dir data/ --reset

# List all indexed documents
python scripts/ingest.py --list
```

---

## 🗺 Roadmap (Production Enhancements)

- [ ] Re-ranking with cross-encoder models
- [ ] Hybrid search (BM25 + dense)
- [ ] Multi-turn conversation memory
- [ ] Document management UI (delete individual files)
- [ ] Export citations as BibTeX / APA
- [ ] REST API backend (FastAPI)
- [ ] Async ingestion pipeline
- [ ] Qdrant Cloud migration for scale

---

## 📄 License

MIT — free for personal and commercial use.
