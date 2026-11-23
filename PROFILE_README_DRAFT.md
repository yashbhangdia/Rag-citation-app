<!--
Draft for your GitHub profile README.
To use it: create a public repo named exactly `yashbhangdia/yashbhangdia`,
paste this file's content (without this comment) into its README.md,
then edit the bio lines and add your other projects.
-->

# Hi, I'm Yash Bhangdia 👋

I build practical AI systems, with a focus on retrieval-augmented generation and trustworthy, citable answers.

## Featured project

### 📊 [FinSight AI](https://github.com/yashbhangdia/Rag-citation-app)
A RAG app that answers questions about Indian annual reports, quarterly results and SEBI circulars, where **every figure is cited to a page and checked against the source text**.

- Hybrid retrieval (dense + BM25 with reciprocal-rank fusion) and a cross-encoder reranker
- Real PDF page numbers for citations, plus verification that cited figures appear in the cited chunk
- Runs free on open models (Qwen2.5) on a Colab GPU; Streamlit and Gradio front ends; Docker and CI included
- Retrieval eval set with expected pages: `python scripts/eval.py`

**Stack:** Python · LangChain · ChromaDB · sentence-transformers · Streamlit · Gradio · pytest · GitHub Actions · Docker

## Other projects
<!-- Add 2–3 more repos here: name, one line on what it does, link. -->

## Skills
Python · RAG and LLM apps · Information retrieval · Evaluation · Docker · CI/CD

## Reach me
<!-- Add your LinkedIn / email / site. -->
