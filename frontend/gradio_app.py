"""
frontend/gradio_app.py — Gradio chat UI (used by the Colab notebook).

    from frontend.gradio_app import build_demo
    build_demo(pipeline).queue().launch(share=True)
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from src.utils.render import citation_card_html, visible_text, warnings_for

CSS = """
.citation-card{border:1px solid #8884;border-left:4px solid #58a6ff;border-radius:8px;
  padding:10px 14px;margin:6px 0;font-size:.85rem}
.ci-index{font-family:monospace;font-weight:600;color:#58a6ff}
.ci-file{font-weight:600}.ci-page{opacity:.7;font-size:.8rem}
.ci-snippet{opacity:.8;font-style:italic;margin-top:4px}
"""


def sources_html(result) -> str:
    cites = result.used_citations()
    parts = [citation_card_html(c) for c in cites] or ["<i>No citations in this answer.</i>"]
    parts += [f"<div class='ci-page'>⚠️ {html.escape(w)}</div>" for w in warnings_for(result)]
    return "".join(parts)


def _text(content) -> str:
    """Gradio 6 stores message content as a list of blocks; flatten to a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, (list, tuple)):
        return " ".join(_text(c) for c in content).strip()
    return str(content or "")


def build_demo(pipeline, title: str = "FinSight AI — cited answers from filings") -> gr.Blocks:
    with gr.Blocks(title=title, css=CSS) as demo:
        gr.Markdown(f"## {title}\nEvery figure is cited to a page and checked against the source text.")
        chat = gr.Chatbot(height=460)
        sources = gr.HTML(label="Sources")
        with gr.Row():
            box = gr.Textbox(placeholder="e.g. What was TCS's revenue in FY25?", scale=5, show_label=False)
            k = gr.Slider(1, 10, value=5, step=1, label="Passages (k)", scale=2)
            strat = gr.Dropdown(["hybrid", "mmr", "similarity"], value="hybrid", label="Retrieval", scale=2)
        clear = gr.Button("Clear")

        def respond(message, history, top_k, strategy):
            history = list(history or [])
            qa = [  # prior turns as {question, answer} for follow-up handling
                {"question": _text(history[i]["content"]), "answer": _text(history[i + 1]["content"])}
                for i in range(0, len(history) - 1, 2)
                if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant"
            ]
            history += [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
            stream, citations = pipeline.query_stream(
                message, top_k=int(top_k), strategy=strategy, history=qa
            )
            text = ""
            for tok in stream:
                text += tok
                history[-1]["content"] = visible_text(text) + " ▌"
                yield history, "", gr.update()
            result = pipeline.finish(visible_text(text), citations)
            history[-1]["content"] = result.answer
            yield history, "", sources_html(result)

        box.submit(respond, [box, chat, k, strat], [chat, box, sources])
        clear.click(lambda: ([], ""), None, [chat, sources])
    return demo


if __name__ == "__main__":
    from src.generation.rag_pipeline import RAGPipeline

    build_demo(RAGPipeline()).queue().launch()
