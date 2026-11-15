"""UI-agnostic rendering helpers (HTML-escaped; shared by Streamlit and Gradio)."""

from __future__ import annotations

import html
import re
from typing import List

from .citation_builder import Citation, CitationResult


def visible_text(streamed: str) -> str:
    """Hide <think>…</think> blocks, including a still-open one mid-stream."""
    text = re.sub(r"<(think|thinking)>.*?</\1>", "", streamed, flags=re.DOTALL)
    text = re.sub(r"<(think|thinking)>.*\Z", "", text, flags=re.DOTALL)
    return text.strip()


def answer_html(answer: str) -> str:
    return html.escape(answer).replace("\n", "<br>")


def status_label(c: Citation) -> str:
    if c.verified is True:
        return "✅ verified"
    if c.verified is False:
        return "⚠️ check: " + "; ".join(c.issues)
    return ""


def citation_card_html(c: Citation) -> str:
    status = status_label(c)
    return (
        '<div class="citation-card">'
        f'<span class="ci-index">[{c.index}]</span> '
        f'<span class="ci-file">{html.escape(c.file_name)}</span> '
        f'<span class="ci-page">— Page {c.page}/{c.total_pages}</span>'
        f'<div class="ci-snippet">{html.escape(c.snippet)}</div>'
        + (f'<div class="ci-page">{html.escape(status)}</div>' if status else "")
        + "</div>"
    )


def warnings_for(result: CitationResult) -> List[str]:
    out = []
    if result.uncited_figures:
        out.append(
            "These statements contain figures without a citation: "
            + " | ".join(result.uncited_figures[:3])
        )
    if result.dropped_markers:
        out.append(f"Removed invalid citation marker(s): {sorted(set(result.dropped_markers))}")
    return out
