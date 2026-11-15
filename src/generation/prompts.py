"""
src/generation/prompts.py
Finance-domain prompt — optimised for due diligence on Indian filings.

Backends receive chat *messages*; the model's own chat template is applied once,
by the backend. No hand-written model tokens here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in Indian public markets.
You help with due diligence on company filings, financial statements, investor presentations, and regulatory documents.

RULES:
1. Answer using ONLY the numbered context passages [1], [2], ... provided by the user.
2. Cite EVERY figure, claim, or regulatory reference with its passage number, e.g. "Revenue was ₹2,40,893 crore [2]."
3. Copy numbers exactly as written in the passage (units, commas, decimals). Never round, convert or infer figures.
4. Only cite a passage that literally supports the sentence. Use only numbers that exist in the passages.
5. If the passages do not contain the answer, say exactly: "This information is not available in the provided documents."
6. For comparisons across companies, use one labelled bullet per company, each cited.
7. Output only the final answer — no reasoning steps. End with a one-line key takeaway when you gave an analysis."""


def build_user_message(context: str, question: str) -> str:
    return f"CONTEXT PASSAGES:\n{context}\n\nQUESTION: {question}\n\nAnswer (cite every fact with [N]):"


def build_messages(
    context: str,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    max_history_turns: int = 3,
) -> List[Dict[str, str]]:
    """
    system + (last few Q/A turns, answers only without context) + user(context, question).
    """
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-max_history_turns:]:
        msgs.append({"role": "user", "content": turn["question"]})
        msgs.append({"role": "assistant", "content": turn["answer"]})
    msgs.append({"role": "user", "content": build_user_message(context, question)})
    return msgs


REWRITE_SYSTEM = (
    "Rewrite the user's last question as a single standalone search query, using the "
    "chat history to resolve pronouns and omitted company names or years. "
    "Output only the rewritten question."
)


def build_rewrite_messages(
    question: str, history: List[Dict[str, str]], max_history_turns: int = 2
) -> List[Dict[str, str]]:
    hist = "\n".join(
        f"Q: {t['question']}\nA: {t['answer'][:300]}" for t in history[-max_history_turns:]
    )
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"Chat history:\n{hist}\n\nLast question: {question}"},
    ]
