"""
src/generation/prompts.py

Prompt templates for the RAG citation pipeline.
Designed to work well with instruction-tuned open-source models.
"""

# ── System instruction injected at the top of every prompt ──────────────────
SYSTEM_PROMPT = """You are a precise, citation-aware research assistant.

RULES:
1. Answer ONLY using the provided context passages numbered [1], [2], [3], etc.
2. Cite EVERY factual claim with its source number, e.g.: "Paris is the capital of France [1]."
3. If a claim is supported by multiple sources, list all: "... [1][3]"
4. If the context does not contain enough information to answer, say:
   "I don't have enough information in the provided documents to answer this question."
5. Do NOT fabricate facts. Do NOT use prior knowledge beyond the context.
6. Be concise but thorough. Use bullet points when listing multiple items.
7. Always end with a one-sentence summary of the key finding.
"""


# ── Main RAG prompt template ─────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """{system}

---
CONTEXT PASSAGES:
{context}

---
QUESTION: {question}

ANSWER (use [N] citation markers for every fact):"""


def build_rag_prompt(context: str, question: str) -> str:
    """
    Construct the full RAG prompt.

    Args:
        context:  Pre-formatted numbered context block from citation_builder.
        question: User's query string.

    Returns:
        Complete prompt string ready for the LLM.
    """
    return RAG_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=question,
    )


# ── Zephyr / ChatML format wrapper ───────────────────────────────────────────
def wrap_for_zephyr(prompt: str) -> str:
    """
    Wrap prompt in Zephyr's <|system|>/<|user|>/<|assistant|> format.
    Required for HuggingFaceH4/zephyr-7b-beta to follow instructions.
    """
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n"
        f"CONTEXT PASSAGES:\n{{context}}\n\n"
        f"QUESTION: {{question}}</s>\n"
        f"<|assistant|>\n"
    )


def build_zephyr_prompt(context: str, question: str) -> str:
    """Builds a Zephyr-formatted RAG prompt."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n"
        f"CONTEXT PASSAGES:\n{context}\n\n"
        f"QUESTION: {question}</s>\n"
        f"<|assistant|>\n"
    )


def build_phi3_prompt(context: str, question: str) -> str:
    """Builds a Phi-3 formatted RAG prompt."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n"
        f"CONTEXT PASSAGES:\n{context}\n\n"
        f"QUESTION: {question}<|end|>\n"
        f"<|assistant|>\n"
    )


# Model-to-formatter mapping
PROMPT_FORMATTERS = {
    "HuggingFaceH4/zephyr-7b-beta":          build_zephyr_prompt,
    "microsoft/Phi-3-mini-4k-instruct":       build_phi3_prompt,
    "mistralai/Mistral-7B-Instruct-v0.3":     build_rag_prompt,
}


def build_prompt_for_model(
    model_name: str,
    context: str,
    question: str,
) -> str:
    """Auto-select the correct prompt format for a given model."""
    formatter = PROMPT_FORMATTERS.get(model_name, build_rag_prompt)
    return formatter(context, question)
