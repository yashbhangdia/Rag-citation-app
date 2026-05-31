"""
src/generation/prompts.py

Prompt templates for the RAG citation pipeline.
Designed to work well with instruction-tuned open-source models.
"""

# ── System instruction injected at the top of every prompt ──────────────────
SYSTEM_PROMPT = """You are a precise research assistant. Answer questions using ONLY the provided context.

STRICT OUTPUT FORMAT — follow exactly:
1. Write a direct, concise answer in plain prose.
2. Cite sources inline using [N] only when that source directly supports the specific claim.
3. Only cite a source if you actually used information from it. Do NOT cite all sources by default.
4. After your answer, list only the sources you cited under a "References" header.
5. Do NOT include any reasoning, thinking steps, or internal monologue in your response.
6. Do NOT start with phrases like "Based on the context..." or "According to the provided passages..."
7. If the context does not contain the answer, say: "The provided documents don't contain enough information to answer this."

OUTPUT ONLY the final answer with inline citations. Nothing else."""


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
