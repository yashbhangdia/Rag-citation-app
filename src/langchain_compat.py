"""LangChain compatibility shim.

Some environments install the newer split packages like langchain-core,
so this helper falls back from `langchain.schema.Document` to
`langchain_core.documents.Document` when needed.
"""

try:
    from langchain.schema import Document
except ImportError:  # pragma: no cover
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise ImportError(
            "Could not import Document from langchain or langchain_core. "
            "Install langchain 1.3.x and langchain-core, or use a supported "
            "Python environment."
        ) from exc

__all__ = ["Document"]
