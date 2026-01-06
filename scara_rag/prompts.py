from typing import Sequence
from .types import RAGContextBlock


def default_prompt(
    context: Sequence[RAGContextBlock],
    question: str,
) -> str:
    ctx = "\n".join(
        f"[{b.doc_id} | score={b.score:.4f}]" for b in context
    )

    return f"""You are a retrieval-grounded assistant.

Rules:
- Use ONLY the context provided
- If information is missing, say "I don't know"

Context:
{ctx}

Question:
{question}

Answer:
"""
