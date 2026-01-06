from typing import Sequence
from .types import RAGContextBlock


def is_answerable(context: Sequence[RAGContextBlock]) -> bool:
    """
    Simple heuristic for now.
    Later: classifier / LLM check.
    """
    return len(context) > 0
