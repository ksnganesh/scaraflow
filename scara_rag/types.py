from dataclasses import dataclass
from typing import Sequence, Any
from scara_core.types import QueryResult


@dataclass(frozen=True)
class RAGContextBlock:
    doc_id: str
    score: float
    content: str | None = None


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    context: Sequence[RAGContextBlock]
    raw_results: Sequence[QueryResult]
    prompt: str
    metadata: dict[str, Any]
