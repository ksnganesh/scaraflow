from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence, Optional

Vector = Sequence[float]


@dataclass(frozen=True)
class Document:
    """
    Canonical unit of knowledge.
    Everything in Scaraflow is reducible to a Document.
    """
    id: str
    content: str
    metadata: dict[str, Any]
    timestamp: datetime | None = None


@dataclass(frozen=True)
class QueryResult:
    """
    Result returned by retrieval systems.
    """
    doc_id: str
    score: float
    payload: Optional[dict[str, Any]] = None
