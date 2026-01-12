from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class LiveRetrievalPolicy:
    """
    Controls time-aware retrieval.
    """
    window: timedelta
    top_k: int = 5
    recency_boost: bool = False