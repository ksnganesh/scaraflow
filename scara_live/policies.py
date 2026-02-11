from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Tuple


@dataclass(frozen=True)
class LiveRetrievalPolicy:
    """
    Controls time-aware retrieval.
    """
    window: timedelta
    top_k: int = 5
    recency_boost: bool = False
    min_score: float = 0.0
    max_context_blocks: Optional[int] = None
    max_context_chars: Optional[int] = None
    recency_weights: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        window_seconds = self.window.total_seconds()
        if window_seconds <= 0:
            raise ValueError("window must be a positive duration")
        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.min_score < 0:
            raise ValueError("min_score must be >= 0")
        if self.max_context_blocks is not None and self.max_context_blocks <= 0:
            raise ValueError("max_context_blocks must be positive")
        if self.max_context_chars is not None and self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        if self.recency_weights is not None:
            semantic_w, recency_w = self.recency_weights
            if semantic_w < 0 or recency_w < 0:
                raise ValueError("recency_weights values must be >= 0")
            total = semantic_w + recency_w
            if total == 0:
                raise ValueError("recency_weights must not sum to zero")
