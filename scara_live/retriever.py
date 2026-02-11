from datetime import datetime, timezone
from typing import List

from scara_core.protocols import Embedder
from scara_core.validators import validate_vector
from scara_core.types import QueryResult
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.filters import range_gte
from .policies import LiveRetrievalPolicy


class LiveRetriever:
    """
    Time-windowed vector retrieval with optional recency-aware re-ranking.

    This component is backend-agnostic at the API level and uses
    scara-index filter helpers for backend-specific implementations.
    """

    __slots__ = ("embedder", "store", "timestamp_field")

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: QdrantVectorStore,
        timestamp_field: str = "ts",
    ):
        self.embedder = embedder
        self.store = store
        self.timestamp_field = timestamp_field

    def retrieve(
        self,
        query: str,
        policy: LiveRetrievalPolicy,
    ) -> List[QueryResult]:
        # 1. Compute time window once
        now_ts = datetime.now(timezone.utc).timestamp()
        window_seconds = policy.window.total_seconds()
        start_ts = now_ts - window_seconds

        # 2. Build time filter
        time_filter = range_gte(self.timestamp_field, start_ts)

        # 3. Embed query and search
        q_vec = self.embedder.embed(query)
        validate_vector(q_vec)

        results: List[QueryResult] = self.store.search(
            query=q_vec,
            k=policy.top_k,
            filters=time_filter,
        )

        # 4. Optional recency-aware re-ranking
        if policy.recency_boost and results:
            results = self._apply_recency_ranking(
                results=results,
                now_ts=now_ts,
                window_seconds=window_seconds,
                weights=policy.recency_weights,
            )

        return results

    def _apply_recency_ranking(
        self,
        *,
        results: List[QueryResult],
        now_ts: float,
        window_seconds: float,
        weights: tuple[float, float] | None = None,
    ) -> List[QueryResult]:
        """
        Combine semantic similarity with recency using a linear weighting model.

        Final score = (semantic_score * SEMANTIC_WEIGHT)
                    + (recency_score  * RECENCY_WEIGHT)

        Recency score is normalized to [0.0, 1.0] within the window.
        """

        if weights is None:
            semantic_weight, recency_weight = 0.7, 0.3
        else:
            semantic_weight, recency_weight = weights

        total_weight = semantic_weight + recency_weight
        if total_weight <= 0:
            return results
        semantic_weight = semantic_weight / total_weight
        recency_weight = recency_weight / total_weight

        def combined_score(r: QueryResult) -> float:
            payload = r.payload or {}
            try:
                doc_ts = float(payload.get(self.timestamp_field, 0.0))
            except (TypeError, ValueError):
                doc_ts = 0.0

            # Age in seconds
            age = max(0.0, now_ts - float(doc_ts))

            # Normalize recency within window
            recency_score = max(0.0, 1.0 - (age / window_seconds))

            return (r.score * semantic_weight) + (recency_score * recency_weight)

        return sorted(results, key=combined_score, reverse=True)
