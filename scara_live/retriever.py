from datetime import datetime, timezone
from typing import List

from qdrant_client.models import Filter, FieldCondition, Range

from scara_core.protocols import Embedder
from scara_core.types import QueryResult
from scara_index.qdrant_store import QdrantVectorStore
from .policies import LiveRetrievalPolicy


class LiveRetriever:
    """
    Time-windowed vector retrieval with optional recency-aware re-ranking.

    This component is backend-agnostic at the API level and does not
    leak Qdrant-specific types beyond scara-index.
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
        time_filter = Filter(
            must=[
                FieldCondition(
                    key=self.timestamp_field,
                    range=Range(gte=start_ts),
                )
            ]
        )

        # 3. Embed query and search
        q_vec = self.embedder.embed(query)

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
            )

        return results

    def _apply_recency_ranking(
        self,
        *,
        results: List[QueryResult],
        now_ts: float,
        window_seconds: float,
    ) -> List[QueryResult]:
        """
        Combine semantic similarity with recency using a linear weighting model.

        Final score = (semantic_score * SEMANTIC_WEIGHT)
                    + (recency_score  * RECENCY_WEIGHT)

        Recency score is normalized to [0.0, 1.0] within the window.
        """

        SEMANTIC_WEIGHT = 0.7
        RECENCY_WEIGHT = 0.3

        def combined_score(r: QueryResult) -> float:
            payload = r.payload or {}
            doc_ts = payload.get(self.timestamp_field, 0.0)

            # Age in seconds
            age = max(0.0, now_ts - float(doc_ts))

            # Normalize recency within window
            recency_score = max(0.0, 1.0 - (age / window_seconds))

            return (r.score * SEMANTIC_WEIGHT) + (recency_score * RECENCY_WEIGHT)

        return sorted(results, key=combined_score, reverse=True)