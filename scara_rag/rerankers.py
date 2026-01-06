from typing import Protocol, Sequence
from scara_core.types import QueryResult


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        results: Sequence[QueryResult],
    ) -> Sequence[QueryResult]: ...


class IdentityReranker:
    """Default: no reranking"""
    def rerank(self, query: str, results: Sequence[QueryResult]):
        return results