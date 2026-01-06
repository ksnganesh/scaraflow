from typing import Protocol, Callable
from .types import Vector, Document, QueryResult


class Embedder(Protocol):
    """
    Converts text into vectors.
    Must be deterministic.
    """

    def embed(self, text: str) -> Vector: ...

    def embed_batch(self, texts: list[str]) -> list[Vector]: ...


class VectorStore(Protocol):
    """
    Abstract vector index.
    SQLite, FAISS, Milvus, Rust ANN — all fit here.
    """

    def upsert(
        self,
        ids: list[str],
        vectors: list[Vector],
        metadata: list[dict],
    ) -> None: ...

    def search(
        self,
        query: Vector,
        k: int,
        filters: dict | None = None,
    ) -> list[QueryResult]: ...


class StreamIndexer(Protocol):
    """
    Append-only, time-aware indexing.
    Required for LiveRAG.
    """

    def index(self, document: Document) -> None: ...

    def search_recent(
        self,
        query: Vector,
        window_seconds: int,
        k: int,
    ) -> list[Document]: ...


class LLM(Protocol):
    """
    Thin callable wrapper.
    Prompt in, text out.
    """

    def __call__(self, prompt: str) -> str: ...
