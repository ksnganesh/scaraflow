from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from scara_core.protocols import Embedder
from scara_index.qdrant_store import QdrantVectorStore


class LiveIndexer:
    """
    Append-only streaming indexer with support for single and batched operations.
    """

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

    def index(
        self,
        *,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Index a single document/event.
        """
        if not text or not text.strip():
            return
        if doc_id is not None and (not isinstance(doc_id, str) or not doc_id.strip()):
            raise ValueError("doc_id must be a non-empty string when provided")

        ts = timestamp or datetime.now(timezone.utc)

        payload = {
            **(metadata or {}),
            self.timestamp_field: ts.timestamp(),
            "text": text,
        }

        vector = self.embedder.embed(text)

        ids = [doc_id] if doc_id is not None else None
        self.store.upsert(
            ids=ids,
            vectors=[vector],
            metadata=[payload],
        )

    def index_batch(
        self,
        *,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        timestamps: Optional[List[datetime]] = None,
    ) -> None:
        """
        Optimized batch indexing to reduce network overhead.
        """
        if doc_ids is not None and len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        if doc_ids is not None and not all(isinstance(i, str) and i for i in doc_ids):
            raise ValueError("All doc_ids must be non-empty strings")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must be the same length as texts")
        if timestamps is not None and len(timestamps) != len(texts):
            raise ValueError("timestamps must be the same length as texts")

        metadatas = metadatas or [{} for _ in texts]
        if timestamps is None:
            ts_values = [datetime.now(timezone.utc).timestamp() for _ in texts]
        else:
            ts_values = [
                (ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)).timestamp()
                for ts in timestamps
            ]

        try:
            vectors = self.embedder.embed_batch(texts)
        except AttributeError:
            vectors = [self.embedder.embed(t) for t in texts]

        payloads = [
            {**meta, self.timestamp_field: ts_value, "text": text}
            for meta, text, ts_value in zip(metadatas, texts, ts_values)
        ]

        self.store.upsert(
            ids=doc_ids,
            vectors=vectors,
            metadata=payloads,
        )
