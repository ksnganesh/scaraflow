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
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Index a single document/event.
        """
        if not text or not text.strip():
            return

        ts = timestamp or datetime.now(timezone.utc)

        payload = {
            **(metadata or {}),
            self.timestamp_field: ts.timestamp(),
            "text": text,
        }

        vector = self.embedder.embed(text)

        self.store.upsert(
            ids=[doc_id],
            vectors=[vector],
            metadata=[payload],
        )

    def index_batch(
        self,
        *,
        doc_ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Optimized batch indexing to reduce network overhead.
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")

        ts_value = datetime.now(timezone.utc).timestamp()
        metadatas = metadatas or [{} for _ in texts]

        try:
            vectors = self.embedder.embed_batch(texts)
        except AttributeError:
            vectors = [self.embedder.embed(t) for t in texts]

        payloads = [
            {**meta, self.timestamp_field: ts_value, "text": text}
            for meta, text in zip(metadatas, texts)
        ]

        self.store.upsert(
            ids=doc_ids,
            vectors=vectors,
            metadata=payloads,
        )
