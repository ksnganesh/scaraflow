from itertools import islice
from typing import Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
)

from scara_core.protocols import VectorStore
from scara_core.types import Vector, QueryResult
from scara_core.validators import validate_vector, validate_batch

from .config import QdrantConfig


def _batch(iterable: Iterable, size: int):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


class QdrantVectorStore(VectorStore):
    """
    Production-grade vector store backed by Qdrant.
    Uses native Rust HNSW internally.
    """

    def __init__(self, config: QdrantConfig, client: Optional[QdrantClient] = None):
        self.config = config
        if client:
            self.client = client
        else:
            self.client = QdrantClient(
                url=config.url,
                timeout=getattr(config, "timeout", 5.0),
            )
        self._ensure_collection()

    # ----------------------------
    # Collection management
    # ----------------------------

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        names = {c.name for c in collections}

        if self.config.collection not in names:
            self.client.create_collection(
                collection_name=self.config.collection,
                vectors_config=VectorParams(
                    size=self.config.vector_dim,
                    distance=Distance.COSINE,
                ),
            )
            return

        # Validate existing collection schema
        info = self.client.get_collection(self.config.collection)
        vector_params = info.config.params.vectors

        if vector_params.size != self.config.vector_dim:
            raise ValueError(
                f"Vector dimension mismatch: "
                f"collection={vector_params.size}, "
                f"config={self.config.vector_dim}"
            )

        if vector_params.distance != Distance.COSINE:
            raise ValueError(
                f"Distance mismatch: "
                f"collection={vector_params.distance}, "
                f"config=Cosine"
            )

    # ----------------------------
    # Write path (BATCHED)
    # ----------------------------

    def upsert(
        self,
        ids: list[str],
        vectors: list[Vector],
        metadata: list[dict],
        *,
        batch_size: int = 256,  # SAFE default
    ) -> None:
        if not (len(ids) == len(vectors) == len(metadata)):
            raise ValueError("ids, vectors, metadata length mismatch")

        validate_batch(vectors)

        for id_batch, vec_batch, meta_batch in zip(
            _batch(ids, batch_size),
            _batch(vectors, batch_size),
            _batch(metadata, batch_size),
        ):
            points = [
                PointStruct(id=i, vector=v, payload=m)
                for i, v, m in zip(id_batch, vec_batch, meta_batch)
            ]

            self.client.upsert(
                collection_name=self.config.collection,
                points=points,
            )

    # ----------------------------
    # Read path (HNSW)
    # ----------------------------

    def search(self,query: Vector,k: int,filters: Filter | None = None,) -> list[QueryResult]:
        validate_vector(query)
        if filters is not None and not isinstance(filters, Filter):
            raise TypeError("filters must be a qdrant_client.models.Filter")
        # ---- Qdrant API compatibility layer ----
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
            collection_name=self.config.collection,
            query=query,
            limit=k,
            query_filter=filters,)
            hits = result.points
        elif hasattr(self.client, "search"):
            hits = self.client.search(
            collection_name=self.config.collection,
            query_vector=query,
            limit=k,
            query_filter=filters,)
        elif hasattr(self.client, "search_points"):
            hits = self.client.search_points(
            collection_name=self.config.collection,
            vector=query,
            limit=k,
            filter=filters,)
        else:
            raise RuntimeError(
            "Unsupported qdrant-client version: "
            "no compatible query method found")
        return [
            QueryResult(
                doc_id=str(hit.id),
                score=float(hit.score),
                payload=hit.payload,
            ) for hit in hits
        ]
