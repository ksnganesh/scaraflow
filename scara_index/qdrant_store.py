from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    Batch,
)

from scara_core.protocols import VectorStore
from scara_core.types import Vector, QueryResult
from scara_core.validators import validate_vector, validate_batch
from scara_core.utils import generate_id

from .config import QdrantConfig


class QdrantVectorStore(VectorStore):
    """
    Production-grade vector store backed by Qdrant.
    Optimized for high-throughput batching and low-latency search.
    """

    def __init__(self, config: QdrantConfig, client: Optional[QdrantClient] = None):
        self.config = config
        self.client = client or QdrantClient(
            url=config.url,
            # Using a more robust timeout for production workloads
            timeout=getattr(config, "timeout", 10.0),
        )
        self._ensure_collection()

    # ----------------------------
    # Collection management
    # ----------------------------

    def _ensure_collection(self) -> None:
        """
        Validates existence and schema of the target collection.
        """
        if not self.client.collection_exists(self.config.collection):
            self.client.create_collection(
                collection_name=self.config.collection,
                vectors_config=VectorParams(
                    size=self.config.vector_dim,
                    distance=Distance.COSINE,
                ),
            )
            return

        # Validate existing collection schema to prevent downstream runtime errors
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
                f"Distance mismatch: collection={vector_params.distance}, expected=Cosine")

    # ----------------------------
    # Write path (BATCHED)
    # ----------------------------

    def upsert(
        self,
        *,
        vectors: List[Vector],
        metadata: List[dict],
        ids: Optional[List[str]] = None,
        batch_size: int = 256,
    ) -> List[str]:
        """
        Efficiently inserts or updates vectors using Qdrant's Batch API.
        """
        if not (len(vectors) == len(metadata)):
            raise ValueError("Input lists (vectors, metadata) must have identical lengths")
        if ids is None:
            ids = [generate_id() for _ in vectors]
        if not (len(ids) == len(vectors) == len(metadata)):
            raise ValueError("Input lists (ids, vectors, metadata) must have identical lengths")
        
        if not all(isinstance(i, str) and i for i in ids):
            raise ValueError("All ids must be non-empty strings")
        
        # Perform high-speed validation from scara_core
        validate_batch(vectors)

        # Optimization: Standard list slicing is faster than itertools.islice 
        # for data already held in memory.
        for i in range(0, len(ids), batch_size):
            self.client.upsert(
                collection_name=self.config.collection,
                points=Batch(
                    ids=ids[i : i + batch_size],
                    vectors=vectors[i : i + batch_size],
                    payloads=metadata[i : i + batch_size],
                ),
            )
        return ids

    # ----------------------------
    # Read path (HNSW)
    # ----------------------------

    def search(
        self, 
        query: Vector, 
        k: int, 
        filters: Optional[Filter] = None
    ) -> List[QueryResult]:
        """
        Executes a vector similarity search with optional filtering.
        """
        if k<=0:
            raise ValueError("k must be a positive integer")
        validate_vector(query)

        if filters is not None and not isinstance(filters, Filter):
            raise TypeError("filters must be an instance of qdrant_client.models.Filter")

        # Optimization: Direct call to search is the fastest path in the current Qdrant Python SDK.
        # We use the search method as the primary driver for HNSW performance.
        try:
            hits = self.client.search(
                collection_name=self.config.collection,
                query_vector=query,
                query_filter=filters,
                limit=k,
                # Ensure we get the payload if needed for QueryResult expansion later
                with_payload=True, 
            )
        except AttributeError:
            # Fallback for newer Qdrant 'query_points' API if 'search' is deprecated in future versions
            result = self.client.query_points(
                collection_name=self.config.collection,
                query=query,
                query_filter=filters,
                limit=k,
                with_payload=True,
            )
            hits = result.points

        return [
            QueryResult(
                doc_id=str(hit.id),
                score=float(hit.score),
                # Note: You can expand QueryResult to include hit.payload if needed
                payload=hit.payload,
            )
            for hit in hits
        ]
