from dataclasses import dataclass


@dataclass(frozen=True)
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection: str = "scara_vectors"
    vector_dim: int = 384
    distance: str = "Cosine"   # enforced
    timeout: float = 5.0          # important in prod
    recreate: bool = False        # for dev / tests