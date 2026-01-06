from sentence_transformers import SentenceTransformer
from typing import List

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> list[float]:
    return _model.encode(text).tolist()

def embed_batch(
    texts: List[str],
    batch_size: int = 64,
) -> list[list[float]]:
    return _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
    ).tolist()
