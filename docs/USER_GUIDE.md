# Scaraflow User Guide

This guide walks you through Scaraflow end-to-end with practical, copy‑pasteable examples.

---

## What Scaraflow Is (And Isn’t)

Scaraflow is a retrieval-first RAG infrastructure focused on deterministic retrieval and predictable behavior.

Scaraflow is not:
- an agent framework
- a prompt playground
- a chain-orchestration SDK

---

## Quick Start (In‑Memory)

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag.engine import RAGEngine
from scara_rag.policies import RetrievalPolicy

client = QdrantClient(":memory:")
store = QdrantVectorStore(
    QdrantConfig(collection="demo", vector_dim=384),
    client=client,
)

model = SentenceTransformer("all-MiniLM-L6-v2")

class Embedder:
    def embed(self, text: str):
        return model.encode(text).tolist()

rag = RAGEngine(
    embedder=Embedder(),
    store=store,
    llm=lambda prompt: f"Simulated answer based on:\n{prompt}",
)

documents = [
    "Scaraflow is retrieval-first.",
    "It prioritizes deterministic behavior.",
    "Qdrant is the reference backend.",
]
vectors = model.encode(documents).tolist()

store.upsert(
    vectors=vectors,
    metadata=[{"text": d} for d in documents],
)

# Note: upsert returns the generated ids if you need them later:
# ids = store.upsert(vectors=vectors, metadata=[{"text": d} for d in documents])

response = rag.query(
    "What does Scaraflow prioritize?",
    policy=RetrievalPolicy(top_k=2),
)

print(response.answer)
```

---

## Production Setup (Local Qdrant)

Run Qdrant:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Connect Scaraflow:

```python
from qdrant_client import QdrantClient
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig

store = QdrantVectorStore(
    QdrantConfig(
        url="http://localhost:6333",
        collection="prod_v1",
        vector_dim=384,
    )
)
```

---

## RAG Engine (Deterministic Flow)

Scaraflow’s `RAGEngine.query()` follows a fixed flow:
- embed the question
- retrieve top‑k from the vector store
- rerank (optional)
- assemble context using policy constraints
- build prompt
- call LLM

The response includes:
- answer
- context blocks
- raw retrieval results
- prompt
- metadata

---

## Retrieval Policy

Use `RetrievalPolicy` to control retrieval behavior:

```python
from scara_rag.policies import RetrievalPolicy

policy = RetrievalPolicy(
    top_k=5,
    min_score=0.2,
    max_context_blocks=6,
    max_context_chars=4000,
    require_context=True,
    allow_empty_answer=False,
)
```

---

## LiveRAG (Real‑Time)

LiveRAG handles time‑windowed retrieval for streaming data.

### Indexing

```python
from scara_live.indexer import LiveIndexer

indexer = LiveIndexer(embedder=embedder, store=store)

indexer.index(
    doc_id="evt-1",
    text="Breaking market news...",
    metadata={"source": "feed"},
)
```

### Querying

```python
from datetime import timedelta
from scara_live.engine import LiveRAGEngine

live_rag = LiveRAGEngine(embedder=embedder, store=store, llm=llm)

response = live_rag.query(
    "What happened in the last hour?",
    window=timedelta(hours=1),
    top_k=5,
)
```

---

## LiveRetrievalPolicy (Advanced)

If you need fine‑grained control:

```python
from datetime import timedelta
from scara_live.policies import LiveRetrievalPolicy

policy = LiveRetrievalPolicy(
    window=timedelta(minutes=30),
    top_k=5,
    min_score=0.1,
    max_context_blocks=4,
    max_context_chars=2000,
    recency_boost=True,
    recency_weights=(0.7, 0.3),
)
```

Notes:
- `recency_weights=(semantic, recency)` are normalized internally.
- `min_score` filters low‑quality results before context assembly.
- `max_context_blocks` and `max_context_chars` bound context size.

---

## Common Gotchas

- If you get `"I don't know."`, check `min_score` and `require_context`.
- If context is empty, verify metadata contains `text` or `content`.
- Make sure vector dimensions match the embedder output.

---

## Benchmarks

Run:

```bash
python testing/benchmarks.py
```

The latest results are tracked in `scaraflow_bench.csv`.

---

## Troubleshooting

- Qdrant connection issues: confirm the service is running and reachable.
- Embedding errors: confirm `sentence-transformers` is installed.
- TLS warnings in the LiveRAG demo: the demo uses a fallback fetch path.
