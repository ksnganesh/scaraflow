# Scaraflow

**Scaraflow** is a **retrieval-first RAG infrastructure** designed for **deterministic, production-grade Retrieval-Augmented Generation**.

Scaraflow is not an agent framework, not a prompt playground, and not a demo SDK.  
It focuses on one thing and does it rigorously:

> **Correct, explicit, and scalable retrieval for LLM systems**

---

## Why Scaraflow

Most RAG frameworks prioritize orchestration and abstraction.  
Scaraflow prioritizes **retrieval correctness, predictability, and streaming readiness**.

### Design principles
- Retrieval before generation
- Explicit contracts over magic
- Deterministic behavior
- Low-variance latency
- Streaming-ready by design
- Infrastructure consistency across dev, notebooks, and production

---

## Architecture Overview

```
scaraflow/
├── scara-core        # strict contracts & invariants
├── scara-index       # vector store backends (Qdrant)
├── scara-rag         # deterministic RAG engine
├── scara-live        # streaming / temporal RAG (planned)
├── scara-graph       # graph-based RAG (planned)
└── scara-llm         # thin LLM adapters
```

---

## Installation

```bash
pip install scaraflow
```

Scaraflow requires a **real vector database**.  
The recommended backend is **Qdrant**.

---

## Quick Start

Scaraflow supports **three official setups**.

---

### Option 1 — Docker (Local Qdrant)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

```python
from sentence_transformers import SentenceTransformer
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag.engine import RAGEngine
from scara_rag.policies import RetrievalPolicy

model = SentenceTransformer("all-MiniLM-L6-v2")
embedder = type("E", (), {"embed": lambda t: model.encode(t).tolist()})

store = QdrantVectorStore(
    QdrantConfig(
        url="http://localhost:6333",
        collection="quickstart",
        vector_dim=384,
    )
)

rag = RAGEngine(
    embedder=embedder,
    store=store,
    llm=lambda prompt: "Demo answer",
)

texts = [
    "Scaraflow is a retrieval-first RAG system.",
    "Qdrant provides Rust-based HNSW indexing.",
]

vectors = model.encode(texts).tolist()

store.upsert(
    ids=[0, 1],
    vectors=vectors,
    metadata=[{"src": "quickstart"} for _ in texts],
)

response = rag.query(
    "What is Scaraflow?",
    policy=RetrievalPolicy(top_k=2),
)

print(response.answer)
```

---

### Option 2 — No Docker (In-Process Qdrant)

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag.engine import RAGEngine

client = QdrantClient(path="./qdrant_data")

store = QdrantVectorStore(
    QdrantConfig(
        collection="local_demo",
        vector_dim=384,
    ),
    client=client,
)

model = SentenceTransformer("all-MiniLM-L6-v2")
embedder = type("E", (), {"embed": lambda t: model.encode(t).tolist()})

rag = RAGEngine(
    embedder=embedder,
    store=store,
    llm=lambda _: "Demo answer",
)

store.upsert(
    ids=[0],
    vectors=[model.encode("Scaraflow works without Docker").tolist()],
    metadata=[{"mode": "local"}],
)

print(rag.query("How does Scaraflow run locally?").answer)
```

---

### Option 3 — Qdrant Cloud / Remote Qdrant

```python
store = QdrantVectorStore(
    QdrantConfig(
        url="https://YOUR_QDRANT_ENDPOINT",
        collection="prod_collection",
        vector_dim=384,
    )
)
```

---

## Benchmarks

Scaraflow includes reproducible benchmarks measuring:

- embedding time
- indexing time
- query latency (avg / p95)
- variance

Example (CPU, 10k docs):

```
Embedding time: ~3.5s
Index time:     ~2.1s
Avg latency:    ~17ms
P95 latency:    ~20ms
Std dev:        low
```

---

## License

MIT License

---

## Author

Built and maintained by **Ganesh (K. S. N. Ganesh)**  
Focus: retrieval systems, streaming RAG, and infrastructure-grade AI tooling.
