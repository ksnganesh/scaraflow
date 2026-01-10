<p align="center">
  <img src="assets/logo.png" alt="Scaraflow Logo" width="420"/>
</p>

<h2 align="center">Scaraflow</h2>

<p align="center">
  Retrieval-first, deterministic RAG infrastructure for production systems
</p>

---

## What is Scaraflow?

**Scaraflow** is a **retrieval-first RAG infrastructure** designed for **deterministic, low-variance, production-grade Retrieval-Augmented Generation**.

Scaraflow is **not**:
- an agent framework
- a prompt playground
- a chain-orchestration SDK

Scaraflow focuses on one problem only:

> **Correct, explicit, and scalable retrieval for LLM systems**

---

## Why Scaraflow Exists

Most modern RAG frameworks optimize for:
- orchestration flexibility
- feature breadth
- rapid prototyping

Scaraflow optimizes for:
- **retrieval correctness**
- **predictable latency**
- **streaming readiness**
- **infrastructure consistency**

Scaraflow treats retrieval as **infrastructure**, not glue code.

---

## Design Principles

- **Retrieval before generation**
- **Explicit contracts over hidden magic**
- **Deterministic behavior**
- **Low-variance latency**
- **Streaming-ready by design**
- **Same semantics in notebooks, services, and production**

---

## Architecture Overview

```
scaraflow/
├── scara-core        # strict contracts & invariants
├── scara-index       # vector store backends (Qdrant)
├── scara-rag         # deterministic RAG engine
├── scara-live        # streaming / temporal RAG (planned)
├── scara-graph       # graph-based RAG (planned)
└── scara-llm         # thin LLM adapters (planned)
```

---

## Installation

```bash
pip install scaraflow
```

**Dependencies**
- `qdrant-client`
- `sentence-transformers`
- standard scientific Python stack

---

## Quick Start (30 Seconds)

### In-Memory Setup (No Docker)

```python
import uuid
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag.engine import RAGEngine
from scara_rag.policies import RetrievalPolicy

# In-process Qdrant
client = QdrantClient(":memory:")
store = QdrantVectorStore(
    QdrantConfig(collection="demo", vector_dim=384),
    client=client
)

model = SentenceTransformer("all-MiniLM-L6-v2")

class Embedder:
    def embed(self, text):
        return model.encode(text).tolist()

rag = RAGEngine(
    embedder=Embedder(),
    store=store,
    llm=lambda _: "Demo answer",
)

documents = [
    "Scaraflow is retrieval-first.",
    "It prioritizes deterministic behavior.",
    "Qdrant is the reference backend.",
]

ids = [str(uuid.uuid4()) for _ in documents]
vectors = model.encode(documents).tolist()

store.upsert(
    ids=ids,
    vectors=vectors,
    metadata=[{"text": d} for d in documents],
)

response = rag.query(
    "What does Scaraflow prioritize?",
    policy=RetrievalPolicy(top_k=2),
)

print(response.answer)
```

---

## Production Setup (Docker / Cloud)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

```python
store = QdrantVectorStore(
    QdrantConfig(
        url="http://localhost:6333",
        collection="prod_v1",
        vector_dim=384,
    )
)
```

---

## Benchmarks

```
Documents        : 10000
Queries          : 100
Embedding Time   : 6.47s
Indexing Time    : 0.34s
Avg Latency      : 7.92 ms
P95 Latency      : 11.03 ms
Latency Std Dev  : 1.24 ms
```

Benchmarks can be run using:

```bash
python testing/benchmarks.py
```

---

## License

MIT License

---

## Author

Built and maintained by **Ganesh (K. S. N. Ganesh)**.
