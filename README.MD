# Scaraflow

**Scaraflow** is a **retrieval-first RAG infrastructure** designed for **deterministic, production-grade Retrieval-Augmented Generation**.

Scaraflow is not an agent framework, not a prompt playground, and not a demo SDK. It focuses on one thing and does it rigorously:

> **Correct, explicit, and scalable retrieval for LLM systems**

---

## Why Scaraflow

Most RAG frameworks prioritize orchestration and abstraction. Scaraflow prioritizes **retrieval correctness, predictability, and streaming readiness**.

## Design Principles

*   **Retrieval before generation**
*   **Explicit contracts over magic**
*   **Deterministic behavior**
*   **Low-variance latency**
*   **Streaming-ready by design**
*   **Infrastructure consistency across dev, notebooks, and production**

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

*Note: Scaraflow depends on `qdrant-client` and standard scientific stack libraries.*

---

## Quick Start (Run in 30 Seconds)

The fastest way to try Scaraflow is with the **In-Memory** setup. No Docker or external database required.

### 1. Create a script `demo.py`

```python
import uuid
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag.engine import RAGEngine
from scara_rag.policies import RetrievalPolicy

# 1. Setup Components
# Use in-memory Qdrant for instant setup
client = QdrantClient(":memory:")
store = QdrantVectorStore(
    QdrantConfig(collection="demo", vector_dim=384),
    client=client
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Wrap embedder to match protocol
class LocalEmbedder:
    def embed(self, text):
        return model.encode(text).tolist()

embedder = LocalEmbedder()

# 2. Initialize Engine
rag = RAGEngine(
    embedder=embedder,
    store=store,
    llm=lambda prompt: f"Generated answer based on: {len(prompt)} chars of context.",
)

# 3. Index Data
documents = [
    "Scaraflow is a retrieval-first RAG infrastructure.",
    "It prioritizes deterministic behavior and explicit contracts.",
    "Qdrant is the recommended vector backend for Scaraflow.",
]

vectors = model.encode(documents).tolist()
ids = [str(uuid.uuid4()) for _ in documents]

store.upsert(
    ids=ids,
    vectors=vectors,
    metadata=[{"content": doc} for doc in documents]
)

# 4. Query
response = rag.query(
    "What are the design principles of Scaraflow?",
    policy=RetrievalPolicy(top_k=2)
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

## Production Setup (Docker / Cloud)

For production, connect Scaraflow to a persistent Qdrant instance.

```bash
# Start Qdrant locally
docker run -p 6333:6333 qdrant/qdrant
```

```python
# Connect to local Docker or Qdrant Cloud
store = QdrantVectorStore(
    QdrantConfig(
        url="http://localhost:6333", # or your Cloud URL
        collection="prod_v1",
        vector_dim=384,
    )
)
```

---

## Benchmarks

Scaraflow includes a built-in benchmarking suite to verify infrastructure performance.

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
