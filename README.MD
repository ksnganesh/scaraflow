# Scaraflow

**Scaraflow** is a **retrieval-first RAG infrastructure** designed for **deterministic, production-grade Retrieval-Augmented Generation**.

It is not an agent framework, a prompt playground, or a demo SDK.  
It focuses on one thing and does it rigorously:

> **Correct, explicit, and scalable retrieval for LLM systems.**

---

## Why Scaraflow?

Most RAG frameworks prioritize orchestration and "magic" abstractions.  
Scaraflow prioritizes **correctness, predictability, and infrastructure quality**.

### Core Design Principles
1.  **Retrieval before Generation:** If the context is wrong, the answer is wrong. We treat retrieval as a strict database query, not a fuzzy search.
2.  **Explicit Contracts:** No hidden prompts, no "auto-magic" context injection. You control exactly what goes into the LLM.
3.  **Deterministic Behavior:** Given the same index and query, the result is predictable.
4.  **Production Ready:** Includes rigorous validation, telemetry, and low-variance latency.

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
from scaraflow.scara_index.qdrant_store import QdrantVectorStore
from scaraflow.scara_index.config import QdrantConfig
from scaraflow.scara_rag.engine import RAGEngine

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

## Architecture

Scaraflow is modular by design.

```
scaraflow/
├── scara-core        # Protocols, types, and strict validators
├── scara-index       # Vector store implementations (Qdrant)
├── scara-rag         # The RAG Engine (retrieval, ranking, assembly)
└── scara-llm         # Adapters for LLM providers
```

---

## Benchmarks

Scaraflow includes a built-in benchmarking suite to verify infrastructure performance.

**Latest Run (10,000 Documents, CPU):**

| Metric | Result |
| :--- | :--- |
| **Embedding Time** | ~24.5s (408 docs/s) |
| **Indexing Time** | ~7.5s (1326 docs/s) |
| **Avg Query Latency** | **80ms** |
| **P95 Latency** | **140ms** |

*Run benchmarks yourself:*
```bash
python benchmarks/run_benchmark.py
```

---

## Development & Testing

Scaraflow is tested against Python 3.9 through 3.12.

```bash
# Run test suite
pytest tests/

# Run validation checks
pytest tests/test_validation.py
```

---

## License

MIT License

---

## Author

Built and maintained by **Ganesh (K. S. N. Ganesh)**.