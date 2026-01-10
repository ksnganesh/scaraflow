#🪲 ScaraFlow
---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![Numpy](https://img.shields.io/badge/numpy-%23013243.svg)]()
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
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

## Quick Start Guide

### 1. In-Memory Setup (No Docker)

Ideal for testing and prototyping without external infrastructure.

```python
import uuid
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag.engine import RAGEngine
from scara_rag.policies import RetrievalPolicy

# 1. Setup In-Process Qdrant
client = QdrantClient(":memory:")
store = QdrantVectorStore(
    QdrantConfig(collection="demo", vector_dim=384),
    client=client
)

# 2. Setup Embedder
model = SentenceTransformer("all-MiniLM-L6-v2")

class Embedder:
    def embed(self, text):
        return model.encode(text).tolist()

# 3. Initialize RAG Engine (with dummy LLM)
rag = RAGEngine(
    embedder=Embedder(),
    store=store,
    llm=lambda prompt: f"Simulated answer based on:\n{prompt}",
)

# 4. Ingest Documents
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

# 5. Query
response = rag.query(
    "What does Scaraflow prioritize?",
    policy=RetrievalPolicy(top_k=2),
)

print(response.answer)
```

### 2. Production Setup (With Docker)

Run Qdrant in a container for persistence and performance.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Connect Scaraflow to the local Qdrant instance:

```python
from qdrant_client import QdrantClient
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig

# Connect to Qdrant on localhost
store = QdrantVectorStore(
    QdrantConfig(
        url="http://localhost:6333",
        collection="prod_v1",
        vector_dim=384,
    )
)
# The rest of the setup (Embedder, RAGEngine) remains the same.
```

### 3. Cloud LLMs (OpenAI / Gemini)

Scaraflow is LLM-agnostic. You simply pass a callable that takes a string (prompt) and returns a string (answer).

#### Using OpenAI

```bash
pip install openai
```

```python
from openai import OpenAI
from scara_rag.engine import RAGEngine

client = OpenAI(api_key="sk-...")

def openai_adapter(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

rag = RAGEngine(
    embedder=Embedder(), # Defined in previous steps
    store=store,         # Defined in previous steps
    llm=openai_adapter,
)

response = rag.query("How does Scaraflow handle retrieval?")
print(response.answer)
```

#### Using Google Gemini

```bash
pip install google-generativeai
```

```python
import google.generativeai as genai
from scara_rag.engine import RAGEngine

genai.configure(api_key="AIza...")
model = genai.GenerativeModel('gemini-pro')

def gemini_adapter(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text

rag = RAGEngine(
    embedder=Embedder(),
    store=store,
    llm=gemini_adapter,
)

response = rag.query("Explain Scaraflow's design principles.")
print(response.answer)
```

### 4. Integration with FastAPI

Build a production API in seconds.

```bash
pip install fastapi uvicorn
```

```python
from fastapi import FastAPI
from pydantic import BaseModel
from scara_rag.policies import RetrievalPolicy

app = FastAPI()

# Assume 'rag' is initialized globally as shown in previous steps

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/rag/query")
def query_rag(request: QueryRequest):
    response = rag.query(
        request.question,
        policy=RetrievalPolicy(top_k=request.top_k)
    )
    return {
        "answer": response.answer,
        "context": [b.content for b in response.context],
        "metadata": response.metadata
    }

# Run with: uvicorn main:app --reload
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