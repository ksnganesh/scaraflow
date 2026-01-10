import time
import numpy as np
import warnings
from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import uuid
# Scaraflow imports
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag.engine import RAGEngine
from scara_rag.policies import RetrievalPolicy

warnings.filterwarnings("ignore")

# ----------------------------
# Configuration
# ----------------------------

NUM_DOCS = 10_000
NUM_QUERIES = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384

# ----------------------------
# Data generation
# ----------------------------

def generate_documents(n: int) -> List[str]:
    base_texts = [
        "Retrieval Augmented Generation is a technique.",
        "Vector databases like Qdrant are efficient.",
        "Scaraflow prioritizes deterministic retrieval.",
        "Production systems require predictable latency.",
        "Embeddings capture semantic meaning.",
        "Benchmarking infrastructure matters.",
        "HNSW enables fast approximate search.",
        "Python is effective for ML pipelines.",
        "Low variance latency improves reliability.",
        "Streaming data needs real-time indexing.",
    ]
    return [
        f"{base_texts[i % len(base_texts)]} (doc_id={i})"
        for i in range(n)
    ]


print(f"Generating {NUM_DOCS} documents...")
texts = generate_documents(NUM_DOCS)

# ----------------------------
# Embedding benchmark
# ----------------------------

print("\n[Embedding] Computing embeddings...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

t0 = time.time()
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
)
embedding_time = time.time() - t0

print(
    f"Embedding Time: {embedding_time:.2f}s "
    f"({NUM_DOCS / embedding_time:.1f} docs/sec)"
)

# ----------------------------
# Setup Scaraflow
# ----------------------------

print("\n[Scaraflow] Initializing vector store...")

client = QdrantClient(":memory:")

config = QdrantConfig(
    url=":memory:",
    collection="scaraflow_bench",
    vector_dim=VECTOR_DIM,
)

store = QdrantVectorStore(config, client=client)

# ----------------------------
# Indexing benchmark
# ----------------------------

print("[Scaraflow] Indexing documents...")

t0 = time.time()
store.upsert(
    ids=[uuid.uuid4().hex for _ in range(NUM_DOCS)],
    vectors=embeddings.tolist(),
    metadata=[{"text": t} for t in texts],
)
index_time = time.time() - t0

print(f"Indexing Time: {index_time:.2f}s")

# ----------------------------
# RAG Engine
# ----------------------------

embedder = type(
    "Embedder",
    (),
    {"embed": lambda _, text: model.encode(text).tolist()},
)()

rag = RAGEngine(
    embedder=embedder,
    store=store,
    llm=lambda _: "Mock Answer",
)

policy = RetrievalPolicy(top_k=5)

# ----------------------------
# Query generation
# ----------------------------

queries = [
    f"What is mentioned about {texts[i % NUM_DOCS].split()[0]}?"
    for i in range(NUM_QUERIES)
]

# ----------------------------
# Latency measurement
# ----------------------------

def measure_latency(fn, queries: List[str], label: str):
    latencies = []

    # Warmup
    for q in queries[:5]:
        fn(q)

    for q in queries:
        start = time.perf_counter()
        fn(q)
        latencies.append((time.perf_counter() - start) * 1000)

    avg = np.mean(latencies)
    p95 = np.percentile(latencies, 95)
    std = np.std(latencies)

    print(
        f"[{label}] Avg: {avg:.2f} ms | "
        f"P95: {p95:.2f} ms | "
        f"Std: {std:.2f} ms"
    )

    return avg, p95, std


print("\n[Scaraflow] Running query benchmark...")

def query_fn(q: str):
    rag.query(q, policy=policy)

metrics = measure_latency(query_fn, queries, "Scaraflow")

# ----------------------------
# Summary
# ----------------------------

print("\n" + "=" * 72)
print("Scaraflow Benchmark Summary")
print("=" * 72)
print(f"Documents        : {NUM_DOCS}")
print(f"Queries          : {NUM_QUERIES}")
print(f"Embedding Time   : {embedding_time:.2f}s")
print(f"Indexing Time    : {index_time:.2f}s")
print(f"Avg Latency      : {metrics[0]:.2f} ms")
print(f"P95 Latency      : {metrics[1]:.2f} ms")
print(f"Latency Std Dev  : {metrics[2]:.2f} ms")
print("=" * 72)