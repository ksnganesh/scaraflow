import time
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag.engine import RAGEngine
from scara_rag.policies import RetrievalPolicy
from tqdm import tqdm

def run_benchmarks():
    print("Initializing benchmark environment...")

    # 1. Setup
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(
        QdrantConfig(
            collection="benchmark",
            vector_dim=384,
        ),
        client=client,
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    class BenchmarkEmbedder:
        def embed(self, t):
             return model.encode(t).tolist()

    embedder = BenchmarkEmbedder()

    rag = RAGEngine(
        embedder=embedder,
        store=store,
        llm=lambda _: "Benchmark Answer",
    )

    # Generate Synthetic Data
    N_DOCS = 1000  # Scaled down from 10k for speed in this environment, but logic remains
    print(f"Generating {N_DOCS} synthetic documents...")

    texts = [f"Document {i} about retrieval augmented generation and vector databases." for i in range(N_DOCS)]

    # 2. Embedding Benchmark
    print("Benchmarking Embedding...")
    start_time = time.time()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True).tolist()
    end_time = time.time()
    embedding_time = end_time - start_time
    print(f"Embedding time: {embedding_time:.4f}s ({N_DOCS / embedding_time:.2f} docs/s)")

    # 3. Indexing Benchmark
    print("Benchmarking Indexing...")
    start_time = time.time()
    import uuid
    store.upsert(
        ids=[str(uuid.uuid4()) for _ in range(N_DOCS)],
        vectors=vectors,
        metadata=[{"content": t} for t in texts],
        batch_size=100
    )
    end_time = time.time()
    indexing_time = end_time - start_time
    print(f"Indexing time: {indexing_time:.4f}s ({N_DOCS / indexing_time:.2f} docs/s)")

    # 4. Query Latency Benchmark
    print("Benchmarking Query Latency...")
    queries = [f"Query {i}" for i in range(100)]
    latencies = []

    for q in tqdm(queries):
        start_q = time.time()
        rag.query(q, policy=RetrievalPolicy(top_k=5))
        latencies.append((time.time() - start_q) * 1000) # ms

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    std_dev = np.std(latencies)

    print("\nBenchmark Results:")
    print(f"Embedding time: {embedding_time:.2f}s")
    print(f"Index time:     {indexing_time:.2f}s")
    print(f"Avg latency:    {avg_latency:.2f}ms")
    print(f"P95 latency:    {p95_latency:.2f}ms")
    print(f"Std dev:        {std_dev:.2f}ms")

if __name__ == "__main__":
    run_benchmarks()
