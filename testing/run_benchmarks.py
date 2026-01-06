import time
import statistics
from tqdm import tqdm
from uuid import uuid4
from bench_data import generate_docs
from embedder import embed, embed_batch


# ------------------------
# Scaraflow
# ------------------------

def bench_scaraflow(docs):
    from scara_index.qdrant_store import QdrantVectorStore
    from scara_index.config import QdrantConfig
    from scara_rag.engine import RAGEngine
    from scara_rag.policies import RetrievalPolicy

    print("\n[Scaraflow] Initializing store...")
    store = QdrantVectorStore(
        QdrantConfig(collection="bench_scara", vector_dim=384)
    )

    # -------- Embedding (batched) --------
    print("[Scaraflow] Embedding documents (batched)...")
    t0 = time.time()
    vectors = embed_batch(docs, batch_size=64)
    embed_time = time.time() - t0

    # -------- Indexing --------
    print("[Scaraflow] Indexing...")
    t0 = time.time()
    store.upsert(
        ids=[uuid4().hex for _ in docs],
        vectors=vectors,
        metadata=[{"src": "bench"} for _ in docs],
    )
    index_time = time.time() - t0

    rag = RAGEngine(
        embedder=type("E", (), {"embed": embed}),
        store=store,
        llm=lambda _: "ok",
    )

    policy = RetrievalPolicy(top_k=5)

    # -------- Query benchmark --------
    latencies = []
    for _ in tqdm(range(100), desc="[Scaraflow] Query"):
        t0 = time.time()
        rag.query(
            "What is retrieval augmented generation?",
            policy=policy,
        )
        latencies.append(time.time() - t0)

    return embed_time, index_time, latencies


# ------------------------
# Reporting
# ------------------------

def summarize(name, embed_time, index_time, latencies):
    print(f"\n{name} RESULTS")
    print("-" * 40)
    print(f"Embedding time: {embed_time:.2f}s")
    print(f"Index time:     {index_time:.2f}s")
    print(f"Avg latency:    {statistics.mean(latencies):.4f}s")
    print(f"P95 latency:    {statistics.quantiles(latencies, n=20)[18]:.4f}s")
    print(f"Std dev:        {statistics.stdev(latencies):.4f}s")
# ------------------------
# Runner
# ------------------------

def main():
    docs = generate_docs(10_000)

    embed_t, index_t, lat = bench_scaraflow(docs)
    summarize("Scaraflow", embed_t, index_t, lat)


if __name__ == "__main__":
    main()


