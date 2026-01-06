import time
from embedder import embed
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag import RAGEngine

store = QdrantVectorStore(
    QdrantConfig(collection="bench_stream", vector_dim=384)
)

rag = RAGEngine(
    embedder=type("E", (), {"embed": embed}),
    store=store,
    llm=lambda _: "ok",
)

# initial data
store.upsert(
    ids=["0"],
    vectors=[embed("Initial system state")],
    metadata=[{"ts": time.time()}],
)

print("Initial query:", rag.query("What is the system state?").answer)

# stream new data
store.upsert(
    ids=["1"],
    vectors=[embed("Breaking update happened just now")],
    metadata=[{"ts": time.time()}],
)

print("Post-stream query:", rag.query("What just happened?").answer)
