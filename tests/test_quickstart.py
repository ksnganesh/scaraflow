import pytest
from sentence_transformers import SentenceTransformer
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag.engine import RAGEngine
from scara_rag.policies import RetrievalPolicy
from qdrant_client import QdrantClient

class MockEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text):
        return self.model.encode(text).tolist()

@pytest.fixture
def embedder():
    return MockEmbedder()

@pytest.fixture
def store():
    # Use local in-memory Qdrant for testing
    client = QdrantClient(":memory:")
    return QdrantVectorStore(
        QdrantConfig(
            collection="test_collection",
            vector_dim=384,
        ),
        client=client,
    )

def test_rag_quickstart_flow(embedder, store):
    rag = RAGEngine(
        embedder=embedder,
        store=store,
        llm=lambda prompt: "Demo answer",
    )

    texts = [
        "Scaraflow is a retrieval-first RAG system.",
        "Qdrant provides Rust-based HNSW indexing.",
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts).tolist()

    store.upsert(
        ids=[0, 1],
        vectors=vectors,
        metadata=[{"src": "test", "content": t} for t in texts],
    )

    response = rag.query(
        "What is Scaraflow?",
        policy=RetrievalPolicy(top_k=1),
    )

    assert len(response.context) == 1
    assert "Scaraflow" in response.context[0].content
    assert response.answer == "Demo answer"
