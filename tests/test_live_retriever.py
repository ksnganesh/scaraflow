from datetime import timedelta

from scara_core.types import QueryResult
from scara_live.policies import LiveRetrievalPolicy
from scara_live.retriever import LiveRetriever


class DummyEmbedder:
    def __init__(self, vec=None):
        self.vec = vec or [0.1, 0.2, 0.3]

    def embed(self, text: str):
        return list(self.vec)


class DummyStore:
    def __init__(self, results):
        self.results = list(results)
        self.last_k = None
        self.last_filters = None

    def search(self, query, k, filters=None):
        self.last_k = k
        self.last_filters = filters
        return list(self.results)


def test_live_retriever_passes_filter_and_k():
    results = [
        QueryResult(doc_id="1", score=0.5, payload={"text": "a", "ts": 0}),
    ]
    store = DummyStore(results)
    retriever = LiveRetriever(embedder=DummyEmbedder(), store=store)

    policy = LiveRetrievalPolicy(window=timedelta(seconds=60), top_k=7)
    retriever.retrieve("hello", policy)

    assert store.last_k == 7
    assert store.last_filters is not None


def test_live_retriever_recency_ranking():
    results = [
        QueryResult(doc_id="old", score=0.9, payload={"text": "old", "ts": 0}),
        QueryResult(doc_id="new", score=0.1, payload={"text": "new", "ts": 1e15}),
    ]
    store = DummyStore(results)
    retriever = LiveRetriever(embedder=DummyEmbedder(), store=store)

    policy = LiveRetrievalPolicy(
        window=timedelta(seconds=60),
        top_k=2,
        recency_boost=True,
        recency_weights=(0.1, 0.9),
    )

    ordered = retriever.retrieve("hello", policy)
    assert ordered[0].doc_id == "new"


def test_live_retriever_no_recency_boost_keeps_order():
    results = [
        QueryResult(doc_id="first", score=0.9, payload={"text": "a", "ts": 0}),
        QueryResult(doc_id="second", score=0.1, payload={"text": "b", "ts": 1e15}),
    ]
    store = DummyStore(results)
    retriever = LiveRetriever(embedder=DummyEmbedder(), store=store)

    policy = LiveRetrievalPolicy(window=timedelta(seconds=60), top_k=2)
    ordered = retriever.retrieve("hello", policy)

    assert ordered[0].doc_id == "first"
