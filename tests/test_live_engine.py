from datetime import timedelta

from scara_core.types import QueryResult
from scara_live.engine import LiveRAGEngine
from scara_live.policies import LiveRetrievalPolicy


class DummyEmbedder:
    def __init__(self, vec=None):
        self.vec = vec or [0.1, 0.2, 0.3]

    def embed(self, text: str):
        return list(self.vec)


class DummyStore:
    def __init__(self, results):
        self.results = list(results)
        self.last_k = None

    def search(self, query, k, filters=None):
        self.last_k = k
        return list(self.results)


def _results():
    return [
        QueryResult(doc_id="1", score=0.2, payload={"text": "abc", "ts": 0}),
        QueryResult(doc_id="2", score=0.2, payload={"text": "def", "ts": 0}),
        QueryResult(doc_id="3", score=0.2, payload={"text": "ghi", "ts": 0}),
    ]


def test_live_engine_min_score_override_allows_context():
    store = DummyStore(_results())
    engine = LiveRAGEngine(embedder=DummyEmbedder(), store=store, llm=lambda _: "ok")

    policy = LiveRetrievalPolicy(window=timedelta(seconds=60), top_k=2, min_score=0.8)
    response = engine.query("q", policy=policy, min_score=0.0)

    assert response.answer == "ok"
    assert len(response.context) > 0


def test_live_engine_max_context_blocks_respected():
    store = DummyStore(_results())
    engine = LiveRAGEngine(embedder=DummyEmbedder(), store=store, llm=lambda _: "ok")

    policy = LiveRetrievalPolicy(
        window=timedelta(seconds=60),
        top_k=3,
        max_context_blocks=1,
    )
    response = engine.query("q", policy=policy)

    assert len(response.context) == 1


def test_live_engine_max_context_chars_override():
    store = DummyStore(_results())
    engine = LiveRAGEngine(embedder=DummyEmbedder(), store=store, llm=lambda _: "ok")

    policy = LiveRetrievalPolicy(
        window=timedelta(seconds=60),
        top_k=3,
        max_context_chars=3,
    )
    response = engine.query("q", policy=policy, max_chars=2)

    assert response.answer == "I don't know."


def test_live_engine_top_k_override_updates_store_call():
    store = DummyStore(_results())
    engine = LiveRAGEngine(embedder=DummyEmbedder(), store=store, llm=lambda _: "ok")

    policy = LiveRetrievalPolicy(window=timedelta(seconds=60), top_k=3)
    engine.query("q", policy=policy, top_k=1)

    assert store.last_k == 1
