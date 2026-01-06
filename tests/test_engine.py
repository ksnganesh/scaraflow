import pytest
from scara_rag.engine import RAGEngine
from scara_rag.policies import RetrievalPolicy
from scara_core.types import QueryResult, Document
from scara_rag.types import RAGResponse
from scara_rag.errors import EmptyContextError

class MockStore:
    def __init__(self, results=None):
        self.results = results or []

    def search(self, query, k, filters=None):
        return self.results

class MockEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]

def test_rag_engine_empty_context_allow_empty():
    rag = RAGEngine(
        embedder=MockEmbedder(),
        store=MockStore([]),
        llm=lambda p: "Answer"
    )

    # By default require_context is True, so we must set it to False
    response = rag.query("question", policy=RetrievalPolicy(require_context=False))
    assert response.answer == "I don't know."
    assert len(response.context) == 0

def test_rag_engine_empty_context_require_error():
    rag = RAGEngine(
        embedder=MockEmbedder(),
        store=MockStore([]),
        llm=lambda p: "Answer"
    )

    with pytest.raises(EmptyContextError):
        rag.query("question", policy=RetrievalPolicy(require_context=True))

def test_rag_engine_llm_failure():
    rag = RAGEngine(
        embedder=MockEmbedder(),
        store=MockStore([QueryResult(doc_id="1", score=0.9, payload={"content": "text"})]),
        llm=lambda p: 1 / 0 # raises ZeroDivisionError
    )

    response = rag.query("question")
    assert "[LLM FAILURE]" in response.answer
