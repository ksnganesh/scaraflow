from typing import Callable, Any
from datetime import timedelta

from scara_core.protocols import Embedder, LLM
from scara_rag.assembler import assemble_context
from scara_rag.prompts import default_prompt
from scara_rag.types import RAGResponse
from scara_rag.errors import EmptyContextError

from .retriever import LiveRetriever
from .policies import LiveRetrievalPolicy
from .config import LiveConfig


class LiveRAGEngine:
    """
    Real-time, time-aware RAG engine.
    Deterministic, synchronous, and production-safe.
    """

    __slots__ = ("retriever", "llm", "prompt_fn", "config")

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: Any,
        llm: LLM,
        config: LiveConfig | None = None,
        prompt_fn: Callable = default_prompt,
    ):
        self.config = config or LiveConfig()

        self.retriever = LiveRetriever(
            embedder=embedder,
            store=store,
            timestamp_field=self.config.timestamp_field,
        )
        self.llm = llm
        self.prompt_fn = prompt_fn

    def query(
        self,
        question: str,
        *,
        window: timedelta | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
        max_chars: int | None = None,
    ) -> RAGResponse:
        if not question or not question.strip():
            return self._empty_response("Please provide a valid question.")

        effective_window = window or self.config.default_window
        effective_max_chars = max_chars or self.config.max_context_chars

        policy = LiveRetrievalPolicy(
            window=effective_window,
            top_k=top_k,
        )

        results = self.retriever.retrieve(question, policy)

        try:
            context = assemble_context(
                results=results,
                min_score=min_score,
                max_blocks=top_k,
                max_chars=effective_max_chars,
            )
        except EmptyContextError:
            return self._empty_response(
                "I don't know.",
                raw_results=results,
                window=effective_window,
            )

        prompt = self.prompt_fn(context, question)
        answer = self.llm(prompt)

        return RAGResponse(
            answer=answer,
            context=context,
            raw_results=results,
            prompt=prompt,
            metadata={
                "mode": "live",
                "window_seconds": effective_window.total_seconds(),
            },
        )

    def _empty_response(
        self,
        answer: str,
        *,
        raw_results=None,
        window: timedelta | None = None,
    ) -> RAGResponse:
        return RAGResponse(
            answer=answer,
            context=[],
            raw_results=raw_results or [],
            prompt="",
            metadata={
                "mode": "live",
                "window_seconds": window.total_seconds() if window else None,
                "status": "no_context_or_invalid_query",
            },
        )