from typing import Callable, Any, Optional
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
        config: Optional[LiveConfig] = None,
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
        policy: Optional[LiveRetrievalPolicy] = None,
        window: Optional[timedelta] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        max_chars: Optional[int] = None,
    ) -> RAGResponse:
        if not question or not question.strip():
            return self._empty_response("Please provide a valid question.")

        if policy is None:
            effective_window = window if window is not None else self.config.default_window
            policy = LiveRetrievalPolicy(
                window=effective_window,
                top_k=top_k if top_k is not None else 5,
                min_score=min_score if min_score is not None else 0.0,
                max_context_chars=max_chars,
            )
        else:
            effective_window = window if window is not None else policy.window
            policy = LiveRetrievalPolicy(
                window=effective_window,
                top_k=top_k if top_k is not None else policy.top_k,
                recency_boost=policy.recency_boost,
                min_score=min_score if min_score is not None else policy.min_score,
                max_context_blocks=policy.max_context_blocks,
                max_context_chars=max_chars if max_chars is not None else policy.max_context_chars,
                recency_weights=policy.recency_weights,
            )

        effective_max_chars = (
            policy.max_context_chars
            if policy.max_context_chars is not None
            else self.config.max_context_chars
        )
        max_blocks = policy.max_context_blocks or policy.top_k

        results = self.retriever.retrieve(question, policy)

        try:
            context = assemble_context(
                results=results,
                min_score=policy.min_score,
                max_blocks=max_blocks,
                max_chars=effective_max_chars,
            )
        except EmptyContextError:
            return self._empty_response(
                "I don't know.",
                raw_results=results,
                window=effective_window,
            )

        prompt = self.prompt_fn(context, question)

        try:
            answer = self.llm(prompt)
        except Exception as e:
            return RAGResponse(
                answer=f"[LLM FAILURE] {str(e)}",
                context=context,
                raw_results=results,
                prompt=prompt,
                metadata={
                    "mode": "live",
                    "window_seconds": effective_window.total_seconds(),
                    "error": str(e),
                },
            )

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
        window: Optional[timedelta] = None,
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
