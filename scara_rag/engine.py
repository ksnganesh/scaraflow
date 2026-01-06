from typing import Callable

from scara_core.protocols import Embedder, VectorStore, LLM
from scara_core.validators import validate_vector
from scara_core.types import QueryResult

from .policies import RetrievalPolicy
from .assembler import assemble_context
from .prompts import default_prompt
from .rerankers import IdentityReranker, Reranker
from .answerability import is_answerable
from .types import RAGResponse
from .errors import EmptyContextError
from .telemetry import build_metadata


class RAGEngine:
    """
    Industry-grade RAG engine.
    Explicit, auditable, extensible.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        *,
        reranker: Reranker = IdentityReranker(),
        prompt_fn: Callable = default_prompt,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.reranker = reranker
        self.prompt_fn = prompt_fn

    def query(
        self,
        question: str,
        *,
        policy: RetrievalPolicy = RetrievalPolicy(),
        filters: dict | None = None,
        telemetry: bool = True,
    ) -> RAGResponse:
        # 1. Embed
        q_vec = self.embedder.embed(question)
        validate_vector(q_vec)

        # 2. Retrieve
        raw_results = self.store.search(
            query=q_vec,
            k=policy.top_k,
            filters=filters,
        )

        # 3. Rerank
        ranked = self.reranker.rerank(question, raw_results)

        # 4. Assemble context
        try:
            context = assemble_context(
                results=ranked,
                min_score=policy.min_score,
                max_blocks=policy.max_context_blocks,
                max_chars=policy.max_context_chars,
            )
        except EmptyContextError:
            if policy.require_context:
                raise
            context = []

        # 5. Answerability gate
        if not is_answerable(context) and not policy.allow_empty_answer:
            return RAGResponse(
                answer="I don't know.",
                context=context,
                raw_results=raw_results,
                prompt="",
                metadata=build_metadata(),
            )

        # 6. Prompt
        prompt = self.prompt_fn(context, question)

        # 7. Generate
        try:
            answer = self.llm(prompt)
        except Exception as e:
            answer = f"[LLM FAILURE] {str(e)}"

        return RAGResponse(
            answer=answer,
            context=context,
            raw_results=raw_results,
            prompt=prompt,
            metadata=build_metadata(),
        )
