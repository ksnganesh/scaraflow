from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalPolicy:
    top_k: int = 8
    min_score: float = 0.15

    # context control
    max_context_blocks: int = 6
    max_context_chars: int = 4000

    # behavior flags
    require_context: bool = True
    allow_empty_answer: bool = False