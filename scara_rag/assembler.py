from typing import Sequence
from scara_core.types import QueryResult
from .types import RAGContextBlock
from .errors import EmptyContextError


def assemble_context(
    results: Sequence[QueryResult],
    min_score: float,
    max_blocks: int,
    max_chars: int,
) -> list[RAGContextBlock]:
    blocks: list[RAGContextBlock] = []
    used_chars = 0

    for r in results:
        if r.score < min_score:
            continue

        content = f"Document ID: {r.doc_id}"
        block_size = len(content)

        if used_chars + block_size > max_chars:
            break

        blocks.append(
            RAGContextBlock(
                doc_id=r.doc_id,
                score=r.score,
                content=content,
            )
        )

        used_chars += block_size

        if len(blocks) >= max_blocks:
            break

    if not blocks:
        raise EmptyContextError("No context passed policy constraints")

    return blocks