from __future__ import annotations

from typing import Any

from ..datatypes import LLMAnswer, UserQuery


def answer_with_retrieval(
    query: UserQuery,
    *,
    retrieval: Any = None,
) -> LLMAnswer:
    """Return a placeholder answer for retrieval-based solving."""
    metadata = None
    if retrieval is not None:
        metadata = {"retrieval": retrieval}
    return LLMAnswer(
        query_id=query.id, answer="RETRIEVAL_PLACEHOLDER", metadata=metadata
    )
