from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Protocol

from ..datatypes import RetrievalResult, UserQuery


class TaskSpec(Protocol):
    @property
    def name(self) -> str: ...

    def load_problems(self, jsonl_path: Path) -> Iterable[Any]: ...

    def to_user_query(self, problem: Any) -> UserQuery: ...

    def build_solver_prompt(
        self,
        mode: str,
        query: UserQuery,
        retrieval: RetrievalResult,
    ) -> tuple[str, str]: ...

    def extract_candidate(self, raw_output: str, query: UserQuery) -> str: ...

    def oracle_validate(
        self, candidate: str, query: UserQuery, problem: Any
    ) -> tuple[bool, str | None]: ...

    def summarize_steps(
        self, raw_output: str, candidate: str, query: UserQuery
    ) -> Dict[str, Any]: ...

    def distill_template_input(
        self, query: UserQuery, steps_summary: Dict[str, Any]
    ) -> str: ...
