from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable

from pydantic import BaseModel

from ..datatypes import RetrievalResult, UserQuery

_NUMERIC_TOKEN_PATTERN = re.compile(r"[-+]?(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.\d+)?")

MGSM_WEAK_JUDGE_RUBRIC_DRAFT = """DRAFT: MGSM weak judge rubric (non-default)
Assess whether the proposed answer matches the numeric result required by the problem.
Return YES only if:
1) The final numeric value is correct.
2) The answer meaningfully addresses the question intent.
Return NO for arithmetic mistakes, unit confusion, or missing final numeric answer.
"""


class MGSMProblem(BaseModel):
    id: str | None = None
    language: str | None = None
    question: str
    answer: str | int | float
    metadata: Dict[str, Any] | None = None


def _normalize_numeric_text(raw: str) -> str | None:
    text = raw.strip()
    text = text.rstrip(".,;:!?")
    text = text.replace(" ", "").replace(",", "")
    if not text:
        return None

    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return None

    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return None

    normalized = format(decimal_value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0", "+0"}:
        normalized = "0"
    return normalized


def _extract_final_numeric(raw_output: str) -> str | None:
    text = raw_output.replace("\u00a0", " ")
    matches = list(_NUMERIC_TOKEN_PATTERN.finditer(text))
    if not matches:
        return None
    return matches[-1].group(0)


@dataclass(frozen=True)
class MGSMTask:
    name: str = "mgsm"
    weak_judge_rubric_draft: str = MGSM_WEAK_JUDGE_RUBRIC_DRAFT

    def load_problems(self, jsonl_path: Path) -> Iterable[Any]:
        problems: list[MGSMProblem] = []
        with jsonl_path.open("r", encoding="utf-8") as file:
            for idx, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                problem = MGSMProblem.model_validate(obj)
                if problem.id is None:
                    problem = problem.model_copy(update={"id": f"mgsm-{idx}"})
                problems.append(problem)
        return problems

    def to_user_query(self, problem: Any) -> UserQuery:
        metadata = dict(getattr(problem, "metadata", None) or {})
        metadata["task"] = self.name
        metadata["gold_answer"] = str(problem.answer)
        language = getattr(problem, "language", None)
        if language is not None:
            metadata["language"] = language

        return UserQuery(
            id=str(problem.id),
            question=str(problem.question),
            metadata=metadata,
        )

    def build_solver_prompt(
        self,
        mode: str,
        query: UserQuery,
        retrieval: RetrievalResult,
    ) -> tuple[str, str]:
        del mode
        system = (
            "Solve the math word problem. "
            "Provide the final numeric answer in the final line as: Answer: <number>."
        )
        user = (
            f"Problem: {query.question}\n"
            f"Retrieved templates/context:\n{retrieval.concatenated_context}\n"
        )
        return system, user

    def extract_candidate(self, raw_output: str, query: UserQuery) -> str:
        del query
        token = _extract_final_numeric(raw_output)
        if token is None:
            return ""
        normalized = _normalize_numeric_text(token)
        return normalized or ""

    def oracle_validate(
        self, candidate: str, query: UserQuery, problem: Any
    ) -> tuple[bool, str | None]:
        metadata = query.metadata or {}
        gold_raw = metadata.get("gold_answer", getattr(problem, "answer", ""))

        normalized_candidate = _normalize_numeric_text(candidate)
        normalized_gold = _normalize_numeric_text(str(gold_raw))
        if normalized_candidate is None or normalized_gold is None:
            return False, "numeric_parse_error"
        if normalized_candidate == normalized_gold:
            return True, None
        return False, "numeric_mismatch"

    def summarize_steps(
        self, raw_output: str, candidate: str, query: UserQuery
    ) -> Dict[str, Any]:
        return {
            "task": self.name,
            "query": query.question,
            "candidate": candidate,
            "summary": f"Extracted final numeric answer {candidate} from solution text.",
            "raw_excerpt": raw_output[-200:],
        }

    def distill_template_input(
        self, query: UserQuery, steps_summary: Dict[str, Any]
    ) -> str:
        summary_text = str(steps_summary.get("summary", "")).strip()
        candidate = str(steps_summary.get("candidate", "")).strip()
        return (
            f"Task: {self.name}\n"
            f"Problem: {query.question}\n"
            f"Solution Steps Summary: {summary_text}\n"
            f"Final Candidate: {candidate}"
        )
