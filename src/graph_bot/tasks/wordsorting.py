from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from pydantic import BaseModel, Field

from ..datatypes import RetrievalResult, UserQuery

_PYTHON_CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*.*?```", re.DOTALL)
_ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


META_REASONER_SYSTEM = """[Meta Reasoner]
You are a Meta Reasoner who are extremely knowledgeable in all kinds of fields including
Computer Science, Math, Physics, Literature, History, Chemistry, Logical reasoning, Culture,
Language..... You are also able to find different high-level thought for different tasks. Here
are three reasoning sturctures:
i) Prompt-based structure:
It has a good performance when dealing with problems like Common Sense Reasoning,
Application Scheduling
ii) Procedure-based structure
It has a good performance when dealing with creative tasks like Creative Language
Generation, and Text Comprehension
iii) Programming-based:
It has a good performance when dealing with Mathematical Reasoning and Code Programming, it can also transform real-world problems into programming problem which could be
solved efficiently.
(Reasoning instantiation)
Your task is:
1. Deliberately consider the context and the problem within the distilled respond from
problem distiller and use your understanding of the question within the distilled respond to
find a domain expert who are suitable to solve the problem.
2. Consider the distilled information, choose one reasoning structures for the problem.
3. If the thought-template is provided, directly follow the thought-template to instantiate for
the given problem"""


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


class WordSortingProblem(BaseModel):
    id: str
    input: str
    target: str
    metadata: Dict[str, Any] | None = Field(default=None)


@dataclass(frozen=True)
class WordSortingTask:
    name: str = "wordsorting"

    def load_problems(self, jsonl_path: Path) -> Iterable[Any]:
        problems: list[WordSortingProblem] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for index, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not obj.get("id"):
                    obj["id"] = f"wordsorting-{index}"
                problems.append(WordSortingProblem.model_validate(obj))
        return problems

    def to_user_query(self, problem: Any) -> UserQuery:
        metadata = dict(problem.metadata or {})
        metadata["task"] = self.name
        metadata["target"] = problem.target
        return UserQuery(id=problem.id, question=problem.input, metadata=metadata)

    def build_solver_prompt(
        self,
        mode: str,
        query: UserQuery,
        retrieval: RetrievalResult,
    ) -> tuple[str, str]:
        io_system = (
            "You are solving a word sorting task.\n"
            "Return only the sorted words on a single line."
        )
        cot_system = (
            "You are solving a word sorting task.\n"
            "You may reason briefly, but the first non-empty line must be the final sorted words."
        )

        user_text = f"Input: {query.question}\nOutput:"
        if mode == "io":
            return io_system, user_text
        if mode == "cot":
            return cot_system, user_text
        if mode == "graph_bot_exec":
            exec_system = (
                f"{META_REASONER_SYSTEM}\n\n{cot_system}\n"
                "For graph_bot_exec, output exactly two parts in order:\n"
                "1) A fenced python code block (```python ... ```) that prints exactly one line with only the final answer.\n"
                "2) A final <answer>...</answer> block containing the same final answer candidate.\n"
                "Constraints: no imports; no file or network access; print only the final answer."
            )
            return (
                exec_system,
                f"{user_text}\nRetrieved templates/context:\n{retrieval.concatenated_context}\n",
            )

        return (
            f"{META_REASONER_SYSTEM}\n\n{cot_system}",
            f"{user_text}\nRetrieved templates/context:\n{retrieval.concatenated_context}\n",
        )

    def extract_candidate(self, raw_output: str, query: UserQuery) -> str:
        del query
        answer_matches = list(_ANSWER_BLOCK_PATTERN.finditer(raw_output))
        if answer_matches:
            answer_text = _normalize_whitespace(answer_matches[-1].group(1))
            if answer_text:
                return answer_text

        text_without_code = _PYTHON_CODE_BLOCK_PATTERN.sub("\n", raw_output)
        for line in text_without_code.splitlines():
            candidate = _normalize_whitespace(line)
            if candidate:
                return candidate
        return ""

    def oracle_validate(
        self, candidate: str, query: UserQuery, problem: Any
    ) -> tuple[bool, str | None]:
        target = ""
        if query.metadata:
            target = str(query.metadata.get("target", ""))
        if not target and problem is not None:
            if isinstance(problem, dict):
                target = str(problem.get("target", ""))
            else:
                target = str(getattr(problem, "target", ""))

        normalized_candidate = _normalize_whitespace(candidate)
        normalized_target = _normalize_whitespace(target)

        if not normalized_target:
            return False, "missing_target"
        if normalized_candidate == normalized_target:
            return True, None
        return False, "mismatch"

    def summarize_steps(
        self, raw_output: str, candidate: str, query: UserQuery
    ) -> Dict[str, Any]:
        del raw_output
        return {
            "task": self.name,
            "query": query.question,
            "candidate": candidate,
            "summary": f"Sort words alphabetically and return: {candidate}",
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
