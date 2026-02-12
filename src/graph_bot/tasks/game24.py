from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from ..datatypes import RetrievalResult, UserQuery
from ..eval.validators import Game24Validator
from ..pipelines.stream_loop import _normalize_candidate_line, load_game24_problems


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


@dataclass(frozen=True)
class Game24Task:
    name: str = "game24"

    def load_problems(self, jsonl_path: Path) -> Iterable[Any]:
        return load_game24_problems(jsonl_path)

    def to_user_query(self, problem: Any) -> UserQuery:
        return problem.to_user_query()

    def build_solver_prompt(
        self,
        mode: str,
        query: UserQuery,
        retrieval: RetrievalResult,
    ) -> tuple[str, str]:
        got_io_system = """You are playing the 24 Game.

Given four numbers, use each number exactly once (in any order) and only the operations +, -, *, /.
Parentheses are allowed. Create one valid Python expression that evaluates to 24.

Output must follow the example format as closely as possible: a single line of the form
Output: <python_expression>
Do not include any additional explanation or text."""

        got_io_user_template = """<Example>
Input: 4 9 10 13
Output: (10 - 4) * (13 - 9)
</Example>

Input: {input}
Output:"""

        got_cot_system = """You are playing the 24 Game.

Given four numbers, use each number exactly once (in any order) and only the operations +, -, *, /.
Parentheses are allowed. Find one valid Python expression that evaluates to 24.

You may show intermediate steps, but you MUST enclose the final Python expression in <answer> tags:
<answer>(1 + 2) * 8</answer>
Do not include any additional text inside the tags."""

        got_cot_user_template = """<Examples>
Input: 4 9 10 13
Work:
(10 - 4) = 6
(13 - 9) = 4
6 * 4 = 24
<answer>(10 - 4) * (13 - 9)</answer>

Input: 1 3 4 6
Work:
(6 / 3) = 2
4 * 2 = 8
8 * 3 = 24  (using 1 and 3? not allowed) -> backtrack
(4 - 1) = 3
6 * 3 = 18
18 + 3 = 21 -> backtrack
(6 - 1) = 5
5 * 4 = 20
20 + 3 = 23 -> backtrack
(6 - 4) = 2
3 * 2 = 6
6 * 4 = 24 (uses 4 twice) -> backtrack
(6 / (1 - 3/4)) = 24
<answer>6 / (1 - 3/4)</answer>
</Examples>

Input: {input}
"""

        numbers = [
            int(float(x)) for x in re.findall(r"(-?\d+\.?\d*)", query.question)[:4]
        ]
        numbers_str = " ".join(str(x) for x in numbers)

        if mode == "io":
            return got_io_system, got_io_user_template.format(input=numbers_str)
        if mode == "cot":
            return got_cot_system, got_cot_user_template.format(input=numbers_str)

        base_user = got_cot_user_template.format(input=numbers_str)
        return (
            f"{META_REASONER_SYSTEM}\n\n{got_cot_system}",
            f"{base_user}\nRetrieved templates/context:\n{retrieval.concatenated_context}\n",
        )

    def extract_candidate(self, raw_output: str, query: UserQuery) -> str:
        numbers = [
            int(float(x)) for x in re.findall(r"(-?\d+\.?\d*)", query.question)[:4]
        ]
        candidate, _ = _normalize_candidate_line(raw_output, allowed_numbers=numbers)
        return candidate

    def oracle_validate(
        self, candidate: str, query: UserQuery, problem: Any
    ) -> tuple[bool, str | None]:
        del problem
        validator = Game24Validator()
        is_valid = bool(validator.validate(candidate, query.question))
        if is_valid:
            return True, None
        return False, validator.failure_reason(candidate, query.question)

    def summarize_steps(
        self, raw_output: str, candidate: str, query: UserQuery
    ) -> Dict[str, Any]:
        del raw_output
        return {
            "task": self.name,
            "query": query.question,
            "candidate": candidate,
            "summary": f"Use arithmetic operations to derive 24 with expression: {candidate}",
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
