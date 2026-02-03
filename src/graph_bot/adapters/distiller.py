from __future__ import annotations

import re
from typing import List

from ..datatypes import ReasoningNode, ReasoningTree
from ..interfaces import AbstractDistiller


class GraphRAGDistiller(AbstractDistiller):
    def distill_query(self, query: str) -> str:
        pattern = re.compile(r"(-?\d+\.?\d*)")
        all_matches = pattern.findall(query)

        if not all_matches:
            return query

        try:
            nums = [int(float(x)) for x in all_matches]
        except ValueError:
            return query

        if len(nums) < 4:
            return query

        inputs = nums[:4]
        inputs.sort()

        sorted_str = " ".join(str(x) for x in inputs)
        return f"Solve 24 with {sorted_str}"

    def distill_trace(self, tree: ReasoningTree) -> List[ReasoningNode]:
        answer_text = _extract_answer_text(tree)
        query = _extract_query(tree)
        if query is None:
            query = ""

        distilled_text = _distill_trace_text(answer_text, query)
        solved = _extract_solved(tree)

        return [
            ReasoningNode(
                node_id=tree.root_id,
                text=distilled_text,
                type="thought",
                attributes={
                    "subtype": "template",
                    "quality": {"validator_passed": solved},
                    "original_answer": answer_text,
                },
            )
        ]


def _extract_answer_text(tree: ReasoningTree) -> str:
    for node in tree.nodes:
        if node.node_id == tree.root_id:
            return node.text
    if tree.nodes:
        return tree.nodes[0].text
    return ""


def _extract_query(tree: ReasoningTree) -> str | None:
    if tree.provenance:
        query = tree.provenance.get("query") or tree.provenance.get("question")
        if isinstance(query, str) and query:
            return query

    for node in tree.nodes:
        if not node.attributes:
            continue
        query = node.attributes.get("query") or node.attributes.get("question")
        if isinstance(query, str) and query:
            return query

    return None


def _extract_solved(tree: ReasoningTree) -> bool:
    if tree.provenance and isinstance(tree.provenance.get("solved"), bool):
        return bool(tree.provenance["solved"])
    return False


def _distill_trace_text(answer_text: str, query: str) -> str:
    pattern = re.compile(r"(-?\d+\.?\d*)")
    all_nums = pattern.findall(query)

    normalized_nums: list[str] = []
    for x in all_nums:
        try:
            num = float(x)
            if num.is_integer():
                normalized_nums.append(str(int(num)))
            else:
                normalized_nums.append(x)
        except ValueError:
            normalized_nums.append(x)

    inputs = normalized_nums[:4]
    input_str = " ".join(inputs)

    template = f"Task: Game24\nInput: {input_str}\nSolution: {answer_text}"

    if len(template) > 500:
        template = template[:500] + "..."

    return template
