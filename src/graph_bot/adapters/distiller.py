from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, List, Dict

from ..datatypes import ReasoningNode, ReasoningTree
from ..interfaces import AbstractDistiller
from ..logsetting import logger
from ..settings import settings


class RuleBasedDistiller(AbstractDistiller):
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
        task = _extract_task(tree)
        steps_summary = _extract_steps_summary(tree)
        final_candidate = _extract_final_candidate(tree, answer_text)

        distilled_text = _distill_trace_text(
            task=task,
            query=query,
            steps_summary=steps_summary,
            final_candidate=final_candidate,
        )
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


class LLMDistiller(AbstractDistiller):
    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.llm_model
        self._fallback_distiller = RuleBasedDistiller()

    def distill_query(self, query: str) -> str:
        raw_query = query
        normalized_query = raw_query.strip()
        if not normalized_query:
            return raw_query

        system_prompt = (
            "[Problem Distiller]\n"
            "As a highly professional and intelligent expert in information distillation, "
            "you excel at extracting essential information to solve problems from user input "
            "queries. You adeptly transform this extracted information into a suitable format "
            "based on the respective type of the issue.\n\n"
            "Please categorize and extract the crucial information required to solve the "
            "problem from the user's input query. The distilled information MUST include:\n"
            "1. Key information:\n"
            "Values and information of key variables extracted from user input, which will "
            "be handed over to the respective expert for task resolution, ensuring all "
            "essential information required to solve the problem is provided.\n"
            "2. Restrictions:\n"
            "The objective of the problem and corresponding constraints.\n"
            "3. Distilled task:\n"
            "Extend the problem based on 1 and 2. Summarize a meta problem that can address "
            "the user query and handle more input and output variations. Incorporate the "
            "real-world scenario of the extended problem along with the types of key variables "
            "and information constraints from the original problem to restrict the key variables "
            "in the extended problem. After that, use the user query input key information as "
            "input to solve the problem as an example.\n\n"
            "Hard constraints:\n"
            "- Do NOT emit raw chain-of-thought.\n"
            "- Output MUST be plain text and MUST contain exactly these top-level section headers "
            "in this order: Key information:, Restrictions:, Distilled task:.\n"
            "- Keep the example concise; do not over-explain."
        )
        user_prompt = f"User input query:\n{normalized_query}"

        distilled_query = self._chat(system=system_prompt, user=user_prompt)
        if not distilled_query:
            return normalized_query

        normalized_distilled_query = _normalize_multiline(distilled_query)
        if not normalized_distilled_query:
            return normalized_query
        return normalized_distilled_query

    def distill_trace(self, tree: ReasoningTree) -> List[ReasoningNode]:
        if not tree.nodes:
            return []

        answer_text = _extract_answer_text(tree)
        query = _extract_query(tree) or ""
        solved = _extract_solved(tree)
        task = _extract_task(tree)
        steps_summary = _extract_steps_summary(tree)
        final_candidate = _extract_final_candidate(tree, answer_text)
        distill_input = _extract_distill_input(
            tree,
            task=task,
            query=query,
            steps_summary=steps_summary,
            final_candidate=final_candidate,
        )

        fallback_text = _distill_trace_text(
            task=task,
            query=query,
            steps_summary=steps_summary,
            final_candidate=final_candidate,
        )

        system_prompt = (
            "Prompt for Template Distillation:\n"
            "User: [Problem Description] + [Solution Steps or Code]\n"
            "To extract and summarize the high-level paradigms and general approaches for solving such\n"
            "problems, please follow these steps in your response:\n"
            "1. Core task summarization:\n"
            "Identify and describe the basic type and core challenges of the problem, such as classifying it\n"
            "as a mathematical problem (e.g., solving a quadratic equation), a data structure problem (e.g.,\n"
            "array sorting), an algorithm problem (e.g., search algorithms), etc. And analyze the most\n"
            "efficient way to solve the problem.\n"
            "2. Solution Steps Description:\n"
            "Outline the general solution steps, including how to define the problem, determine variables,\n"
            "list key equations or constraints, choose appropriate solving strategies and methods, and how\n"
            "to verify the correctness of the results.\n"
            "3. General Answer Template:\n"
            "Based on the above analysis, propose a template or approach that can be widely applied\n"
            "to this type of problem, including possible variables, functions, class definitions, etc. If it\n"
            "is a programming problem, provide a set of base classes and interfaces that can be used to\n"
            "construct solutions to specific problems.\n"
            "Please ensure that your response is highly concise and structured, so that specific solutions\n"
            "can be transformed into generalizable methods.\n"
            "[Optional] Here are some exemplars of the thought-template: (Choose cross-task or\n"
            "in-task exemplars based on the analysis of the Core task summarization.)\n\n"
            "Hard constraints:\n"
            "- Do NOT emit raw chain-of-thought.\n"
            "- Output plain text only.\n"
            "- Be concise and structured."
        )
        user_prompt = (
            "User:\n"
            "[Problem Description]\n"
            f"{query}\n\n"
            "[Solution Steps or Code]\n"
            f"{steps_summary}\n\n"
            "[Final Candidate]\n"
            f"{final_candidate or ''}\n\n"
            "[Optional thought-template exemplars]\n"
            "<none>"
        )
        distilled_text = self._chat(system=system_prompt, user=user_prompt)
        if not distilled_text:
            distilled_text = fallback_text
        distilled_text = _ensure_task_prefixed_template(task=task, text=distilled_text)
        distilled_text = _truncate_template(distilled_text)

        node_id = tree.root_id or tree.nodes[0].node_id or "distilled-template"
        now_iso = datetime.now(timezone.utc).isoformat()
        parse_ok = bool(_normalize_whitespace(distilled_text))

        return [
            ReasoningNode(
                node_id=node_id,
                text=distilled_text,
                type="thought",
                attributes={
                    "task": task,
                    "subtype": "template",
                    "created_at": now_iso,
                    "last_used_at": now_iso,
                    "quality": {
                        "validator_passed": solved,
                        "parse_ok": parse_ok,
                    },
                    "stats": {
                        "n_seen": 0,
                        "n_used": 0,
                        "n_success": 0,
                        "n_fail": 0,
                        "ema_success": 0.0,
                        "avg_tokens": 0.0,
                        "avg_latency_ms": 0.0,
                        "avg_cost_usd": 0.0,
                    },
                    "original_answer": answer_text,
                    "steps_summary": steps_summary,
                    "final_candidate": final_candidate,
                },
            )
        ]

    def _chat(self, *, system: str, user: str) -> str | None:
        try:
            client = self._build_client()
            response_text, _ = client.chat(system=system, user=user, temperature=0.0)
        except Exception as exc:
            logger.warning(f"LLMDistiller call failed: {exc}")
            return None

        sanitized = _sanitize_llm_output(response_text)
        return sanitized or None

    def _build_client(self) -> Any:
        if settings.llm_provider == "mock":
            from .mock_client import MockLLMClient

            return MockLLMClient(model=self._model)

        from .vllm_openai_client import VLLMOpenAIClient

        return VLLMOpenAIClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=self._model,
        )


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


def _extract_task(tree: ReasoningTree) -> str:
    if tree.provenance:
        task = tree.provenance.get("task")
        if isinstance(task, str) and task:
            return task
    return "game24"


def _extract_steps_summary(tree: ReasoningTree) -> str:
    if not tree.provenance:
        return ""
    steps_summary = tree.provenance.get("steps_summary")
    if isinstance(steps_summary, str):
        return steps_summary.strip()
    if isinstance(steps_summary, dict):
        summary = steps_summary.get("summary")
        if isinstance(summary, str):
            return summary.strip()
        return _normalize_whitespace(str(steps_summary))
    if steps_summary is None:
        return ""
    return _normalize_whitespace(str(steps_summary))


def _extract_final_candidate(tree: ReasoningTree, answer_text: str) -> str | None:
    if tree.provenance:
        final_candidate = tree.provenance.get("final_candidate")
        if isinstance(final_candidate, str):
            normalized = final_candidate.strip()
            if normalized:
                return normalized
    normalized_answer = answer_text.strip()
    if normalized_answer:
        return normalized_answer
    return None


def _extract_distill_input(
    tree: ReasoningTree,
    *,
    task: str,
    query: str,
    steps_summary: str,
    final_candidate: str | None,
) -> str:
    if tree.provenance:
        value = tree.provenance.get("distill_input")
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return _build_distill_input(
        task=task,
        query=query,
        steps_summary=steps_summary,
        final_candidate=final_candidate,
    )


def _build_distill_input(
    *,
    task: str,
    query: str,
    steps_summary: str,
    final_candidate: str | None,
) -> str:
    lines = [
        f"Task: {task}",
        f"Problem: {query}",
        f"Solution Steps Summary: {steps_summary}",
    ]
    if final_candidate:
        lines.append(f"Final Candidate: {final_candidate}")
    return "\n".join(lines)


def _distill_trace_text(
    *,
    task: str,
    query: str,
    steps_summary: str,
    final_candidate: str | None,
) -> str:
    summary_text = _normalize_whitespace(steps_summary)
    if not summary_text:
        summary_text = "Use a concise, valid procedure to derive the final answer."

    template_lines = [
        f"Task: {task}",
        "Thought Template:",
        f"1. Restate the problem: {query}",
        f"2. Apply the summarized method: {summary_text}",
        "Applicability:",
        f"- Reuse for similar {task} problems.",
        "Answer Schema:",
    ]
    if final_candidate:
        template_lines.append(f"- Final Candidate: {final_candidate}")
    else:
        template_lines.append("- Final Candidate: <final answer>")

    template = "\n".join(template_lines)

    if len(template) > 500:
        template = template[:500] + "..."

    return template


def _ensure_task_prefixed_template(*, task: str, text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return f"Task: {task}"

    lines = normalized.splitlines()
    if lines and lines[0].lower().startswith("task:"):
        lines[0] = f"Task: {task}"
        return "\n".join(lines).strip()

    return f"Task: {task}\n{normalized}"


def _is_cold_start_query(query: str) -> bool:
    numbers = re.findall(r"(-?\d+\.?\d*)", query)
    has_alpha = bool(re.search(r"[A-Za-z]", query))
    token_count = len(query.split())
    return len(numbers) >= 4 and (not has_alpha or token_count <= 8)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_multiline(text: str) -> str:
    value = text.strip()
    if not value:
        return ""
    lines = [ln.strip() for ln in value.splitlines()]
    out: List[str] = []
    last_was_empty = False
    for ln in lines:
        if not ln:
            if not out:
                continue
            if last_was_empty:
                continue
            out.append("")
            last_was_empty = True
            continue
        out.append(ln)
        last_was_empty = False
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def _sanitize_llm_output(text: str) -> str:
    value = text.strip()
    if not value:
        return ""

    if value.startswith("```"):
        lines = value.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()

    prefixes = ("normalized query:", "query:", "template:", "distilled:")
    lowered = value.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            value = value[len(prefix) :].strip()
            break

    return value


def _truncate_template(text: str) -> str:
    if len(text) > 500:
        return text[:500] + "..."
    return text


class NullDistiller(AbstractDistiller):
    """A no-op distiller used when distiller-mode is set to 'none'."""

    def distill_query(self, query: str) -> str:
        return query

    def distill_trace(self, tree: ReasoningTree) -> List[ReasoningNode]:
        # No distillation; return empty list
        return []


_DISTILLER_REGISTRY: Dict[str, type[AbstractDistiller]] = {
    "rulebased": RuleBasedDistiller,
    "llm": LLMDistiller,
    "none": NullDistiller,
}


def get_distiller(mode: str) -> AbstractDistiller:
    """Factory function to instantiate a distiller by mode.

    Args:
        mode: Distiller mode. One of: rulebased, llm, none

    Returns:
        An instance of a class implementing AbstractDistiller
    """
    distiller_class = _DISTILLER_REGISTRY.get(mode)
    if distiller_class is None:
        raise ValueError(f"Unknown distiller mode: {mode}")
    return distiller_class()
