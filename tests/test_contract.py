from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import jsonschema
import pytest

from graph_bot.datatypes import ReasoningNode, RetrievalResult, UserQuery
from graph_bot.pipelines.stream_loop import (
    _solve_with_retries,
    run_continual_stream,
    _normalize_candidate_line,
)
from graph_bot.settings import settings


def test_normalize_candidate_line():
    assert _normalize_candidate_line("  \n  (1+2+3)*4  \n  ") == (
        "(1+2+3)*4",
        "fallback_bottom_scan",
    )
    assert _normalize_candidate_line("") == ("", "empty")
    assert _normalize_candidate_line("\n\n") == ("", "empty")
    assert _normalize_candidate_line("Answer: 24\n1+2+3+4") == (
        "1+2+3+4",
        "fallback_bottom_scan",
    )
    assert _normalize_candidate_line("Output: (1+2+3)*4 = 24") == (
        "(1+2+3)*4",
        "got_output_format",
    )


def test_normalize_candidate_priority_and_edge_cases():
    # 1. Conflict: <answer>A</answer> vs Output: B -> Expect A.
    raw_1 = """
    Reasoning...
    Output: (1+2+3)*4 = 24
    <answer>(4-1)*8 = 24</answer>
    """
    assert _normalize_candidate_line(raw_1) == ("(4-1)*8", "answer_block")

    # 2. Malformed: <answer>A vs Output: B -> Expect B.
    raw_2 = """
    Reasoning...
    Output: (1+2+3)*4 = 24
    <answer>(4-1)*8 = 24
    """
    assert _normalize_candidate_line(raw_2) == ("(1+2+3)*4", "got_output_format")

    # 3. Multiple: <answer>A</answer> ... <answer>B</answer> -> Expect A (current regex behavior).
    raw_3 = """
    <answer>(1+2+3)*4 = 24</answer>
    <answer>(4-1)*8 = 24</answer>
    """
    assert _normalize_candidate_line(raw_3) == ("(1+2+3)*4", "answer_block")


class _StubMetrics:
    def __init__(self) -> None:
        self.run_id = "test_stream_run"
        self._counter = 0
        self.calls: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []

    def new_call_id(self) -> str:
        self._counter += 1
        return f"call-{self._counter}"

    def log_call(self, call) -> None:
        self.calls.append(call.model_dump())

    def log_token_event(self, event: dict[str, object]) -> None:
        self.events.append(event)


class _OracleValidatorStub:
    def validate(self, node, problem=None):
        del node, problem
        return 1.0

    def failure_reason(self, answer: str, problem: str) -> str | None:
        del answer, problem
        return None

    def get_validator_name(self) -> str:
        return "oracle"


class _NodeValidatorStub:
    def __init__(self) -> None:
        self.last_node: ReasoningNode | None = None

    def validate(self, node, problem=None):
        del problem
        assert isinstance(node, ReasoningNode)
        self.last_node = node
        return 1.0

    def failure_reason(self, answer: str, problem: str) -> str | None:
        del answer, problem
        return None

    def get_validator_name(self) -> str:
        return "weak_llm_judge"


class _TaskStub:
    def __init__(self) -> None:
        self.modes: list[str] = []

    def build_solver_prompt(
        self, mode: str, query: UserQuery, retrieval: RetrievalResult
    ):
        del query, retrieval
        self.modes.append(mode)
        return "system", f"mode={mode}"

    def extract_candidate(self, raw_output: str, query: UserQuery) -> str:
        del query
        marker = "CANDIDATE:"
        idx = raw_output.rfind(marker)
        if idx == -1:
            return ""
        return raw_output[idx + len(marker) :].strip()

    def oracle_validate(self, candidate: str, query: UserQuery, problem: object):
        del query, problem
        if candidate == "42":
            return True, None
        return False, "oracle_mismatch"


def test_solve_with_retries_graph_bot_exec_fallback_and_exec_logging(monkeypatch):
    old_provider = settings.llm_provider
    old_model = settings.llm_model
    old_attempts = settings.retry_max_attempts
    old_t1 = settings.retry_temperature_1
    old_t2 = settings.retry_temperature_2
    old_t3 = settings.retry_temperature_3

    settings.llm_provider = "mock"
    settings.llm_model = "gpt-4o-mini"
    settings.retry_max_attempts = 2
    settings.retry_temperature_1 = 0.0
    settings.retry_temperature_2 = 0.0
    settings.retry_temperature_3 = 0.0

    users: list[str] = []

    huge_tail = "X" * 1500
    first_raw_output = (
        "```python\nprint('41')\n```\n" f"{huge_tail}\n" "CANDIDATE: 42\n"
    )
    second_raw_output = "No code\nCANDIDATE: 42\n"

    responses = [first_raw_output, second_raw_output]

    def mocked_chat(self, system, user, temperature):
        del self, system, temperature
        from graph_bot.adapters.mock_client import LLMUsage

        users.append(user)
        return responses.pop(0), LLMUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=3.0,
        )

    monkeypatch.setattr(
        "graph_bot.adapters.mock_client.MockLLMClient.chat", mocked_chat
    )

    metrics = _StubMetrics()
    task = _TaskStub()
    query = UserQuery(id="q1", question="2 4 6 8 -> 24", metadata={"task": "game24"})
    retrieval = RetrievalResult(query_id="q1", paths=[], concatenated_context="")

    try:
        answer, _, solved, solved_attempt, _, attempts_used = _solve_with_retries(
            task=task,
            problem={"id": "q1"},
            query=query,
            retrieval=retrieval,
            validator=_OracleValidatorStub(),
            metrics=metrics,
            pricing_table={
                "models": {
                    "gpt-4o-mini": {"input_usd_per_1k": 0.0, "output_usd_per_1k": 0.0}
                }
            },
            parent_call_id="retrieve-1",
            t=1,
            problem_id="q1",
            mode="graph_bot_exec",
        )
    finally:
        settings.llm_provider = old_provider
        settings.llm_model = old_model
        settings.retry_max_attempts = old_attempts
        settings.retry_temperature_1 = old_t1
        settings.retry_temperature_2 = old_t2
        settings.retry_temperature_3 = old_t3

    assert solved is True
    assert solved_attempt == 2
    assert attempts_used == 2
    assert answer == "42"

    assert task.modes == ["graph_bot_exec", "graph_bot"]

    exec_calls = [call for call in metrics.calls if call["operation"] == "exec"]
    assert len(exec_calls) == 1
    assert exec_calls[0]["failure_reason"] == "exec_mismatch"

    exec_tool_events = [
        event
        for event in metrics.events
        if event["event_type"] == "tool_call" and event["operation"] == "exec"
    ]
    assert len(exec_tool_events) == 1
    assert exec_tool_events[0]["model"] == "python"

    assert len(users) == 2
    marker = "Previous output: "
    retry_user = users[1]
    assert marker in retry_user
    injected = retry_user.split(marker, maxsplit=1)[1].split(
        "\nPlease fix it and provide a valid final answer in the required task format.",
        maxsplit=1,
    )[0]
    assert len(injected) <= 1000


def test_solve_with_retries_non_oracle_validator_gets_reasoning_node(monkeypatch):
    old_provider = settings.llm_provider
    old_model = settings.llm_model
    old_attempts = settings.retry_max_attempts

    settings.llm_provider = "mock"
    settings.llm_model = "gpt-4o-mini"
    settings.retry_max_attempts = 1

    def mocked_chat(self, system, user, temperature):
        del self, system, user, temperature
        from graph_bot.adapters.mock_client import LLMUsage

        return "Some reasoning\nCANDIDATE: final-answer", LLMUsage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_ms=2.0,
        )

    monkeypatch.setattr(
        "graph_bot.adapters.mock_client.MockLLMClient.chat", mocked_chat
    )

    metrics = _StubMetrics()
    task = _TaskStub()
    validator = _NodeValidatorStub()
    query = UserQuery(id="q2", question="A problem", metadata={"task": "wordsorting"})
    retrieval = RetrievalResult(query_id="q2", paths=[], concatenated_context="ctx")

    try:
        _solve_with_retries(
            task=task,
            problem={"id": "q2"},
            query=query,
            retrieval=retrieval,
            validator=validator,
            metrics=metrics,
            pricing_table={
                "models": {
                    "gpt-4o-mini": {"input_usd_per_1k": 0.0, "output_usd_per_1k": 0.0}
                }
            },
            parent_call_id="retrieve-2",
            t=7,
            problem_id="q2",
            mode="graph_bot",
        )
    finally:
        settings.llm_provider = old_provider
        settings.llm_model = old_model
        settings.retry_max_attempts = old_attempts

    assert validator.last_node is not None
    attrs = validator.last_node.attributes or {}
    assert attrs.get("problem") == "A problem"
    assert "raw_output" in attrs
    assert attrs.get("task") == "wordsorting"
    assert attrs.get("metrics") is metrics
    assert attrs.get("stream_run_id") == metrics.run_id
    assert attrs.get("problem_id") == "q2"
    assert attrs.get("t") == 7

    validate_calls = [c for c in metrics.calls if c["operation"] == "validate"]
    assert len(validate_calls) == 1


def test_stream_contract_and_schema():
    test_dir = Path("outputs/test_contract")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    problems_file = test_dir / "problems.jsonl"
    with problems_file.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps({"id": "q-contract", "numbers": [1, 2, 3, 4], "target": 24})
            + "\n"
        )
        f.write(
            json.dumps({"id": "q-timeout", "numbers": [5, 6, 7, 8], "target": 24})
            + "\n"
        )

    schema_path = Path("docs/specs/token_tracker_schema_v0.json")
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    old_timeout = settings.execution_timeout_sec
    old_provider = settings.llm_provider
    old_model = settings.llm_model
    old_retrieval_backend = settings.retrieval_backend
    old_embedding_provider = settings.embedding_provider

    try:
        settings.execution_timeout_sec = 0.5
        settings.llm_provider = "mock"
        settings.llm_model = "gpt-4o-mini"
        settings.retrieval_backend = "sparse_jaccard"

        metrics_dir = test_dir / "metrics"

        def mocked_chat(system, user, temperature):
            from graph_bot.adapters.mock_client import LLMUsage

            if "5 6 7 8" in user:
                time.sleep(1.0)

            return "(1+2+3)*4", LLMUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                latency_ms=200.0,
            )

        with patch(
            "graph_bot.adapters.mock_client.MockLLMClient.chat", side_effect=mocked_chat
        ):
            run_continual_stream(
                problems_file=problems_file,
                metrics_out_dir=metrics_dir,
                run_id="test_contract",
                max_problems=2,
            )

        calls_file = metrics_dir / "test_contract.calls.jsonl"
        token_events_file = metrics_dir / "test_contract.token_events.jsonl"
        assert calls_file.exists()
        assert token_events_file.exists()

        with calls_file.open("r", encoding="utf-8") as f:
            calls = [json.loads(line) for line in f]

        retrieve_calls = [c for c in calls if c["operation"] == "retrieve"]
        validate_calls = [c for c in calls if c["operation"] == "validate"]
        solve_calls = [c for c in calls if c["operation"] == "solve"]
        timeout_calls = [c for c in calls if c["operation"] == "timeout"]

        assert len(retrieve_calls) >= 1
        assert "latency_ms" in retrieve_calls[0]
        assert retrieve_calls[0]["latency_ms"] > 0

        assert len(validate_calls) >= 1
        assert "latency_ms" in validate_calls[0]
        assert validate_calls[0]["latency_ms"] >= 0

        assert len(solve_calls) >= 1
        assert "api_cost_usd" in solve_calls[0]
        assert solve_calls[0]["api_cost_usd"] > 0

        assert len(timeout_calls) >= 1
        assert timeout_calls[0]["error_type"] == "ERR_TIMEOUT"
        assert timeout_calls[0]["operation"] == "timeout"

        with token_events_file.open("r", encoding="utf-8") as f:
            token_events = []
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                token_events.append(event)

                try:
                    jsonschema.validate(instance=event, schema=schema)
                except jsonschema.ValidationError as e:
                    pytest.fail(f"Schema validation failed: {e}")

                if event["operation"] == "solve":
                    assert event["usage"]["prompt_tokens"] == 100
                    assert event["usage"]["completion_tokens"] == 50
                    assert event["cost_usd"] > 0
                elif event["operation"] == "timeout":
                    assert event["status"] == "timeout"
                    assert event["cost_usd"] == 0.0

        rag_retrieval_events = [
            event for event in token_events if event["event_type"] == "rag_retrieval"
        ]
        assert len(rag_retrieval_events) >= 1
        assert (
            rag_retrieval_events[0]["metadata"].get("retrieval_backend")
            == "sparse_jaccard"
        )
        assert rag_retrieval_events[0]["model"] == "sparse_jaccard"
        sparse_embedding_events = [
            event for event in token_events if event["event_type"] == "embedding"
        ]
        assert sparse_embedding_events == []

        dense_problems_file = test_dir / "problems_dense.jsonl"
        with dense_problems_file.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps({"id": "q-dense-1", "numbers": [1, 2, 3, 4], "target": 24})
                + "\n"
            )
            f.write(
                json.dumps({"id": "q-dense-2", "numbers": [2, 3, 4, 5], "target": 24})
                + "\n"
            )

        settings.retrieval_backend = "dense_template"
        settings.embedding_provider = "deterministic"

        metrics_dir_dense = test_dir / "metrics_dense"

        def mocked_chat_dense(system, user, temperature):
            del system, user, temperature
            from graph_bot.adapters.mock_client import LLMUsage

            return "(1+2+3)*4", LLMUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                latency_ms=10.0,
            )

        with patch(
            "graph_bot.adapters.mock_client.MockLLMClient.chat",
            side_effect=mocked_chat_dense,
        ):
            run_continual_stream(
                problems_file=dense_problems_file,
                metrics_out_dir=metrics_dir_dense,
                run_id="test_contract_dense",
                max_problems=2,
            )

        token_events_dense_file = (
            metrics_dir_dense / "test_contract_dense.token_events.jsonl"
        )
        assert token_events_dense_file.exists()

        with token_events_dense_file.open("r", encoding="utf-8") as f:
            token_events_dense = [json.loads(line) for line in f if line.strip()]

        rag_retrieval_events_dense = [
            event
            for event in token_events_dense
            if event["event_type"] == "rag_retrieval"
        ]
        assert len(rag_retrieval_events_dense) >= 1
        dense_event = rag_retrieval_events_dense[-1]
        assert dense_event["metadata"].get("retrieval_backend") == "dense_template"
        assert str(dense_event["model"]).startswith("deterministic-hash-")
        assert dense_event["metadata"].get("embedding_provider") == "deterministic"
        assert str(
            dense_event["metadata"].get("embedding_model_actual", "")
        ).startswith("deterministic-hash-")

        embedding_events_dense = [
            event for event in token_events_dense if event["event_type"] == "embedding"
        ]
        assert len(embedding_events_dense) >= 1
        dense_embedding_event = embedding_events_dense[-1]
        assert (
            dense_embedding_event["metadata"].get("retrieval_backend")
            == "dense_template"
        )
        assert (
            dense_embedding_event["metadata"].get("embedding_provider")
            == "deterministic"
        )
        assert str(dense_embedding_event["model"]).startswith("deterministic-hash-")

    finally:
        settings.execution_timeout_sec = old_timeout
        settings.llm_provider = old_provider
        settings.llm_model = old_model
        settings.retrieval_backend = old_retrieval_backend
        settings.embedding_provider = old_embedding_provider
        if test_dir.exists():
            shutil.rmtree(test_dir)
