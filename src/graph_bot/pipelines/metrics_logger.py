from __future__ import annotations

import json
import uuid
from pathlib import Path

from ..datatypes import StreamCallMetrics, StreamCumulativeMetrics, StreamProblemMetrics


class StreamMetricsLogger:
    def __init__(self, *, out_dir: Path, run_id: str) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id

        self._calls_path = self._out_dir / f"{run_id}.calls.jsonl"
        self._problems_path = self._out_dir / f"{run_id}.problems.jsonl"
        self._stream_path = self._out_dir / f"{run_id}.stream.jsonl"
        self._token_events_path = self._out_dir / f"{run_id}.token_events.jsonl"

        self._cumulative_solved = 0
        self._cumulative_cost_usd = 0.0
        self._cumulative_contaminated = 0
        self._cumulative_retrieved_templates = 0
        self._cumulative_updates = 0
        self._cumulative_poisoned_updates = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def calls_path(self) -> Path:
        return self._calls_path

    @property
    def problems_path(self) -> Path:
        return self._problems_path

    @property
    def stream_path(self) -> Path:
        return self._stream_path

    def new_call_id(self) -> str:
        return str(uuid.uuid4())

    def log_call(self, event: StreamCallMetrics) -> None:
        self._append_jsonl(self._calls_path, event.model_dump())

    def log_token_event(self, event: dict) -> None:
        self._append_jsonl(self._token_events_path, event)

    def log_problem(self, event: StreamProblemMetrics) -> StreamCumulativeMetrics:
        self._append_jsonl(self._problems_path, event.model_dump())

        if event.solved:
            self._cumulative_solved += 1
        self._cumulative_cost_usd += float(event.api_cost_usd)

        if event.contamination_rate is not None:
            retrieved = max(event.reuse_count, 0)
            self._cumulative_retrieved_templates += retrieved
            contaminated = int(round(float(event.contamination_rate) * retrieved))
            self._cumulative_contaminated += contaminated

        if event.poisoned_update_rate is not None:
            self._cumulative_updates += 1
            self._cumulative_poisoned_updates += float(event.poisoned_update_rate)

        cost_per_solved = self._cumulative_cost_usd / max(1, self._cumulative_solved)
        contamination_rate = None
        if self._cumulative_retrieved_templates > 0:
            contamination_rate = (
                self._cumulative_contaminated / self._cumulative_retrieved_templates
            )

        cumulative_poisoned_rate = None
        if self._cumulative_updates > 0:
            cumulative_poisoned_rate = (
                self._cumulative_poisoned_updates / self._cumulative_updates
            )

        cumulative = StreamCumulativeMetrics(
            t=event.t,
            cumulative_solved=self._cumulative_solved,
            cumulative_api_cost_usd=self._cumulative_cost_usd,
            cost_per_solved=cost_per_solved,
            contamination_rate=contamination_rate,
            poisoned_update_rate=cumulative_poisoned_rate,
        )
        self._append_jsonl(self._stream_path, cumulative.model_dump())
        return cumulative

    def _append_jsonl(self, path: Path, obj: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
