from __future__ import annotations

from pathlib import Path
from graph_bot.pipelines.stream_loop import run_continual_stream


def _assert_jsonl_valid(path: Path) -> None:
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0
    import json

    for line in lines:
        if not line.strip():
            continue
        json.loads(line)


def test_integration_modes_with_mock(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/game24_smoke.jsonl")
    modes = {
        "bot": tmp_path / "stream_logs_bot",
        "got": tmp_path / "stream_logs_got",
        "tot": tmp_path / "stream_logs_tot",
    }
    runs = {mode: f"test_run_{mode}" for mode in modes}
    for mode, out_dir in modes.items():
        run_id = runs[mode]
        run_continual_stream(
            problems_file=fixture,
            task="game24",
            mode=mode,
            metrics_out_dir=out_dir,
            run_id=run_id,
        )
        for ext in (
            ".calls.jsonl",
            ".problems.jsonl",
            ".stream.jsonl",
            ".token_events.jsonl",
        ):
            path = out_dir / f"{run_id}{ext}"
            _assert_jsonl_valid(path)
