from __future__ import annotations

import json

from graph_bot.pipelines.stream_loop import load_game24_problems, run_continual_stream
from graph_bot.settings import settings


def _write_jsonl(file_path, rows):
    file_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _configure_mock_stream_settings(monkeypatch, metagraph_path):
    monkeypatch.setattr(settings, "metagraph_path", metagraph_path)
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "retry_max_attempts", 1)


def _assert_metrics_artifacts(metrics_dir, run_id):
    for suffix in ("calls", "problems", "stream", "token_events"):
        assert (metrics_dir / f"{run_id}.{suffix}.jsonl").exists()


def _load_metagraph(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _template_tasks(graph_data):
    tasks = []
    for node in graph_data.get("nodes", []):
        attributes = node.get("attributes") or {}
        if (
            node.get("type") == "thought"
            and attributes.get("subtype") == "template"
            and attributes.get("task")
        ):
            tasks.append(str(attributes["task"]))
    return tasks


def _sum_n_used_for_task(graph_data, task_name):
    total = 0
    for node in graph_data.get("nodes", []):
        attributes = node.get("attributes") or {}
        if attributes.get("task") != task_name:
            continue
        stats = attributes.get("stats") or {}
        total += int(stats.get("n_used", 0))
    return total


def test_load_game24_problems(tmp_path):
    problems_file = tmp_path / "problems.jsonl"
    problems = [
        {"id": "q1", "numbers": [2, 5, 8, 11]},
        {"id": "q2", "numbers": [1, 3, 4, 6], "target": 24},
    ]
    problems_file.write_text(
        "\n".join(json.dumps(problem) for problem in problems) + "\n",
        encoding="utf-8",
    )

    loaded = load_game24_problems(problems_file)

    assert len(loaded) == 2
    assert loaded[0].id == "q1"
    assert loaded[0].target == 24.0

    query = loaded[0].to_user_query()
    assert query.id == "q1"
    assert query.metadata is not None
    assert query.metadata["task"] == "game24"


def test_run_continual_stream_baseline_io(tmp_path, monkeypatch):
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 4, 6, 8], "target": 24}) + "\n",
        encoding="utf-8",
    )

    metrics_dir = tmp_path / "metrics"

    monkeypatch.setattr(settings, "metagraph_path", tmp_path / "metagraph.json")
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "retry_max_attempts", 1)

    results = run_continual_stream(
        problems_file=problems_file,
        mode="io",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id="test_io",
    )

    assert len(results) == 1
    assert results[0]["problem_id"] == "q1"
    assert results[0]["solved"] is True

    assert (metrics_dir / "test_io.calls.jsonl").exists()
    assert (metrics_dir / "test_io.problems.jsonl").exists()
    assert (metrics_dir / "test_io.stream.jsonl").exists()
    assert (metrics_dir / "test_io.token_events.jsonl").exists()


def test_run_continual_stream_graph_bot_mode(tmp_path, monkeypatch):
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 4, 6, 8], "target": 24}) + "\n",
        encoding="utf-8",
    )

    metrics_dir = tmp_path / "metrics"
    metagraph_path = tmp_path / "metagraph.json"

    monkeypatch.setattr(settings, "metagraph_path", metagraph_path)
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "retry_max_attempts", 1)

    run_continual_stream(
        problems_file=problems_file,
        mode="graph_bot",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id="test_run1",
    )

    results = run_continual_stream(
        problems_file=problems_file,
        mode="graph_bot",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id="test_run2",
    )

    assert len(results) == 1
    reuse_count = results[0]["reuse_count"]
    assert isinstance(reuse_count, int)
    assert reuse_count >= 1
    assert (metrics_dir / "test_run2.stream.jsonl").exists()


def test_run_continual_stream_all_tasks_create_artifacts_and_task_templates(
    tmp_path, monkeypatch
):
    task_cases = [
        (
            "game24",
            [{"id": "g1", "numbers": [2, 4, 6, 8], "target": 24}],
            "g1",
        ),
        (
            "wordsorting",
            [
                {
                    "id": "w1",
                    "input": "Sort these words alphabetically: pear apple banana",
                    "target": "apple banana pear",
                }
            ],
            "w1",
        ),
        (
            "mgsm",
            [
                {
                    "id": "m1",
                    "question": (
                        "Mina has 3 pencils and gets 5 more. "
                        "How many pencils does she have now?"
                    ),
                    "answer": 8,
                }
            ],
            "m1",
        ),
    ]

    for task_name, rows, expected_problem_id in task_cases:
        case_dir = tmp_path / task_name
        case_dir.mkdir()
        problems_file = case_dir / "problems.jsonl"
        metrics_dir = case_dir / "metrics"
        metagraph_path = case_dir / "metagraph.json"
        run_id = f"test_{task_name}_artifacts"

        _write_jsonl(problems_file, rows)
        _configure_mock_stream_settings(monkeypatch, metagraph_path)

        results = run_continual_stream(
            problems_file=problems_file,
            task=task_name,
            mode="graph_bot",
            max_problems=1,
            metrics_out_dir=metrics_dir,
            run_id=run_id,
        )

        assert len(results) == 1
        assert results[0]["problem_id"] == expected_problem_id
        assert results[0]["solved"] is True
        _assert_metrics_artifacts(metrics_dir, run_id)

        graph_data = _load_metagraph(metagraph_path)
        assert task_name in _template_tasks(graph_data)


def test_run_continual_stream_retrieval_isolation_and_cross_task_override(
    tmp_path, monkeypatch
):
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()

    game24_file = shared_dir / "game24.jsonl"
    wordsorting_file = shared_dir / "wordsorting.jsonl"
    metrics_dir = shared_dir / "metrics"
    metagraph_path = shared_dir / "metagraph.json"

    _write_jsonl(
        game24_file,
        [{"id": "g1", "numbers": [2, 4, 6, 8], "target": 24}],
    )
    _write_jsonl(
        wordsorting_file,
        [
            {
                "id": "w1",
                "input": "Sort these words alphabetically: pear apple banana",
                "target": "apple banana pear",
            }
        ],
    )

    _configure_mock_stream_settings(monkeypatch, metagraph_path)

    game24_seed_run_id = "test_isolation_game24_seed"
    wordsorting_cold_start_run_id = "test_isolation_wordsorting_cold_start"
    wordsorting_seed_run_id = "test_isolation_wordsorting_seed"
    wordsorting_default_run_id = "test_isolation_wordsorting_default"
    wordsorting_cross_task_run_id = "test_isolation_wordsorting_cross_task"

    game24_seed = run_continual_stream(
        problems_file=game24_file,
        task="game24",
        mode="graph_bot",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id=game24_seed_run_id,
    )
    assert game24_seed[0]["solved"] is True
    _assert_metrics_artifacts(metrics_dir, game24_seed_run_id)

    graph_before_cold_start = _load_metagraph(metagraph_path)
    game24_n_used_before_cold_start = _sum_n_used_for_task(
        graph_before_cold_start, "game24"
    )

    wordsorting_cold_start = run_continual_stream(
        problems_file=wordsorting_file,
        task="wordsorting",
        mode="graph_bot",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id=wordsorting_cold_start_run_id,
    )
    assert wordsorting_cold_start[0]["solved"] is True
    _assert_metrics_artifacts(metrics_dir, wordsorting_cold_start_run_id)

    graph_after_cold_start = _load_metagraph(metagraph_path)
    game24_n_used_after_cold_start = _sum_n_used_for_task(
        graph_after_cold_start, "game24"
    )

    assert game24_n_used_after_cold_start == game24_n_used_before_cold_start

    wordsorting_seed = run_continual_stream(
        problems_file=wordsorting_file,
        task="wordsorting",
        mode="graph_bot",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id=wordsorting_seed_run_id,
    )
    assert wordsorting_seed[0]["solved"] is True
    _assert_metrics_artifacts(metrics_dir, wordsorting_seed_run_id)

    graph_before_default = _load_metagraph(metagraph_path)
    game24_n_used_before_default = _sum_n_used_for_task(graph_before_default, "game24")
    wordsorting_n_used_before_default = _sum_n_used_for_task(
        graph_before_default, "wordsorting"
    )

    wordsorting_default = run_continual_stream(
        problems_file=wordsorting_file,
        task="wordsorting",
        cross_task_retrieval=False,
        mode="graph_bot",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id=wordsorting_default_run_id,
    )
    assert wordsorting_default[0]["solved"] is True
    _assert_metrics_artifacts(metrics_dir, wordsorting_default_run_id)

    graph_after_default = _load_metagraph(metagraph_path)
    game24_n_used_after_default = _sum_n_used_for_task(graph_after_default, "game24")
    wordsorting_n_used_after_default = _sum_n_used_for_task(
        graph_after_default, "wordsorting"
    )

    assert game24_n_used_after_default == game24_n_used_before_default
    assert wordsorting_n_used_after_default >= wordsorting_n_used_before_default

    wordsorting_cross_task = run_continual_stream(
        problems_file=wordsorting_file,
        task="wordsorting",
        cross_task_retrieval=True,
        mode="graph_bot",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id=wordsorting_cross_task_run_id,
    )
    assert wordsorting_cross_task[0]["solved"] is True
    _assert_metrics_artifacts(metrics_dir, wordsorting_cross_task_run_id)

    graph_after_cross_task = _load_metagraph(metagraph_path)
    game24_n_used_after_cross_task = _sum_n_used_for_task(
        graph_after_cross_task, "game24"
    )

    assert game24_n_used_after_cross_task > game24_n_used_after_default
