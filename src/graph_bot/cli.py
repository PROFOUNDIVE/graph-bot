from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional, cast

import typer

from .logsetting import logger
from .datatypes import ReasoningTree
from .adapters.graphrag import GraphRAGAdapter
from .pipelines.stream_loop import run_continual_stream
from .settings import settings
from .utils.amortization import generate_amortization_curve
import argparse
from multiprocessing import Process
import socket
import signal
import os
import sys
import time


app = typer.Typer(help="Graph-augmented Buffer of Thoughts")


class _CompatArgumentParser(argparse.ArgumentParser):
    """vLLM이 add_argument(..., deprecated=True) 등을 넘겨도 깨지지 않게 하는 호환 파서."""

    def add_argument(self, *args, **kwargs):
        # vLLM가 사용하는 확장 키워드를 무시
        kwargs.pop("deprecated", None)
        return super().add_argument(*args, **kwargs)

    # 그룹에도 동일하게 적용되도록 그룹 팩토리 오버라이드
    def add_argument_group(self, *args, **kwargs):
        group = _CompatArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group

    def add_mutually_exclusive_group(self, **kwargs):
        group = _CompatMutuallyExclusiveGroup(self, **kwargs)
        self._mutually_exclusive_groups.append(group)
        return group


class _CompatArgumentGroup(argparse._ArgumentGroup):
    """그룹 레벨 add_argument에도 deprecated 키워드가 들어오므로 여기서도 무시."""

    def add_argument(self, *args, **kwargs):
        kwargs.pop("deprecated", None)
        return super().add_argument(*args, **kwargs)


class _CompatMutuallyExclusiveGroup(argparse._MutuallyExclusiveGroup):
    """mutually exclusive group 경로도 동일 처리."""

    def add_argument(self, *args, **kwargs):
        kwargs.pop("deprecated", None)
        return super().add_argument(*args, **kwargs)


def _make_vllm_parser():
    """
    vLLM make_arg_parser의 시그니처/키워드 차이를 흡수하기 위한 호환 래퍼
    - 일부 버전: make_arg_parser()  (인자 없이)
    - 일부 버전: make_arg_parser(parser) (커스텀 파서 필요, deprecated 키워드 사용)
    """
    # NOTE: vllm import is intentionally lazy to avoid importing vllm (and its
    # transitive deps like sentencepiece) during normal CLI import/test
    # collection.
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser as _vllm_make_arg_parser,
    )

    make_arg_parser = cast(Any, _vllm_make_arg_parser)

    try:
        # 인자 없이 동작하는 버전
        return make_arg_parser()
    except TypeError:
        # 커스텀 파서를 기대하는 버전: deprecated 키워드를 무시하는 파서 전달
        return make_arg_parser(_CompatArgumentParser(prog="vllm-api-server"))


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def _load_json_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json_file(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _serve_entrypoint(
    *,
    args_namespace,
    cuda_visible_devices: Optional[str],
    log_file: Path,
):
    """
    별도 프로세스에서 실행되는 엔트리포인트:
    - CUDA_VISIBLE_DEVICES 설정
    - 표준 출력/에러를 로그 파일로 리다이렉트
    - uvloop.run(run_server(args)) 호출
    """
    # Lazy imports keep `import graph_bot.cli` lightweight.
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server

    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    # 줄단위 버퍼링으로 실시간 로그
    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, "a", buffering=1, encoding="utf-8")
    # 가능한 한 광범위한 로그를 잡기 위해 stdout/stderr를 파일로 바꿉니다.
    sys.stdout = f  # type: ignore[assignment]
    sys.stderr = f  # type: ignore[assignment]

    logger.debug(f"[vllm] Starting server with args: {args_namespace}")
    try:
        uvloop.run(run_server(args_namespace))
    except Exception as e:
        logger.error(f"[vllm] FATAL: {e!r}")
        raise
    finally:
        try:
            f.flush()
        except Exception:
            pass
        # 파일 핸들은 프로세스 종료 시 OS가 회수함.


@app.command("llm-server")
def llm_server(
    option: str = typer.Argument(
        ..., help="LLM server option", show_default=False, metavar="[start|stop]"
    ),
    # 실행 파라미터 (sh 스크립트의 기본값을 반영)
    model_name: str = typer.Option(
        "/home/hyunwoo/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        "--model",
        help="HF repo or local model path",
        rich_help_panel="Execution Parameters",
    ),
    served_model_name: str = typer.Option(
        "llama3-8b-instruct",
        help="Name exposed by the OpenAI-compatible API",
        rich_help_panel="Execution Parameters",
    ),
    gpu_id: int = typer.Option(
        0,
        "--gpu-id",
        help="CUDA_VISIBLE_DEVICES index",
        rich_help_panel="Execution Parameters",
    ),
    host: str = typer.Option(
        "0.0.0.0", help="Bind host", rich_help_panel="Execution Parameters"
    ),
    port: int = typer.Option(
        2427, help="Bind port", rich_help_panel="Execution Parameters"
    ),
    max_model_len: int = typer.Option(
        5000, help="--max-model-len", rich_help_panel="Execution Parameters"
    ),
    tensor_parallel_size: int = typer.Option(
        1, help="--tensor-parallel-size", rich_help_panel="Execution Parameters"
    ),
    dtype: str = typer.Option(
        "bfloat16", help="--dtype", rich_help_panel="Execution Parameters"
    ),
    max_num_seqs: int = typer.Option(
        512, help="--max-num-seqs", rich_help_panel="Execution Parameters"
    ),
    gpu_memory_utilization: float = typer.Option(
        0.75, help="--gpu-memory-utilization", rich_help_panel="Execution Parameters"
    ),
    # 관리 파라미터
    max_wait: int = typer.Option(
        500,
        help="Max seconds to wait for server readiness",
        rich_help_panel="Management Parameters",
    ),
    sleep_interval: int = typer.Option(
        20,
        help="Polling interval for readiness check",
        rich_help_panel="Management Parameters",
    ),
    log_file: Path = typer.Option(
        Path("vllm_server.log"),
        help="Server log file path",
        rich_help_panel="Management Parameters",
    ),
    pid_file: Path = typer.Option(
        Path("vllm_server.pid"),
        help="PID file path",
        rich_help_panel="Management Parameters",
    ),
    # 추후 arguments 변경(configs에서 관리하도록 수정) 예정
):
    opt = option.lower().strip()
    if opt not in {"start", "stop"}:
        raise typer.BadParameter("option must be 'start' or 'stop'.")
    if opt == "start":
        logger.debug("start option detected")
        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text().strip())
                os.kill(old_pid, 0)  # 프로세스 존재 여부 확인
                logger.warn(f"Server seems already running (pid={old_pid}).")
                raise typer.Exit(code=0)
            except (OSError, ValueError):
                # PID 파일이 낡았거나 프로세스 없음 => 계속 진행
                pass

        # vLLM args 구성
        parser = _make_vllm_parser()
        cli_argv = [
            "--model",
            model_name,
            "--served-model-name",
            served_model_name,
            "--host",
            host,
            "--port",
            str(port),
            "--max-model-len",
            str(max_model_len),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--dtype",
            dtype,
            "--max-num-seqs",
            str(max_num_seqs),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
        ]
        args = parser.parse_args(cli_argv)

        # Lazy import (vLLM) only when actually starting server.
        from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args

        args_namespace = cast(argparse.Namespace, args)
        validate_parsed_serve_args(cast(Any, args_namespace))
        logger.debug(f"args: {args_namespace}")

        # 서버 프로세스 생성 (모듈 직접 호출)
        proc = Process(
            target=_serve_entrypoint,
            kwargs=dict(
                args_namespace=args_namespace,
                cuda_visible_devices=str(gpu_id),
                log_file=log_file,
            ),
            daemon=False,
        )
        proc.start()

        pid_file.write_text(str(proc.pid))

        proc.join(timeout=0.1)

        if not proc.is_alive():
            typer.secho(
                f"[ERROR] Server process exited early. Check {log_file}", fg="red"
            )
            raise typer.Exit(code=1)
        typer.secho(
            f"Starting vLLM server on GPU {gpu_id} with port {port} (pid={proc.pid})...",
            fg="cyan",
        )
        typer.secho(f"Logs -> {log_file}", fg="cyan")

        # 포트 레디니스 대기 (nc -z 대체)
        elapsed = 0
        while not _port_open("127.0.0.1", port):
            time.sleep(sleep_interval)
            elapsed += sleep_interval
            typer.echo(f"Still waiting for vLLM server... ({elapsed}s elapsed)")
            if not proc.is_alive():
                typer.secho(
                    f"[ERROR] Server process exited early. Check {log_file}", fg="red"
                )
                raise typer.Exit(code=1)
            if elapsed >= max_wait:
                typer.secho(
                    f"[ERROR] vLLM server failed to start within {max_wait}s. Check {log_file}",
                    fg="red",
                )
                try:
                    if proc.pid is not None:
                        os.kill(proc.pid, signal.SIGTERM)
                except Exception:
                    pass
                raise typer.Exit(code=1)

        typer.secho(
            f"✅ vLLM server is up after {elapsed}s (http://127.0.0.1:{port}).",
            fg="green",
        )
    elif opt == "stop":
        logger.debug("stop option detected")
        if not pid_file.exists():
            typer.secho("No PID file found; server may not be running.", fg="yellow")
            raise typer.Exit(code=0)
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            typer.secho("PID file is corrupted. Removing it.", fg="yellow")
            pid_file.unlink(missing_ok=True)
            raise typer.Exit(code=0)

        try:
            os.kill(pid, signal.SIGTERM)
            typer.secho(f"Sent SIGTERM to vLLM server (pid={pid}).", fg="cyan")
        except ProcessLookupError:
            typer.secho("Process not found; cleaning up PID file.", fg="yellow")
        finally:
            pid_file.unlink(missing_ok=True)

        # 종료 확인 (최대 10초)
        for _ in range(20):
            if not _port_open("127.0.0.1", port):
                break
            time.sleep(0.5)

        typer.secho("✅ vLLM server stopped.", fg="green")


@app.command("trees-insert")
def trees_insert(
    trees_file: Path = typer.Argument(..., help="JSON file: list[ReasoningTree]"),
):
    """Insert trees as graphs of documents in GraphRAG DB.

    Input: Reasoning trees (JSON)
    Output: Write in GraphRAG DB (counts)
    """
    raw = _load_json_file(trees_file)
    trees = [ReasoningTree.model_validate(x) for x in raw]
    _adapter = GraphRAGAdapter()
    count = _adapter.insert_trees(trees)
    typer.echo(f"Inserted {count} trees")


@app.command("stream")
def stream(
    problems_file: Path = typer.Argument(
        ...,
        help=(
            "JSONL problems file by task: "
            "game24={id,numbers,target?}; "
            "wordsorting={id,input,target}; "
            "mgsm={id?,question,answer,language?,metadata?}"
        ),
    ),
    task: str = typer.Option(
        "game24", "--task", help="Task name: game24, wordsorting, mgsm"
    ),
    cross_task_retrieval: bool = typer.Option(
        False,
        "--cross-task-retrieval/--no-cross-task-retrieval",
        help="Allow retrieval across task boundaries (default: task-scoped retrieval).",
    ),
    mode: Optional[str] = typer.Option(
        None, "--mode", help="Execution mode: graph_bot or flat_template_rag"
    ),
    use_edges: Optional[bool] = typer.Option(
        None, "--use-edges", help="Use graph edges for path construction"
    ),
    policy_id: Optional[str] = typer.Option(
        None,
        "--policy-id",
        help="Selection policy: semantic_only or semantic_topK_stats_rerank",
    ),
    retrieval_backend: Optional[
        Literal["sparse_jaccard", "dense_template"]
    ] = typer.Option(
        None,
        "--retrieval-backend",
        help="Retrieval backend: sparse_jaccard or dense_template",
    ),
    validator_mode: Optional[str] = typer.Option(
        None,
        "--validator-mode",
        help="Validator mode: oracle, exec_repair, weak_llm_judge",
    ),
    distiller_mode: Optional[str] = typer.Option(
        None,
        "--distiller-mode",
        help="Distiller mode to use: rulebased, llm, or none",
    ),
    validator_model: Optional[str] = typer.Option(
        None,
        "--validator-model",
        help="Validator model to use for WeakLLMJudge (env GRAPH_BOT_VALIDATOR_MODEL if unset)",
    ),
    max_problems: Optional[int] = typer.Option(
        None, "--max-problems", help="Optional limit on number of problems"
    ),
    run_id: str = typer.Option("run", "--run-id", help="Run id prefix for log files"),
    metrics_out_dir: Path = typer.Option(
        Path("outputs/stream_logs"),
        "--metrics-out-dir",
        help="Directory to write stream JSONL logs",
    ),
    validator_gated_update: bool = typer.Option(
        True,
        "--validator-gated-update/--no-validator-gated-update",
        help="If True, only insert solved problems into MetaGraph.",
    ),
):
    """Run continual stream loop for a selected task."""
    # Resolve final validator model: CLI arg takes precedence, else settings
    final_validator_model = validator_model or getattr(
        settings, "validator_model", None
    )
    run_continual_stream(
        problems_file=problems_file,
        task=task,
        cross_task_retrieval=cross_task_retrieval,
        mode=mode,
        use_edges=use_edges,
        policy_id=policy_id,
        retrieval_backend=retrieval_backend,
        validator_mode=validator_mode or getattr(settings, "validator_mode", "oracle"),
        validator_model=final_validator_model,
        distiller_mode=distiller_mode,
        validator_gated_update=validator_gated_update,
        max_problems=max_problems,
        run_id=run_id,
        metrics_out_dir=metrics_out_dir,
    )


@app.command("amortize")
def amortize(
    stream_metrics_jsonl: Path = typer.Argument(
        ..., help="Path to *.stream.jsonl output from stream run"
    ),
    out_csv: Path = typer.Option(
        Path("outputs/amortization_curve.csv"), "--out", help="Output CSV path"
    ),
):
    """Generate EXP1 amortization curve from stream logs."""
    if not stream_metrics_jsonl.exists():
        typer.secho(f"Error: {stream_metrics_jsonl} not found.", fg="red")
        raise typer.Exit(code=1)

    run_id = stream_metrics_jsonl.name.replace(".stream.jsonl", "")
    log_dir = stream_metrics_jsonl.parent

    problems_path = log_dir / f"{run_id}.problems.jsonl"
    events_path = log_dir / f"{run_id}.token_events.jsonl"

    if not problems_path.exists():
        typer.secho(f"Error: Required sibling {problems_path} not found.", fg="red")
        raise typer.Exit(code=1)
    if not events_path.exists():
        typer.secho(f"Error: Required sibling {events_path} not found.", fg="red")
        raise typer.Exit(code=1)

    generate_amortization_curve(
        stream_metrics_jsonl=stream_metrics_jsonl,
        out_csv=out_csv,
        problems_jsonl=problems_path,
        token_events_jsonl=events_path,
    )
    typer.echo(f"Wrote amortization curve CSV to {out_csv}")
