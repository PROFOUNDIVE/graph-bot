from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from .logsetting import logger
from .types import SeedData, ReasoningTree, UserQuery
from .pipelines.build_trees import build_reasoning_trees_from_seeds
from .adapters.graphrag import GraphRAGAdapter
from .pipelines.main_loop import answer_with_retrieval, postprocess_after_T_inputs
from .pipelines.stream_loop import run_continual_stream
from .settings import settings
from .utils.amortization import generate_amortization_curve

from vllm.entrypoints.openai.api_server import run_server  # noqa: E402
from vllm.entrypoints.openai.cli_args import (  # noqa: E402
    make_arg_parser as _vllm_make_arg_parser,
    validate_parsed_serve_args,
)
import argparse
from multiprocessing import Process
import socket
import signal
import os
import sys
import time
import uvloop  # noqa: E402


app = typer.Typer(help="Graph-augmented Buffer of Thoughts (stubs)")


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
    try:
        # 인자 없이 동작하는 버전
        return _vllm_make_arg_parser()
    except TypeError:
        # 커스텀 파서를 기대하는 버전: deprecated 키워드를 무시하는 파서 전달
        return _vllm_make_arg_parser(_CompatArgumentParser(prog="vllm-api-server"))


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
        validate_parsed_serve_args(args)
        logger.debug(f"args: {args}")

        # 서버 프로세스 생성 (모듈 직접 호출)
        proc = Process(
            target=_serve_entrypoint,
            kwargs=dict(
                args_namespace=args,
                cuda_visible_devices=str(gpu_id),
                log_file=log_file,
            ),
            daemon=False,  # 부모 종료 후에도 계속 동작
        )
        proc.start()

        # PID 파일 기록
        pid_file.write_text(str(proc.pid))
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


@app.command("seeds-build")
def seeds_build(
    seeds_file: Path = typer.Argument(
        ..., help="JSONL file of seeds: {id, content, metadata?}"
    ),
    out_file: Optional[Path] = typer.Option(
        None, "--out", help="Write resulting trees to JSON file"
    ),
):
    """Implement HiAR-ICL to get reasoning trees.

    Input: Seed data (JSONL)
    Output: Reasoning tree (printed or saved JSON)
    """
    seeds: List[SeedData] = []
    with seeds_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            seeds.append(SeedData.model_validate(obj))

    trees = build_reasoning_trees_from_seeds(seeds)
    trees_json = [t.model_dump() for t in trees]

    if out_file:
        _dump_json_file(out_file, trees_json)
        typer.echo(f"Wrote {len(trees)} trees to {out_file}")
    else:
        typer.echo(json.dumps(trees_json, ensure_ascii=False, indent=2))


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


@app.command("postprocess")
def postprocess(
    t_inputs: int = typer.Option(0, "--t", help="After T inputs, run postprocessing"),
):
    """After T inputs, do reranking/verbalization/pruning/augmentation."""
    pruned = postprocess_after_T_inputs(t_inputs)
    typer.echo(f"Postprocess pruned {pruned} nodes after T={t_inputs} inputs")


@app.command("stream")
def stream(
    problems_file: Path = typer.Argument(
        ..., help="JSONL file of Game24 problems: {id, numbers, target?}"
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
    validator_mode: Optional[str] = typer.Option(
        None,
        "--validator-mode",
        help="Validator mode: oracle, exec_repair, weak_llm_judge",
    ),
    max_problems: Optional[int] = typer.Option(
        None, "--max-problems", help="Optional limit on number of problems"
    ),
):
    """Run continual stream loop for Game of 24."""
    run_continual_stream(
        problems_file=problems_file,
        mode=mode,
        use_edges=use_edges,
        policy_id=policy_id,
        validator_mode=validator_mode or settings.validator_mode,
        max_problems=max_problems,
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
    generate_amortization_curve(
        stream_metrics_jsonl=stream_metrics_jsonl, out_csv=out_csv
    )
    typer.echo(f"Wrote amortization curve CSV to {out_csv}")


@app.command("retrieve")
def retrieve(
    query: str = typer.Argument(..., help="User query"),
    k: Optional[int] = typer.Option(None, "--k", help="Override top-k paths"),
    show_paths: bool = typer.Option(
        False, "--show-paths", help="Print retrieved paths"
    ),
    task: Optional[str] = typer.Option(
        None, "--task", help="Task label for retrieval metadata"
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
):
    """Retrieve & Instantiate per input w/ k optimal paths and answer via LLM."""
    metadata = {"task": task} if task else None
    q = UserQuery(id="q-1", question=query, metadata=metadata)
    adapter = GraphRAGAdapter(
        mode=mode or settings.mode,
        use_edges=use_edges if use_edges is not None else settings.use_edges,
        policy_id=policy_id or settings.policy_id,
    )
    result = adapter.retrieve_paths(q, k=k or settings.top_k_paths)
    if show_paths:
        typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    answer = answer_with_retrieval(q, retrieval=result, adapter=adapter)
    typer.echo(answer.answer)


@app.command("loop-once")
def loop_once(
    query: str = typer.Argument(
        ..., help="User query to answer and then generate new trees for"
    ),
    task: Optional[str] = typer.Option(
        None, "--task", help="Task label for query/seed metadata"
    ),
):
    """Return to 1: Input user query & k optimal paths -> Output reasoning trees."""
    metadata = {"task": task} if task else None
    # 1) Retrieve & answer
    q = UserQuery(id="q-1", question=query, metadata=metadata)
    adapter = GraphRAGAdapter()
    retrieval = adapter.retrieve_paths(q, k=settings.top_k_paths)
    ans = answer_with_retrieval(q, retrieval=retrieval, adapter=adapter)
    typer.echo(ans.answer)

    # 2) Use the query as a seed to generate new trees and insert
    seed = SeedData(id="seed-from-query", content=query, metadata=metadata)
    trees = build_reasoning_trees_from_seeds([seed])
    inserted = adapter.insert_trees(trees)
    typer.echo(f"Generated and inserted {inserted} new trees from query")
