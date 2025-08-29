from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from .core.models import SeedData, ReasoningTree, UserQuery
from .pipelines.build_trees import build_reasoning_trees_from_seeds
from .adapters.graphrag import GraphRAGAdapter
from .pipelines.main_loop import answer_with_retrieval


app = typer.Typer(help="Graph-augmented Buffer of Thoughts (stubs)")


def _load_json_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json_file(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
    t_inputs: int = typer.Option(
        0, "--t", help="After T inputs, run postprocessing (stub)"
    ),
):
    """After T inputs, do reranking/verbalization/pruning/augmentation (stub)."""
    # Placeholder - no-op since adapter is in-memory per process
    typer.echo(f"Postprocess stub executed after T={t_inputs} inputs")


@app.command("retrieve")
def retrieve(
    query: str = typer.Argument(..., help="User query"),
    k: Optional[int] = typer.Option(None, "--k", help="Override top-k paths"),
    show_paths: bool = typer.Option(
        False, "--show-paths", help="Print retrieved paths"
    ),
):
    """Retrieve & Instantiate per input w/ k optimal paths and answer via LLM (stub)."""
    if k is not None:
        # Temporarily override via environment-like setting by monkeypatching settings if needed.
        # Kept simple: adapter.retrieve_paths takes explicit k when needed.
        pass
    q = UserQuery(id="q-1", question=query)
    _adapter = GraphRAGAdapter()
    result = _adapter.retrieve_paths(q, k=k or 3)
    if show_paths:
        typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    # Produce answer text
    answer = answer_with_retrieval(q)
    typer.echo(answer.answer)


@app.command("loop-once")
def loop_once(
    query: str = typer.Argument(
        ..., help="User query to answer and then generate new trees for"
    ),
):
    """Return to 1: Input user query & k optimal paths -> Output reasoning trees (stub)."""
    # 1) Retrieve & answer
    q = UserQuery(id="q-1", question=query)
    ans = answer_with_retrieval(q)
    typer.echo(ans.answer)

    # 2) Use the query as a seed to generate new trees and insert
    seed = SeedData(id="seed-from-query", content=query)
    trees = build_reasoning_trees_from_seeds([seed])
    _adapter = GraphRAGAdapter()
    inserted = _adapter.insert_trees(trees)
    typer.echo(f"Generated and inserted {inserted} new trees from query")
