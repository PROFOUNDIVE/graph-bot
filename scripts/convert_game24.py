#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
import random


from typing import Optional


def parse_line(line: str) -> Optional[Dict]:
    # Each line is like: 00001| {"input": "2 5 8 11", "target": "24"}
    if "|" not in line:
        line = line.strip()
        if not line:
            return None
        # fallback: try to parse as raw json
        return json.loads(line)
    _prefix, json_str = line.split("|", 1)
    json_str = json_str.strip()
    return json.loads(json_str)


def to_entry(obj: Dict, idx: int) -> Dict:
    nums = [int(x) for x in obj["input"].split()]
    tgt = int(obj["target"])
    return {
        "id": f"game24-{idx:04d}",
        "numbers": nums,
        "target": tgt,
    }


def write_jsonl(path: Path, entries: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main() -> int:
    # Locate source data (provided by the repository guidelines)
    repo_root = Path(__file__).resolve().parents[1]  # graph-bot repo root
    source_path = Path(
        "/home/hyunwoo/git/buffer-of-thought-llm/benchmarks/gameof24.jsonl"
    )
    # Fallback if environment differs
    if not source_path.exists():
        source_path = (
            repo_root.parent / "buffer-of-thought-llm" / "benchmarks" / "gameof24.jsonl"
        )
    if not source_path.exists():
        print(f"Source not found: {source_path}", flush=True)
        return 1

    # Read original 98 entries
    lines: List[str] = []
    with source_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            lines.append(raw)

    items: List[Dict] = []
    for i, line in enumerate(lines, start=1):
        data = parse_line(line)
        if data is None:
            continue
        items.append(to_entry(data, i))

    # Ensure we have exactly 98 items
    if len(items) != 98:
        print(f"Expected 98 items, got {len(items)}", flush=True)
        return 1

    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) base order
    base_path = data_dir / "game24_buffer_98.jsonl"
    write_jsonl(base_path, items)

    # 2) deterministic shuffles
    seeds = [41, 43, 47]
    shuffle_paths = []
    for seed in seeds:
        rng = random.Random(seed)
        shuffled = items.copy()
        rng.shuffle(shuffled)
        shuffle_path = (
            data_dir / f"game24_buffer_98.shuffle_{seeds.index(seed) + 1}.jsonl"
        )
        write_jsonl(shuffle_path, shuffled)
        shuffle_paths.append(shuffle_path)

    # 3) seed_10: first 10 from shuffle_1, rest: remaining 88 from shuffle_1
    shuffle1 = []
    with open(shuffle_paths[0], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                shuffle1.append(json.loads(line))
    seed10_path = data_dir / "game24_buffer_98.seed_10.jsonl"
    rest_path = data_dir / "game24_buffer_98.rest.jsonl"
    seed10_entries = shuffle1[:10]
    rest_entries = shuffle1[10:]
    write_jsonl(seed10_path, seed10_entries)
    write_jsonl(rest_path, rest_entries)

    # 4) manifest
    manifest_dir = repo_root / "outputs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "exp_manifest.json"
    manifest = {
        "source": str(source_path.resolve()),
        "shuffles": seeds,
        "M": 10,
        "files": [
            str(base_path),
            str(shuffle_paths[0]),
            str(shuffle_paths[1]),
            str(shuffle_paths[2]),
            str(seed10_path),
            str(rest_path),
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Game24 data prepared successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
