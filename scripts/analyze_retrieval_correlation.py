#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List

def analyze_correlation(file_paths: List[Path]) -> None:
    """
    Analyzes the correlation between retrieval hits and problem success.
    Calculates P(Success | Hit) vs P(Success | Miss).
    """
    hits_solved = 0
    hits_failed = 0
    miss_solved = 0
    miss_failed = 0

    for file_path in file_paths:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Use .get() to avoid KeyError if fields are missing
                    solved = data.get("solved", False)
                    hit = data.get("retrieval_hit", False)

                    if hit:
                        if solved:
                            hits_solved += 1
                        else:
                            hits_failed += 1
                    else:
                        if solved:
                            miss_solved += 1
                        else:
                            miss_failed += 1
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {file_path}", file=sys.stderr)

    total_hits = hits_solved + hits_failed
    total_miss = miss_solved + miss_failed

    p_success_hit = hits_solved / total_hits if total_hits > 0 else 0.0
    p_success_miss = miss_solved / total_miss if total_miss > 0 else 0.0
    delta = p_success_hit - p_success_miss

    print("Contingency Table:")
    print(f"{'':<15} | {'Solved':<10} | {'Failed':<10} | {'Total':<10}")
    print("-" * 55)
    print(f"{'Retrieval Hit':<15} | {hits_solved:<10} | {hits_failed:<10} | {total_hits:<10}")
    print(f"{'Retrieval Miss':<15} | {miss_solved:<10} | {miss_failed:<10} | {total_miss:<10}")
    print("-" * 55)
    print(f"P(Success | Hit):  {p_success_hit:.4f}")
    print(f"P(Success | Miss): {p_success_miss:.4f}")
    print(f"Delta (Lift):      {delta:.4f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze correlation between retrieval hits and problem success.")
    parser.add_argument(
        "pattern", 
        nargs="?", 
        default="outputs/stream_logs/exp1_*.problems.jsonl", 
        help="Glob pattern for problem JSONL files (default: outputs/stream_logs/exp1_*.problems.jsonl)"
    )
    args = parser.parse_args()

    # Expand the glob pattern
    files = [Path(p) for p in glob.glob(args.pattern)]
    
    if not files:
        print(f"No files found matching pattern: {args.pattern}")
        sys.exit(1)
    
    print(f"Analyzing {len(files)} files...")
    analyze_correlation(files)

if __name__ == "__main__":
    main()
