from __future__ import annotations

import argparse
from pathlib import Path

from graph_bot.utils.task12_seed import write_task12_seeded_metagraph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a seeded metagraph for Task 12 Gate 0 mock validation"
    )
    parser.add_argument("--out", required=True, help="Output metagraph JSON path")
    parser.add_argument(
        "--numbers",
        nargs="*",
        type=int,
        default=None,
        help="Problem numbers that should match the seeded retrieval root",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = Path(args.out)
    write_task12_seeded_metagraph(out_path, numbers=args.numbers)
    print(f"Wrote seeded Task 12 metagraph to {out_path}")


if __name__ == "__main__":
    main()
