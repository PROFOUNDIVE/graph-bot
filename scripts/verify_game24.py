#!/usr/bin/env python3
import json
from pathlib import Path


def main():
    p = Path("data/game24_buffer_98.jsonl")
    lines = [
        line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 98, f"Expected 98 lines, got {len(lines)}"
    for line in lines:
        obj = json.loads(line)
        assert set(obj.keys()) == {"id", "numbers", "target"}
        assert isinstance(obj["numbers"], list)
        assert isinstance(obj["target"], int)
    print("OK")


if __name__ == "__main__":
    main()
