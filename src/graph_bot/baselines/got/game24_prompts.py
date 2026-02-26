from __future__ import annotations

import json
from typing import Any

GAME24_NEXT_MOVE_PROMPT_JSONL = """<Instruction>
You are an expert player of the 24 Game.
Goal: Combine numbers using +, -, *, / to reach 24.
Task: Generate exactly {num_branches} diverse next moves from the given state.

State: A list of items, each with an 'id' and 'value'.
Move: Select two different ids and an operator.

Rules for high-quality generation:
1. Diversity is key. Use a mix of all operators (+, -, *, /). Do not rely only on + and *.
2. Commutativity:
   - + and * are commutative. picking [a, b] is the same as [b, a].
   - - and / are NOT commutative. Consider BOTH [a, b] and [b, a] when valid.
   - Example: 10 - 4 = 6, but 4 - 10 = -6. Both might be useful.
3. Strategic thinking:
   - Prioritize operations that create factors of 24 (3, 4, 6, 8, 12).
   - Prioritize operations that create numbers close to 24.
   - Do not ignore fractions if they can lead to 24 (e.g., 6 / (1/4) = 24).
4. Valid JSONL: output ONLY valid JSON objects, one per line.

Schema:
{{"pick": [id1, id2], "op": "operator"}}
</Instruction>

<Example>
Current Items:
[{{"id": 0, "value": 10.0, "expr": "10"}}, {{"id": 1, "value": 4.0, "expr": "4"}}, {{"id": 2, "value": 2.0, "expr": "2"}}]

Output:
{{"pick": [0, 1], "op": "-"}}
{{"pick": [1, 0], "op": "-"}}
{{"pick": [0, 2], "op": "/"}}
{{"pick": [0, 1], "op": "+"}}
{{"pick": [1, 2], "op": "*"}}
{{"pick": [0, 2], "op": "-"}}
... (up to {num_branches} lines)
</Example>

<CurrentItems>
{items_json}
</CurrentItems>

Output:
"""

GAME24_IMPROVE_PROMPT_JSONL = """<Instruction>
You are correcting a failed move in the 24 Game search.
The last move led to a dead end, so propose a DIFFERENT move.

Previous items:
{prev_items_json}

Last move (do NOT repeat this exact pick/op):
{last_move_json}

Output exactly one candidate move as a single JSON object line:
{{"pick": [id1, id2], "op": "operator"}}

Requirements:
- pick two different ids from the previous items.
- op must be one of: "+", "-", "*", "/".
- avoid repeating the same pick/op as the last move.
- for division, do NOT choose a divisor that is 0.
</Instruction>

Output:
"""


def build_next_move_prompt(*, items_json: str, num_branches: int) -> str:
    return GAME24_NEXT_MOVE_PROMPT_JSONL.format(
        num_branches=num_branches,
        items_json=items_json,
    )


def build_improve_prompt(
    *, prev_items_json: str, last_move: dict[str, Any] | None
) -> str:
    return GAME24_IMPROVE_PROMPT_JSONL.format(
        prev_items_json=prev_items_json,
        last_move_json=json.dumps(last_move),
    )
