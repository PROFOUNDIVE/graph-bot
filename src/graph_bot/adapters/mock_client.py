from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    audit_prompt_tokens: int | None = None


# Mock classes to mimic OpenAI API response structure
class MockMessage:
    def __init__(self, content: str):
        self.content = content


class MockChoice:
    def __init__(self, message: MockMessage):
        self.message = message


class MockUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockResponse:
    def __init__(self, choices: List[MockChoice], usage: MockUsage):
        self.choices = choices
        self.usage = usage


class MockCompletions:
    def __init__(self, model: str):
        self._model = model

    def create(
        self,
        *,
        model: str,
        messages: List[dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> MockResponse:
        # Extract user message
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Simple heuristic to extract numbers and form a valid expression
        numbers_match = re.findall(
            r"(?:Numbers:|Input:)\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)", user_content
        )
        text = "1 + 1"

        if numbers_match:
            nums = numbers_match[-1]
            # Hardcoded logic for test case "4 6 8 2"
            if sorted(nums) == sorted(["4", "6", "8", "2"]):
                text = "6 * 8 / (4 - 2)"
            else:
                text = f"{nums[0]} + {nums[1]} + {nums[2]} + {nums[3]}"

        # Simulate response structure
        return MockResponse(
            choices=[MockChoice(message=MockMessage(content=text))],
            usage=MockUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )


class MockChat:
    def __init__(self, model: str):
        self.completions = MockCompletions(model)


class MockClient:
    def __init__(self, model: str):
        self.chat = MockChat(model)


class MockLLMClient:
    """Mock LLM client that returns fixed answers for Game of 24, mimicking OpenAI structure.
    Enhanced to deterministically support additional tasks used in tests:
    - wordsorting: return alphabetically sorted words parsed from the problem input.
    - mgsm: return the gold answer encoded as a final numeric value (as text).
    This keeps outputs predictable and parser-friendly for the extractor logic.
    """

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        timeout_s: float = 120.0,
    ) -> None:
        self._model = model
        # Internal mock client that looks like openai.OpenAI()
        self._client = MockClient(model)

    def chat(
        self, *, system: str, user: str, temperature: float = 0.0
    ) -> tuple[str, LLMUsage]:
        start = time.perf_counter()

        if "24-game search assistant" in system and "JSONL move objects" in system:
            raw_output = self._maybe_build_game24_moves(user)
            if raw_output is not None:
                latency_ms = (time.perf_counter() - start) * 1000.0 + 100.0
                prompt_tokens = max(1, len(user) // 4)
                completion_tokens = max(1, len(raw_output) // 8)
                return (
                    raw_output,
                    LLMUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        latency_ms=latency_ms,
                        audit_prompt_tokens=None,
                    ),
                )

        # Deterministic overrides before calling the internal mock (to influence content)
        resp_text = None
        try:
            m = re.search(r"Input:\s*(.*)", user)
            if m:
                input_line = m.group(1).strip()
                ws = input_line.split(":", 1)
                if len(ws) == 2:
                    task_input = ws[1].strip()
                    words = [w for w in task_input.split() if w]
                    if words:
                        resp_text = " ".join(sorted(words))
            # If WordSorting path not triggered, attempt MGSM deterministic path from Problem text
            if resp_text is None and "Problem:" in user:
                nums = re.findall(r"-?\d+", user)
                if nums:
                    total = sum(int(n) for n in nums)
                    resp_text = f"Answer: {total}"
            if resp_text is None:
                m2 = re.search(
                    r"gold_answer\s*[:=]\s*(\-?\d+(?:\.\d+)?)", user, re.IGNORECASE
                )
                if m2:
                    resp_text = m2.group(1)
        except Exception:
            resp_text = None

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        if resp_text:
            resp.choices[0].message.content = resp_text

        latency_ms = (time.perf_counter() - start) * 1000.0 + 100.0  # Add fake delay

        choice = resp.choices[0]
        text = choice.message.content or ""

        usage = getattr(resp, "usage", None)
        return (
            text,
            LLMUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                latency_ms=latency_ms,
                audit_prompt_tokens=None,
            ),
        )

    def _maybe_build_game24_moves(self, user: str) -> str | None:
        num_branches = 10
        m_n = re.search(r"Generate exactly\s+(\d+)", user)
        if m_n:
            try:
                num_branches = int(m_n.group(1))
            except Exception:
                num_branches = 10

        items_json: str | None = None
        mode: str | None = None
        if "<CurrentItems>" in user and "</CurrentItems>" in user:
            mode = "generate"
            m_items = re.search(
                r"<CurrentItems>\s*(\[.*?\])\s*</CurrentItems>", user, re.S
            )
            if m_items:
                items_json = m_items.group(1).strip()
        elif "Previous items:" in user and "Last move" in user:
            mode = "improve"
            m_items = re.search(r"Previous items:\s*(\[.*?\])\s*Last move", user, re.S)
            if m_items:
                items_json = m_items.group(1).strip()

        if items_json is None or mode is None:
            return None

        try:
            items_obj = json.loads(items_json)
        except Exception:
            return None
        if not isinstance(items_obj, list) or not items_obj:
            return None

        items: list[dict[str, Any]] = []
        for item in items_obj:
            if not isinstance(item, dict):
                continue
            if "id" not in item or "value" not in item:
                continue
            try:
                item_id = int(item["id"])
                value = float(item["value"])
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            items.append({"id": item_id, "value": value})
        if not items:
            return None

        last_move: dict[str, Any] | None = None
        if mode == "improve":
            m_last = re.search(r"Last move.*?\n(.*?)\n\s*\nOutput", user, re.S)
            if m_last:
                try:
                    last_move = json.loads(m_last.group(1).strip())
                except Exception:
                    last_move = None

        ops = ["+", "-", "*", "/"]
        scored_moves: list[tuple[float, int, int, int, str]] = []
        seen: set[tuple[int, int, str]] = set()
        for a_item in items:
            for b_item in items:
                id1 = int(a_item["id"])
                id2 = int(b_item["id"])
                if id1 == id2:
                    continue
                a = float(a_item["value"])
                b = float(b_item["value"])
                for op in ops:
                    if op in {"+", "*"} and id2 < id1:
                        continue
                    if op == "/" and abs(b) < 1e-12:
                        continue
                    if op == "+":
                        result = a + b
                    elif op == "-":
                        result = a - b
                    elif op == "*":
                        result = a * b
                    else:
                        result = a / b

                    if not math.isfinite(result):
                        continue
                    if abs(result) > 1e6:
                        continue

                    key = (id1, id2, op)
                    if key in seen:
                        continue
                    seen.add(key)

                    if last_move is not None:
                        pick = last_move.get("pick")
                        if (
                            last_move.get("op") == op
                            and isinstance(pick, list)
                            and len(pick) == 2
                            and int(pick[0]) == id1
                            and int(pick[1]) == id2
                        ):
                            continue

                    score = abs(result - 24.0)
                    op_rank = ops.index(op)
                    scored_moves.append((score, op_rank, id1, id2, op))

        if not scored_moves:
            return None
        scored_moves.sort()

        if mode == "improve":
            _, _, id1, id2, op = scored_moves[0]
            return json.dumps({"pick": [id1, id2], "op": op}, separators=(",", ":"))

        lines: list[str] = []
        for _, _, id1, id2, op in scored_moves[: max(1, num_branches)]:
            lines.append(
                json.dumps({"pick": [id1, id2], "op": op}, separators=(",", ":"))
            )
        return "\n".join(lines)
