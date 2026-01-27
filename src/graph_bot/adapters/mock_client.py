from __future__ import annotations

import time
import re
from dataclasses import dataclass
from typing import List, Any


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float


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
            r"Numbers:\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)", user_content
        )
        text = "1 + 1"

        if numbers_match:
            nums = numbers_match[0]
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
    """Mock LLM client that returns fixed answers for Game of 24, mimicking OpenAI structure."""

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

        # Call the internal mock structure (mimicking vllm_openai_client.py's implementation)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )

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
            ),
        )
