from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    audit_prompt_tokens: int | None = None


class VLLMOpenAIClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
        )
        self._model = model

    def chat(
        self, *, system: str, user: str, temperature: float = 0.0
    ) -> tuple[str, LLMUsage]:
        audit_prompt_tokens = None
        if tiktoken and self._model.startswith(("gpt", "o1")):
            try:
                encoding = tiktoken.encoding_for_model(self._model)
                audit_prompt_tokens = len(encoding.encode(system)) + len(
                    encoding.encode(user)
                )
            except Exception:
                pass

        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

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
                audit_prompt_tokens=audit_prompt_tokens,
            ),
        )
