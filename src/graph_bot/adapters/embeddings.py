from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Optional, Protocol

import numpy as np

from ..logsetting import logger
from ..settings import settings
from ..utils.pricing import load_pricing_table


EmbeddingMeta = dict[str, str | float | Optional[int]]


class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str]) -> tuple[np.ndarray, EmbeddingMeta]: ...


class DeterministicHashEmbeddingProvider:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def encode(self, texts: list[str]) -> tuple[np.ndarray, EmbeddingMeta]:
        start = time.perf_counter()
        if not texts:
            vectors = np.zeros((0, self._dim), dtype=np.float32)
        else:
            vectors = np.vstack([self._embed_text(text) for text in texts]).astype(
                np.float32
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return (
            vectors,
            {
                "provider": "deterministic",
                "model": f"deterministic-hash-{self._dim}",
                "latency_ms": latency_ms,
                "cost_usd": 0.0,
                "usage_tokens": None,
            },
        )

    def _embed_text(self, text: str) -> np.ndarray:
        values: list[float] = []
        salt = 0
        while len(values) < self._dim:
            digest = hashlib.sha256(f"{text}|{salt}".encode("utf-8")).digest()
            for idx in range(0, len(digest), 4):
                chunk = digest[idx : idx + 4]
                if len(chunk) < 4:
                    continue
                raw = int.from_bytes(chunk, byteorder="big", signed=False)
                mapped = (raw / 4294967295.0) * 2.0 - 1.0
                values.append(mapped)
                if len(values) >= self._dim:
                    break
            salt += 1

        vector = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        return vector


class SentenceTransformerProvider:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def encode(self, texts: list[str]) -> tuple[np.ndarray, EmbeddingMeta]:
        start = time.perf_counter()
        model = self._get_model()
        vectors = np.asarray(model.encode(texts), dtype=np.float32)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return (
            vectors,
            {
                "provider": "local",
                "model": self._model_name,
                "latency_ms": latency_ms,
                "cost_usd": 0.0,
                "usage_tokens": None,
            },
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerProvider"
            ) from exc

        self._model = SentenceTransformer(self._model_name)
        return self._model


class OpenAIEmbeddingProvider:
    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self._model_name = model_name
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._client: Any | None = None
        self._pricing_table = load_pricing_table(settings.pricing_path)

    def is_configured(self) -> bool:
        return bool(self._api_key and self._api_key != "EMPTY")

    def encode(self, texts: list[str]) -> tuple[np.ndarray, EmbeddingMeta]:
        if not self.is_configured():
            raise RuntimeError("OpenAI embedding key is missing")

        start = time.perf_counter()
        client = self._get_client()
        response = client.embeddings.create(model=self._model_name, input=texts)
        latency_ms = (time.perf_counter() - start) * 1000.0

        vectors = np.asarray(
            [item.embedding for item in response.data], dtype=np.float32
        )
        usage = getattr(response, "usage", None)
        usage_tokens = getattr(usage, "total_tokens", None)
        if usage_tokens is None:
            usage_tokens = getattr(usage, "prompt_tokens", None)
        usage_tokens_int = int(usage_tokens) if usage_tokens is not None else None
        cost_usd = self._calculate_cost(usage_tokens_int or 0)

        return (
            vectors,
            {
                "provider": "openai",
                "model": self._model_name,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "usage_tokens": usage_tokens_int,
            },
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIEmbeddingProvider"
            ) from exc

        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _calculate_cost(self, usage_tokens: int) -> float:
        model_config = self._pricing_table.get("models", {}).get(self._model_name, {})
        input_usd_per_1k = float(model_config.get("input_usd_per_1k", 0.0))
        return (usage_tokens / 1000.0) * input_usd_per_1k


class HybridEmbeddingProvider:
    def __init__(self, primary: EmbeddingProvider, fallback: EmbeddingProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    def encode(self, texts: list[str]) -> tuple[np.ndarray, EmbeddingMeta]:
        if self._primary_unavailable():
            logger.warning(
                "Embedding primary unavailable; falling back to local provider"
            )
            logging.getLogger().warning(
                "Embedding primary unavailable; falling back to local provider"
            )
            return self._fallback.encode(texts)

        try:
            return self._primary.encode(texts)
        except Exception as exc:
            logger.warning(
                f"Embedding primary failed; falling back to local provider: {exc}"
            )
            logging.getLogger().warning(
                f"Embedding primary failed; falling back to local provider: {exc}"
            )
            return self._fallback.encode(texts)

    def _primary_unavailable(self) -> bool:
        checker = getattr(self._primary, "is_configured", None)
        if callable(checker):
            try:
                return not bool(checker())
            except Exception:
                return True
        return False
