from __future__ import annotations

from typing import Any

import numpy as np

from graph_bot.adapters.embeddings import (
    DeterministicHashEmbeddingProvider,
    EmbeddingProvider,
    HybridEmbeddingProvider,
    SentenceTransformerProvider,
)


class _FailingProvider:
    def encode(self, texts: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
        raise RuntimeError("primary failed")


class _StaticProvider:
    def __init__(self, value: float = 1.0) -> None:
        self._value = value

    def encode(self, texts: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
        vectors = np.full((len(texts), 4), self._value, dtype=np.float32)
        return (
            vectors,
            {
                "provider": "local",
                "model": "static-test",
                "latency_ms": 0.1,
                "cost_usd": 0.0,
                "usage_tokens": None,
            },
        )


def test_deterministic_hash_provider_returns_stable_vectors_and_meta() -> None:
    provider = DeterministicHashEmbeddingProvider(dim=8)

    vectors_1, meta_1 = provider.encode(["alpha", "beta", "alpha"])
    vectors_2, meta_2 = provider.encode(["alpha", "beta", "alpha"])

    assert vectors_1.shape == (3, 8)
    assert vectors_1.dtype == np.float32
    assert np.allclose(vectors_1, vectors_2)
    assert np.allclose(vectors_1[0], vectors_1[2])
    assert not np.allclose(vectors_1[0], vectors_1[1])

    for meta in (meta_1, meta_2):
        assert meta["provider"] == "deterministic"
        assert meta["model"] == "deterministic-hash-8"
        assert isinstance(meta["latency_ms"], float)
        assert meta["latency_ms"] >= 0.0
        assert meta["cost_usd"] == 0.0
        assert meta["usage_tokens"] is None


def test_deterministic_hash_provider_batch_shape_is_stable() -> None:
    provider = DeterministicHashEmbeddingProvider(dim=16)

    vectors, _meta = provider.encode(["a", "b", "c", "d"])

    assert vectors.shape == (4, 16)


def test_hybrid_provider_falls_back_and_logs_warning(caplog: Any) -> None:
    hybrid: EmbeddingProvider = HybridEmbeddingProvider(
        primary=_FailingProvider(),
        fallback=_StaticProvider(value=7.0),
    )

    with caplog.at_level("WARNING"):
        vectors, meta = hybrid.encode(["hello"])

    assert vectors.shape == (1, 4)
    assert float(vectors[0][0]) == 7.0
    assert meta["provider"] == "local"
    assert "falling back" in caplog.text.lower()


def test_sentence_transformer_provider_is_lazy_initialized() -> None:
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    assert provider._model is None
