"""Deterministic offline providers implementing production service protocols."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from app.generation.errors import (
    GenerationTimeoutError,
    LlmModelUnavailableError,
)
from app.generation.models import PreparedGenerationInput, ProviderIdentity
from app.retrieval.errors import EmbeddingModelUnavailableError
from app.retrieval.models import ModelIdentity


class OfflineEmbeddingProvider:
    """Small deterministic provider with no model or network access."""

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(
            name="aerodocs-evaluation-fake",
            revision="offline-v1",
            dimension=8,
            maximum_sequence_length=512,
        )

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray([_vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return np.asarray([_vector(text)], dtype=np.float32)


class OfflineLlmProvider:
    """Deterministic structured abstention provider for offline orchestration."""

    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            name="aerodocs-evaluation-fake",
            revision="offline-v1",
            device="cpu",
        )

    def prepare(
        self,
        _system_prompt: str,
        _user_prompt: str,
    ) -> PreparedGenerationInput:
        return PreparedGenerationInput(
            input_ids=object(),
            attention_mask=object(),
            prompt_token_count=1,
            model_context_limit=4_096,
            maximum_new_tokens=32,
            provider=self.identity(),
            response_prefix="",
            deadline_monotonic=float("inf"),
        )

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def generate(self, _prepared: PreparedGenerationInput) -> str:
        return '{"status":"abstain","reason_code":"insufficient_evidence"}'


class MalformedOfflineLlmProvider(OfflineLlmProvider):
    """Reach production structured-output validation with malformed output."""

    def generate(self, _prepared: PreparedGenerationInput) -> str:
        return "{malformed"


class UnavailableOfflineLlmProvider(OfflineLlmProvider):
    """Reach the production model-unavailable handling boundary."""

    def prepare(
        self,
        _system_prompt: str,
        _user_prompt: str,
    ) -> PreparedGenerationInput:
        raise LlmModelUnavailableError


class TimeoutOfflineLlmProvider(OfflineLlmProvider):
    """Reach the production generation-timeout handling boundary immediately."""

    def generate(self, _prepared: PreparedGenerationInput) -> str:
        raise GenerationTimeoutError("generation")


class FailingOfflineEmbeddingProvider(OfflineEmbeddingProvider):
    """Reach the production retrieval provider-failure boundary."""

    def embed_query(self, _text: str) -> NDArray[np.float32]:
        raise EmbeddingModelUnavailableError


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [float(value + 1) for value in digest[:8]]
