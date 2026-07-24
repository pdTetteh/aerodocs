"""Lazy local embedding providers with strict output validation."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from threading import Lock
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from app.core.config import Settings
from app.retrieval.errors import (
    EmbeddingDimensionError,
    EmbeddingModelUnavailableError,
    EmbeddingShapeError,
    NonFiniteEmbeddingError,
)
from app.retrieval.models import ModelIdentity

FloatMatrix = NDArray[np.float32]


class EmbeddingProvider(Protocol):
    def model_identity(self) -> ModelIdentity: ...

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix: ...

    def embed_query(self, text: str) -> FloatMatrix: ...


class _TorchModule(Protocol):
    def set_num_threads(self, threads: int) -> None: ...

    def set_num_interop_threads(self, threads: int) -> None: ...

    def init_num_threads(self) -> None: ...


class _ProcessTorchRuntime:
    """Coordinate PyTorch's process-global CPU runtime behind one lock."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._thread_count: int | None = None
        self._interop_attempted = False
        self._initialization_failed = False

    def configure(self, torch_module: _TorchModule, thread_count: int) -> None:
        """Configure process-global CPU threading once and reject conflicts."""
        with self._lock:
            if self._initialization_failed:
                raise EmbeddingModelUnavailableError
            if self._thread_count is not None:
                if self._thread_count != thread_count:
                    raise EmbeddingModelUnavailableError
                return
            try:
                torch_module.set_num_threads(thread_count)
                self._thread_count = thread_count
                if not self._interop_attempted:
                    self._interop_attempted = True
                    torch_module.set_num_interop_threads(1)
            except Exception as error:
                self._initialization_failed = True
                raise EmbeddingModelUnavailableError from error


# PyTorch CPU thread pools are process-global. This private coordinator is the
# only mutable process-level state and protects every transition with its lock.
_PROCESS_TORCH_RUNTIME = _ProcessTorchRuntime()


class SentenceTransformerEmbeddingProvider:
    """Load one SentenceTransformer lazily for local inference."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: object | None = None
        self._model_lock = Lock()
        self._inference_lock = Lock()

    def _torch(self) -> _TorchModule:
        try:
            torch_module = cast(_TorchModule, import_module("torch"))
            _PROCESS_TORCH_RUNTIME.configure(
                torch_module,
                self.settings.embedding_torch_num_threads,
            )
        except EmbeddingModelUnavailableError:
            raise
        except Exception as error:
            raise EmbeddingModelUnavailableError from error
        return torch_module

    def _load(self, torch_module: _TorchModule | None = None) -> object:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                if torch_module is None:
                    torch_module = self._torch()
                else:
                    _PROCESS_TORCH_RUNTIME.configure(
                        torch_module,
                        self.settings.embedding_torch_num_threads,
                    )
                sentence_transformers = import_module("sentence_transformers")
                constructor = sentence_transformers.SentenceTransformer

                self._model = constructor(
                    self.settings.embedding_model_name,
                    revision=self.settings.embedding_model_revision,
                    device=self.settings.embedding_device,
                )
            except Exception as error:
                raise EmbeddingModelUnavailableError from error
        return self._model

    def model_identity(self) -> ModelIdentity:
        model = self._load()
        try:
            dimension_getter = getattr(model, "get_embedding_dimension", None)
            if dimension_getter is None:
                dimension_getter = model.get_sentence_embedding_dimension  # type: ignore[attr-defined]
            dimension = int(dimension_getter())
            maximum = int(model.max_seq_length)  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError) as error:
            raise EmbeddingShapeError from error
        return ModelIdentity(
            name=self.settings.embedding_model_name,
            revision=self.settings.embedding_model_revision,
            dimension=dimension,
            maximum_sequence_length=maximum,
        )

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            raise EmbeddingShapeError
        return self._embed(list(texts))

    def embed_query(self, text: str) -> FloatMatrix:
        return self._embed([text])

    def _embed(self, texts: list[str]) -> FloatMatrix:
        with self._inference_lock:
            torch_module = self._torch()
            try:
                torch_module.init_num_threads()
            except Exception as error:
                raise EmbeddingModelUnavailableError from error
            model = self._load(torch_module)
            dimension = self.model_identity().dimension
            try:
                raw = model.encode(  # type: ignore[attr-defined]
                    texts,
                    batch_size=self.settings.embedding_batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            except Exception as error:
                raise EmbeddingModelUnavailableError from error
            return validate_matrix(raw, rows=len(texts), dimension=dimension)


def validate_matrix(raw: object, *, rows: int, dimension: int) -> FloatMatrix:
    try:
        matrix = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise EmbeddingShapeError from error
    if matrix.ndim != 2 or matrix.shape[0] != rows:
        raise EmbeddingShapeError
    if matrix.shape[1] != dimension:
        raise EmbeddingDimensionError
    if not np.isfinite(matrix).all():
        raise NonFiniteEmbeddingError
    return cast(FloatMatrix, np.ascontiguousarray(matrix))


def normalize_matrix(matrix: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or bool(np.any(norms == 0)):
        raise NonFiniteEmbeddingError
    return cast(FloatMatrix, np.ascontiguousarray(matrix / norms, dtype=np.float32))
