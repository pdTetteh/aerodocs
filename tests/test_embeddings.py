"""Offline tests for the real provider's lazy PyTorch integration."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.factory import create_app
from app.retrieval import embeddings
from app.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from app.retrieval.errors import EmbeddingModelUnavailableError


class FakeTorch:
    def __init__(self) -> None:
        self.set_threads: list[int] = []
        self.set_interop_threads: list[int] = []
        self.worker_initializations = 0
        self.events: list[str] = []

    def set_num_threads(self, threads: int) -> None:
        self.events.append("set_num_threads")
        self.set_threads.append(threads)

    def set_num_interop_threads(self, threads: int) -> None:
        self.events.append("set_num_interop_threads")
        self.set_interop_threads.append(threads)

    def init_num_threads(self) -> None:
        self.events.append("init_num_threads")
        self.worker_initializations += 1


class FakeModel:
    active = 0
    maximum_active = 0
    activity_lock = Lock()

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.max_seq_length = 256
        revision = str(_kwargs.get("resolved_revision", "test"))
        self.tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": revision})
        self.auto_model = SimpleNamespace(config=SimpleNamespace(_commit_hash=revision))

    def __getitem__(self, index: int) -> "FakeModel":
        if index != 0:
            raise IndexError(index)
        return self

    def get_embedding_dimension(self) -> int:
        return 3

    def get_sentence_embedding_dimension(self) -> int:
        raise AssertionError("deprecated dimension API must not be called")

    def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        with self.activity_lock:
            type(self).active += 1
            type(self).maximum_active = max(
                type(self).maximum_active,
                type(self).active,
            )
        try:
            sleep(0.03)
            return np.asarray([[1.0, 2.0, 3.0] for _text in texts], dtype=np.float32)
        finally:
            with self.activity_lock:
                type(self).active -= 1


@pytest.fixture
def provider_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeTorch, list[str]]:
    """Install fake local modules and an isolated process-runtime coordinator."""
    fake_torch = FakeTorch()
    construction_events: list[str] = []

    def constructor(*args: object, **kwargs: object) -> FakeModel:
        del args
        construction_events.append(str(kwargs["revision"]))
        fake_torch.events.append("model_constructed")
        return FakeModel(resolved_revision=kwargs["revision"])

    modules: dict[str, object] = {
        "torch": fake_torch,
        "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor),
    }
    monkeypatch.setattr(
        embeddings,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        embeddings,
        "_PROCESS_TORCH_RUNTIME",
        embeddings._ProcessTorchRuntime(),  # noqa: SLF001
    )
    FakeModel.active = 0
    FakeModel.maximum_active = 0
    return fake_torch, construction_events


def provider_settings(default_settings: Settings, tmp_path: Path) -> Settings:
    return default_settings.model_copy(
        update={
            "document_storage_root": tmp_path / "documents",
            "index_runtime_root": tmp_path / "indexes",
            "embedding_model_name": "fake/model",
            "embedding_model_revision": "test",
            "embedding_torch_num_threads": 2,
        }
    )


def test_process_runtime_initialization_is_idempotent_across_providers(
    default_settings: Settings,
    tmp_path: Path,
    provider_environment: tuple[FakeTorch, list[str]],
) -> None:
    fake_torch, construction_events = provider_environment
    settings = provider_settings(default_settings, tmp_path)
    first = SentenceTransformerEmbeddingProvider(settings)
    second = SentenceTransformerEmbeddingProvider(settings)

    first.embed_documents(["first"])
    second.embed_documents(["second"])

    assert fake_torch.set_threads == [2]
    assert fake_torch.set_interop_threads == [1]
    assert fake_torch.events.index("set_num_threads") < fake_torch.events.index(
        "model_constructed"
    )
    assert len(construction_events) == 2


def test_incompatible_process_thread_configuration_fails_safely(
    default_settings: Settings,
    tmp_path: Path,
    provider_environment: tuple[FakeTorch, list[str]],
) -> None:
    fake_torch, _construction_events = provider_environment
    settings = provider_settings(default_settings, tmp_path)
    SentenceTransformerEmbeddingProvider(settings).model_identity()
    incompatible = settings.model_copy(update={"embedding_torch_num_threads": 3})

    with pytest.raises(EmbeddingModelUnavailableError):
        SentenceTransformerEmbeddingProvider(incompatible).model_identity()

    assert fake_torch.set_threads == [2]
    assert fake_torch.set_interop_threads == [1]


def test_each_embedding_call_initializes_worker_and_inference_is_serialized(
    default_settings: Settings,
    tmp_path: Path,
    provider_environment: tuple[FakeTorch, list[str]],
) -> None:
    fake_torch, _construction_events = provider_environment
    provider = SentenceTransformerEmbeddingProvider(
        provider_settings(default_settings, tmp_path)
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        document = executor.submit(provider.embed_documents, ["document"])
        query = executor.submit(provider.embed_query, "query")
        assert document.result().shape == (1, 3)
        assert query.result().shape == (1, 3)

    assert fake_torch.worker_initializations == 2
    assert FakeModel.maximum_active == 1


def test_health_and_missing_index_do_not_initialize_pytorch_or_model(
    default_settings: Settings,
    tmp_path: Path,
    provider_environment: tuple[FakeTorch, list[str]],
) -> None:
    fake_torch, construction_events = provider_environment
    settings = provider_settings(default_settings, tmp_path)

    with TestClient(create_app(settings)) as first_client:
        assert first_client.get("/health").status_code == 200
        assert first_client.get("/indexes/active").status_code == 404
    with TestClient(create_app(settings)) as second_client:
        assert second_client.get("/health").status_code == 200

    assert fake_torch.events == []
    assert construction_events == []


def test_new_dimension_api_is_preferred(
    default_settings: Settings,
    tmp_path: Path,
    provider_environment: tuple[FakeTorch, list[str]],
) -> None:
    provider = SentenceTransformerEmbeddingProvider(
        provider_settings(default_settings, tmp_path)
    )

    assert provider.model_identity().dimension == 3


def test_exact_embedding_revision_is_passed_and_resolved_identity_is_verified(
    default_settings: Settings,
    tmp_path: Path,
    provider_environment: tuple[FakeTorch, list[str]],
) -> None:
    _fake_torch, construction_events = provider_environment
    settings = provider_settings(default_settings, tmp_path).model_copy(
        update={"embedding_model_revision": "a" * 40}
    )

    identity = SentenceTransformerEmbeddingProvider(settings).model_identity()

    assert construction_events == ["a" * 40]
    assert identity.revision == "a" * 40


def test_embedding_resolved_revision_mismatch_is_rejected(
    default_settings: Settings,
    tmp_path: Path,
    provider_environment: tuple[FakeTorch, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch, _construction_events = provider_environment
    modules: dict[str, object] = {
        "torch": fake_torch,
        "sentence_transformers": SimpleNamespace(
            SentenceTransformer=lambda *_args, **_kwargs: FakeModel(
                resolved_revision="b" * 40
            )
        ),
    }
    monkeypatch.setattr(embeddings, "import_module", lambda name: modules[name])
    settings = provider_settings(default_settings, tmp_path).model_copy(
        update={"embedding_model_revision": "a" * 40}
    )

    with pytest.raises(EmbeddingModelUnavailableError):
        SentenceTransformerEmbeddingProvider(settings).model_identity()


def test_deprecated_dimension_api_is_only_a_fallback(
    default_settings: Settings,
    tmp_path: Path,
    provider_environment: tuple[FakeTorch, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch, _construction_events = provider_environment

    class LegacyModel:
        max_seq_length = 128

        @staticmethod
        def get_sentence_embedding_dimension() -> int:
            return 7

    modules: dict[str, Any] = {
        "torch": fake_torch,
        "sentence_transformers": SimpleNamespace(
            SentenceTransformer=lambda *_args, **_kwargs: LegacyModel()
        ),
    }
    monkeypatch.setattr(embeddings, "import_module", lambda name: modules[name])
    provider = SentenceTransformerEmbeddingProvider(
        provider_settings(default_settings, tmp_path)
    )

    assert provider.model_identity().dimension == 7
