"""Offline tests for lazy local generation-provider lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from time import sleep
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.factory import create_app
from app.generation import provider as provider_module
from app.generation.errors import (
    GenerationTimeoutError,
    LlmModelUnavailableError,
)
from app.generation.models import PreparedGenerationInput, ProviderIdentity
from app.generation.provider import TransformersLlmProvider
from app.retrieval.models import ModelIdentity

TEST_REVISION = "a" * 40


class MinimalEmbeddingProvider:
    def model_identity(self) -> ModelIdentity:
        raise AssertionError("health and metadata must not load embeddings")

    def embed_documents(self, _texts: object) -> np.ndarray:
        raise AssertionError("health must not embed")

    def embed_query(self, _text: str) -> np.ndarray:
        raise AssertionError("health must not embed")


class TrackingLlm:
    def __init__(self) -> None:
        self.calls = 0

    def identity(self) -> ProviderIdentity:
        self.calls += 1
        return ProviderIdentity(name="fake", revision="test", device="cpu")

    def prepare(self, _system: str, _user: str) -> PreparedGenerationInput:
        self.calls += 1
        raise AssertionError

    def count_tokens(self, _text: str) -> int:
        self.calls += 1
        return 1

    def generate(self, _prepared: PreparedGenerationInput) -> str:
        self.calls += 1
        return "{}"


class FakeTensor:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values
        self.shape = (len(values), len(values[0]))

    def to(self, _device: str) -> "FakeTensor":
        return self

    def __getitem__(self, index: int) -> list[int]:
        return self.values[index]


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = None
    model_max_length = 128

    def __init__(self, revision: str) -> None:
        self.init_kwargs = {"_commit_hash": revision}
        self.messages: list[dict[str, str]] = []
        self.template_kwargs: dict[str, object] = {}

    def __call__(self, text: str, **_kwargs: object) -> object:
        return {"input_ids": text.split()}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> dict[str, FakeTensor]:
        self.messages = messages
        self.template_kwargs = kwargs
        return {
            "input_ids": FakeTensor([[1, 2, 3, 4]]),
            "attention_mask": FakeTensor([[1, 1, 1, 1]]),
        }

    def decode(self, tokens: list[int], **_kwargs: object) -> str:
        assert tokens == [9, 10]
        return '"status":"abstain","reason_code":"insufficient_evidence"}'


class FakeModel:
    def __init__(self, events: list[str], revision: str) -> None:
        self.events = events
        self.config = SimpleNamespace(
            _commit_hash=revision,
            max_position_embeddings=256,
        )
        self.slow = False
        self.slow_delay = 0.02
        self.generate_calls = 0
        self.entered = Event()

    def to(self, _device: str) -> None:
        self.events.append("model_to")

    def eval(self) -> None:
        self.events.append("model_eval")

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generate_calls += 1
        self.entered.set()
        self.events.append("generate")
        if self.slow:
            sleep(self.slow_delay)
        stopping = kwargs["stopping_criteria"]
        stopping[0](None, None)  # type: ignore[index]
        return FakeTensor([[1, 2, 3, 4, 9, 10]])


def _settings(default_settings: Settings, tmp_path: Path) -> Settings:
    return default_settings.model_copy(
        update={
            "document_storage_root": tmp_path / "documents",
            "index_runtime_root": tmp_path / "indexes",
            "generation_audit_root": tmp_path / "audits",
            "generation_model_name": "fake/model",
            "generation_model_revision": TEST_REVISION,
        }
    )


@pytest.fixture
def provider_environment(
    default_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TransformersLlmProvider, FakeTokenizer, FakeModel, list[str]]:
    events: list[str] = []
    tokenizer = FakeTokenizer(TEST_REVISION)
    model = FakeModel(events, TEST_REVISION)

    def load_tokenizer(*_args: object, **kwargs: object) -> FakeTokenizer:
        assert kwargs["revision"] == TEST_REVISION
        events.append("tokenizer_load")
        return tokenizer

    def load_model(*_args: object, **kwargs: object) -> FakeModel:
        assert kwargs["revision"] == TEST_REVISION
        events.append("model_load")
        return model

    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        StoppingCriteriaList=list,
    )
    fake_torch = SimpleNamespace(inference_mode=nullcontext)

    def configure(_settings: Settings) -> object:
        events.append("runtime_configure")
        return fake_torch

    monkeypatch.setattr(provider_module, "import_module", lambda _name: transformers)
    monkeypatch.setattr(
        provider_module,
        "configure_torch_runtime",
        configure,
    )
    monkeypatch.setattr(
        provider_module,
        "initialize_torch_worker",
        lambda _torch: events.append("worker_initialize"),
    )
    return (
        TransformersLlmProvider(_settings(default_settings, tmp_path)),
        tokenizer,
        model,
        events,
    )


def test_health_and_index_metadata_do_not_touch_generation_provider(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    llm = TrackingLlm()
    with TestClient(
        create_app(
            _settings(default_settings, tmp_path),
            embedding_provider=MinimalEmbeddingProvider(),
            llm_provider=llm,
        )
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/indexes/active").status_code == 404
        assert client.get("/documents/missing").status_code == 404

    assert llm.calls == 0


def test_prepare_uses_exact_chat_template_and_generate_excludes_prompt(
    provider_environment: tuple[
        TransformersLlmProvider,
        FakeTokenizer,
        FakeModel,
        list[str],
    ],
) -> None:
    provider, tokenizer, _model, events = provider_environment

    assert provider.identity().name == "fake/model"
    assert events == []
    prepared = provider.prepare("system", "user")
    output = provider.generate(prepared)

    assert prepared.prompt_token_count == 4
    assert prepared.model_context_limit == 128
    assert tokenizer.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "{"},
    ]
    assert tokenizer.template_kwargs == {
        "tokenize": True,
        "continue_final_message": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    assert output.startswith('{"status":"abstain"')
    assert events.index("runtime_configure") < events.index("model_load")
    assert events.index("worker_initialize") < events.index("generate")


def test_revision_mismatch_is_rejected(
    default_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer("wrong")
    model = FakeModel([], TEST_REVISION)
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: tokenizer
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: model
        ),
    )
    monkeypatch.setattr(provider_module, "import_module", lambda _name: transformers)
    monkeypatch.setattr(
        provider_module,
        "configure_torch_runtime",
        lambda _settings: object(),
    )

    with pytest.raises(LlmModelUnavailableError):
        TransformersLlmProvider(_settings(default_settings, tmp_path)).prepare(
            "system",
            "user",
        )


def test_timeout_discards_output_and_releases_provider_lock(
    default_settings: Settings,
    tmp_path: Path,
    provider_environment: tuple[
        TransformersLlmProvider,
        FakeTokenizer,
        FakeModel,
        list[str],
    ],
) -> None:
    provider, _tokenizer, model, _events = provider_environment
    provider.settings = _settings(default_settings, tmp_path).model_copy(
        update={"generation_timeout_seconds": 0.001}
    )
    prepared = provider.prepare("system", "user")
    model.slow = True
    model.slow_delay = 0.06

    with pytest.raises(GenerationTimeoutError):
        provider.generate(prepared)

    model.slow = False
    assert provider.generate(provider.prepare("system", "later")).startswith(
        '{"status":"abstain"'
    )


def test_queued_generations_share_total_deadline_without_deadlock(
    default_settings: Settings,
    tmp_path: Path,
    provider_environment: tuple[
        TransformersLlmProvider,
        FakeTokenizer,
        FakeModel,
        list[str],
    ],
) -> None:
    provider, _tokenizer, model, _events = provider_environment
    provider.settings = _settings(default_settings, tmp_path).model_copy(
        update={"generation_timeout_seconds": 0.03}
    )
    model.slow = True
    model.slow_delay = 0.06
    prepared = [provider.prepare("system", f"user-{index}") for index in range(4)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(provider.generate, item) for item in prepared]
        assert model.entered.wait(timeout=1)
        errors = [future.exception(timeout=1) for future in futures]

    assert model.generate_calls == 1
    assert all(isinstance(error, GenerationTimeoutError) for error in errors)
    assert (
        sum(
            error.stage == "lock_wait"
            for error in errors
            if isinstance(error, GenerationTimeoutError)
        )
        >= 3
    )

    model.slow = False
    assert provider.generate(provider.prepare("system", "later")).startswith(
        '{"status":"abstain"'
    )
