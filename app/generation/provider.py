"""Lazy local language-model provider with exact prompt preparation."""

from __future__ import annotations

import time
from importlib import import_module
from threading import Lock
from typing import Protocol

from app.core.config import Settings
from app.generation.errors import (
    GenerationFailedError,
    GenerationTimeoutError,
    LlmModelUnavailableError,
)
from app.generation.models import PreparedGenerationInput, ProviderIdentity
from app.retrieval.embeddings import (
    configure_torch_runtime,
    initialize_torch_worker,
)
from app.retrieval.errors import EmbeddingModelUnavailableError


class LlmProvider(Protocol):
    """Small interface required by grounded answer generation."""

    def identity(self) -> ProviderIdentity: ...

    def prepare(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> PreparedGenerationInput: ...

    def count_tokens(self, text: str) -> int: ...

    def generate(self, prepared: PreparedGenerationInput) -> str: ...


class _DeadlineStoppingCriteria:
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.reached = False

    def __call__(self, *_args: object, **_kwargs: object) -> bool:
        self.reached = time.monotonic() >= self.deadline
        return self.reached


class TransformersLlmProvider:
    """Load one pinned Hugging Face model lazily for serialized CPU inference."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tokenizer: object | None = None
        self._model: object | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def identity(self) -> ProviderIdentity:
        """Return configured identity without loading model artifacts."""
        return ProviderIdentity(
            name=self.settings.generation_model_name,
            revision=self.settings.generation_model_revision,
            device=self.settings.generation_device,
        )

    def _load(self) -> tuple[object, object]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        with self._load_lock:
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model
            try:
                configure_torch_runtime(self.settings)
                transformers = import_module("transformers")
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.settings.generation_model_name,
                    revision=self.settings.generation_model_revision,
                )
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.settings.generation_model_name,
                    revision=self.settings.generation_model_revision,
                )
                self._validate_revision(tokenizer, model)
                model.to(self.settings.generation_device)
                model.eval()
            except Exception as error:
                raise LlmModelUnavailableError from error
            self._tokenizer = tokenizer
            self._model = model
            return tokenizer, model

    def prepare(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> PreparedGenerationInput:
        """Render and tokenize the exact chat prompt passed to generation."""
        deadline = time.monotonic() + self.settings.generation_timeout_seconds
        tokenizer, model = self._load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "{"},
        ]
        try:
            encoded = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=True,
                continue_final_message=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = encoded["input_ids"].to(self.settings.generation_device)
            attention_mask = encoded["attention_mask"].to(
                self.settings.generation_device
            )
            prompt_tokens = int(input_ids.shape[1])
            context_limit = self._context_limit(tokenizer, model)
        except Exception as error:
            raise GenerationFailedError from error
        return PreparedGenerationInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_token_count=prompt_tokens,
            model_context_limit=context_limit,
            maximum_new_tokens=self.settings.maximum_generation_output_tokens,
            provider=self.identity(),
            response_prefix="{",
            deadline_monotonic=deadline,
        )

    def count_tokens(self, text: str) -> int:
        """Count generated text tokens without chat-template additions."""
        tokenizer, _model = self._load()
        try:
            encoded = tokenizer(text, add_special_tokens=False)  # type: ignore[operator]
            return len(encoded["input_ids"])
        except Exception as error:
            raise GenerationFailedError from error

    def generate(self, prepared: PreparedGenerationInput) -> str:
        """Generate one deterministic response and discard timed-out output."""
        if prepared.provider != self.identity():
            raise GenerationFailedError
        remaining = prepared.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise GenerationTimeoutError("preparation")
        acquired = self._inference_lock.acquire(timeout=remaining)
        if not acquired:
            raise GenerationTimeoutError("lock_wait")
        try:
            remaining = prepared.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise GenerationTimeoutError("lock_wait")
            try:
                torch_module = configure_torch_runtime(self.settings)
                initialize_torch_worker(torch_module)
            except EmbeddingModelUnavailableError as error:
                raise LlmModelUnavailableError from error
            tokenizer, model = self._load()
            transformers = import_module("transformers")
            deadline = prepared.deadline_monotonic
            criterion = _DeadlineStoppingCriteria(deadline)
            stopping = transformers.StoppingCriteriaList([criterion])
            try:
                with torch_module.inference_mode():  # type: ignore[attr-defined]
                    output = model.generate(  # type: ignore[attr-defined]
                        input_ids=prepared.input_ids,
                        attention_mask=prepared.attention_mask,
                        do_sample=False,
                        max_new_tokens=prepared.maximum_new_tokens,
                        stopping_criteria=stopping,
                        eos_token_id=tokenizer.eos_token_id,  # type: ignore[attr-defined]
                        pad_token_id=(
                            tokenizer.pad_token_id  # type: ignore[attr-defined]
                            or tokenizer.eos_token_id  # type: ignore[attr-defined]
                        ),
                    )
                if criterion.reached or time.monotonic() >= deadline:
                    raise GenerationTimeoutError
                generated = output[0][prepared.prompt_token_count :]
                remainder = str(
                    tokenizer.decode(  # type: ignore[attr-defined]
                        generated,
                        skip_special_tokens=True,
                    )
                )
                return prepared.response_prefix + remainder
            except GenerationTimeoutError:
                raise
            except Exception as error:
                raise GenerationFailedError from error
        finally:
            self._inference_lock.release()

    def _validate_revision(self, tokenizer: object, model: object) -> None:
        expected = self.settings.generation_model_revision
        tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
        resolved = {
            value
            for value in (tokenizer_revision, model_revision)
            if isinstance(value, str)
        }
        if resolved and (resolved != {expected}):
            raise LlmModelUnavailableError

    @staticmethod
    def _context_limit(tokenizer: object, model: object) -> int:
        values: list[int] = []
        for value in (
            getattr(tokenizer, "model_max_length", None),
            getattr(getattr(model, "config", None), "max_position_embeddings", None),
        ):
            if isinstance(value, int) and 0 < value < 1_000_000_000:
                values.append(value)
        if not values:
            raise GenerationFailedError
        return min(values)
