"""Strict public limits used by the read-only browser presentation layer."""

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from app.core.config import Settings
from app.ingestion.limits import MAXIMUM_SANITIZED_FILENAME_CHARACTERS
from app.ingestion.pdf import (
    MAXIMUM_DOCUMENT_AUTHOR_CHARACTERS,
    MAXIMUM_DOCUMENT_SUBJECT_CHARACTERS,
    MAXIMUM_DOCUMENT_TITLE_CHARACTERS,
)
from app.ui.errors import UIConfigurationError

PUBLIC_DOCUMENT_ID_CHARACTERS = 68
PUBLIC_CHUNK_ID_CHARACTERS = 68
PUBLIC_EVIDENCE_ID_CHARACTERS = 68
PUBLIC_CLAIM_ID_CHARACTERS = 32
PUBLIC_FILENAME_CHARACTERS = MAXIMUM_SANITIZED_FILENAME_CHARACTERS
PUBLIC_TITLE_CHARACTERS = MAXIMUM_DOCUMENT_TITLE_CHARACTERS
PUBLIC_AUTHOR_CHARACTERS = MAXIMUM_DOCUMENT_AUTHOR_CHARACTERS
PUBLIC_SUBJECT_CHARACTERS = MAXIMUM_DOCUMENT_SUBJECT_CHARACTERS
PUBLIC_VERSION_CHARACTERS = 128
PUBLIC_PROFILE_ID_CHARACTERS = 128
PUBLIC_ERROR_CODE_CHARACTERS = 128
PUBLIC_ERROR_MESSAGE_CHARACTERS = 1_024
ABSOLUTE_UI_RESPONSE_BYTES = 512 * 1024 * 1024
SERIALIZED_CHARACTER_BYTES = 6


class UiResponseBudgets(BaseModel):
    """Endpoint-specific bounded JSON budgets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    health: int
    documents: int
    metadata: int
    index: int
    search: int
    answer: int
    error: int

    @model_validator(mode="after")
    def validate_budgets(self) -> "UiResponseBudgets":
        values = (
            self.health,
            self.documents,
            self.metadata,
            self.index,
            self.search,
            self.answer,
            self.error,
        )
        if any(value <= 0 or value > ABSOLUTE_UI_RESPONSE_BYTES for value in values):
            raise ValueError("UI response budget exceeds the absolute safety ceiling")
        return self


class PublicUiConfig(BaseModel):
    """Frozen allowlist of non-sensitive browser limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_page_size: int
    document_page_maximum: int
    query_characters: int
    retrieval_top_k_default: int
    retrieval_top_k_maximum: int
    question_characters: int
    answer_top_k_default: int
    answer_top_k_maximum: int
    document_id_characters: int = PUBLIC_DOCUMENT_ID_CHARACTERS
    chunk_id_characters: int = PUBLIC_CHUNK_ID_CHARACTERS
    evidence_id_characters: int = PUBLIC_EVIDENCE_ID_CHARACTERS
    claim_id_characters: int = PUBLIC_CLAIM_ID_CHARACTERS
    filename_characters: int = PUBLIC_FILENAME_CHARACTERS
    title_characters: int = PUBLIC_TITLE_CHARACTERS
    subject_characters: int = PUBLIC_SUBJECT_CHARACTERS
    section_characters: int
    chunk_characters: int
    answer_evidence_count: int
    answer_evidence_characters: int
    claim_count: int
    claim_characters: int
    supports_per_claim: int
    quote_characters: int
    conflict_evidence_count: int
    search_available: bool
    answer_available: bool
    response_bytes: UiResponseBudgets

    @model_validator(mode="after")
    def validate_relationships(self) -> "PublicUiConfig":
        values = tuple(
            value
            for name, value in self
            if name != "response_bytes"
            and isinstance(value, int)
            and not isinstance(value, bool)
        )
        if any(value <= 0 for value in values):
            raise ValueError("public UI limits must be positive")
        if self.document_page_size > self.document_page_maximum:
            raise ValueError("default document page exceeds maximum")
        if self.retrieval_top_k_default > self.retrieval_top_k_maximum:
            raise ValueError("default retrieval top-k exceeds maximum")
        if self.answer_top_k_default > self.answer_top_k_maximum:
            raise ValueError("default answer top-k exceeds maximum")
        if self.answer_top_k_maximum > self.retrieval_top_k_maximum:
            raise ValueError("answer top-k exceeds retrieval maximum")
        return self


def build_public_ui_config(settings: Settings) -> PublicUiConfig:
    """Derive browser limits and byte budgets from authoritative settings."""
    document_entry = SERIALIZED_CHARACTER_BYTES * (
        PUBLIC_DOCUMENT_ID_CHARACTERS
        + PUBLIC_FILENAME_CHARACTERS
        + PUBLIC_TITLE_CHARACTERS
        + PUBLIC_PROFILE_ID_CHARACTERS
        + 256
    )
    search_entry = SERIALIZED_CHARACTER_BYTES * (
        settings.chunk_max_characters
        + settings.chunk_max_characters
        + PUBLIC_DOCUMENT_ID_CHARACTERS
        + PUBLIC_CHUNK_ID_CHARACTERS
        + PUBLIC_FILENAME_CHARACTERS
        + PUBLIC_TITLE_CHARACTERS
        + 512
    )
    answer_evidence_entry = SERIALIZED_CHARACTER_BYTES * (
        settings.maximum_answer_evidence_characters
        + settings.chunk_max_characters
        + PUBLIC_DOCUMENT_ID_CHARACTERS
        + PUBLIC_CHUNK_ID_CHARACTERS
        + PUBLIC_EVIDENCE_ID_CHARACTERS
        + PUBLIC_FILENAME_CHARACTERS
        + PUBLIC_TITLE_CHARACTERS
        + 512
    )
    answer_claims = settings.maximum_generation_claims * (
        SERIALIZED_CHARACTER_BYTES
        * (
            settings.maximum_generation_claim_characters
            + PUBLIC_CLAIM_ID_CHARACTERS
            + settings.maximum_generation_supports_per_claim
            * (
                settings.maximum_generation_quote_characters
                + PUBLIC_EVIDENCE_ID_CHARACTERS
                + PUBLIC_CLAIM_ID_CHARACTERS
                + 128
            )
        )
    )
    answer_citations = (
        settings.maximum_generation_claims
        * settings.maximum_generation_supports_per_claim
        * SERIALIZED_CHARACTER_BYTES
        * (
            PUBLIC_EVIDENCE_ID_CHARACTERS
            + PUBLIC_DOCUMENT_ID_CHARACTERS
            + PUBLIC_CHUNK_ID_CHARACTERS
            + PUBLIC_FILENAME_CHARACTERS
            + PUBLIC_TITLE_CHARACTERS
            + settings.chunk_max_characters
            + 512
        )
    )
    search_budget = (
        16 * 1024
        + SERIALIZED_CHARACTER_BYTES * settings.maximum_query_characters
        + settings.retrieval_maximum_top_k * search_entry
    )
    answer_budget = (
        64 * 1024
        + SERIALIZED_CHARACTER_BYTES
        * (
            settings.maximum_answer_question_characters
            + settings.maximum_generation_claims
            * settings.maximum_generation_claim_characters
            * 2
        )
        + settings.maximum_answer_context_chunks * answer_evidence_entry
        + answer_claims
        + answer_citations
    )
    try:
        budgets = UiResponseBudgets(
            health=4 * 1024,
            documents=(16 * 1024 + settings.ui_maximum_page_size * document_entry),
            metadata=32 * 1024
            + SERIALIZED_CHARACTER_BYTES
            * (
                PUBLIC_DOCUMENT_ID_CHARACTERS
                + PUBLIC_FILENAME_CHARACTERS
                + PUBLIC_TITLE_CHARACTERS
                + PUBLIC_SUBJECT_CHARACTERS
            ),
            # The compact projection contains no corpus-sized arrays.
            index=64 * 1024,
            search=min(search_budget, ABSOLUTE_UI_RESPONSE_BYTES),
            answer=min(answer_budget, ABSOLUTE_UI_RESPONSE_BYTES),
            error=64 * 1024,
        )
        return PublicUiConfig(
            document_page_size=settings.ui_default_page_size,
            document_page_maximum=settings.ui_maximum_page_size,
            query_characters=settings.maximum_query_characters,
            retrieval_top_k_default=settings.retrieval_default_top_k,
            retrieval_top_k_maximum=settings.retrieval_maximum_top_k,
            question_characters=settings.maximum_answer_question_characters,
            answer_top_k_default=settings.answer_default_top_k,
            answer_top_k_maximum=settings.answer_maximum_top_k,
            section_characters=settings.chunk_max_characters,
            chunk_characters=settings.chunk_max_characters,
            answer_evidence_count=settings.maximum_answer_context_chunks,
            answer_evidence_characters=settings.maximum_answer_evidence_characters,
            claim_count=settings.maximum_generation_claims,
            claim_characters=settings.maximum_generation_claim_characters,
            supports_per_claim=settings.maximum_generation_supports_per_claim,
            quote_characters=settings.maximum_generation_quote_characters,
            conflict_evidence_count=settings.maximum_generation_conflict_evidence_ids,
            search_available=search_budget <= ABSOLUTE_UI_RESPONSE_BYTES,
            answer_available=answer_budget <= ABSOLUTE_UI_RESPONSE_BYTES,
            response_bytes=budgets,
        )
    except (ValidationError, OverflowError) as error:
        raise UIConfigurationError from error
