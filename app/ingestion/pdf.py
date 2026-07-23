"""PDF validation, metadata normalization, and page extraction."""

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.ingestion.errors import (
    EncryptedPdfError,
    InvalidPdfError,
    PdfExpansionLimitError,
    ZeroPagePdfError,
)
from app.ingestion.models import ExtractionWarning, PageRecord


@dataclass(frozen=True)
class PdfExtraction:
    """Normalized results read from a PDF."""

    title: str | None
    author: str | None
    subject: str | None
    pages: list[PageRecord]


def extract_pdf(
    path: Path,
    document_id: str,
    *,
    maximum_pages: int,
    maximum_characters_per_page: int,
    maximum_characters_total: int,
) -> PdfExtraction:
    """Validate and extract a PDF without OCR."""
    try:
        document = pymupdf.open(path)  # type: ignore[no-untyped-call]
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise InvalidPdfError from error

    try:
        if document.needs_pass or document.is_encrypted:
            raise EncryptedPdfError
        if document.page_count <= 0:
            raise ZeroPagePdfError
        if document.page_count > maximum_pages:
            raise PdfExpansionLimitError

        metadata = document.metadata or {}
        pages: list[PageRecord] = []
        total_characters = 0
        for page_index in range(document.page_count):
            page = document.load_page(page_index)  # type: ignore[no-untyped-call]
            text = normalize_page_text(page.get_text("text"))
            character_count = len(text)
            if character_count > maximum_characters_per_page:
                raise PdfExpansionLimitError
            total_characters += character_count
            if total_characters > maximum_characters_total:
                raise PdfExpansionLimitError
            warnings: list[ExtractionWarning] = (
                ["no_extractable_text"] if not text else []
            )
            pages.append(
                PageRecord(
                    document_id=document_id,
                    page_number=page_index + 1,
                    text=text,
                    character_count=character_count,
                    warnings=warnings,
                )
            )

        return PdfExtraction(
            title=normalize_metadata_field(metadata.get("title"), 300),
            author=normalize_metadata_field(metadata.get("author"), 300),
            subject=normalize_metadata_field(metadata.get("subject"), 1_000),
            pages=pages,
        )
    except (EncryptedPdfError, PdfExpansionLimitError, ZeroPagePdfError):
        raise
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise InvalidPdfError from error
    finally:
        document.close()  # type: ignore[no-untyped-call]


def normalize_page_text(value: str) -> str:
    """Apply minimal deterministic normalization to extracted page text."""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "")
    return normalized.strip()


def normalize_metadata_field(value: object, maximum_length: int) -> str | None:
    """Normalize untrusted PDF metadata to a bounded single line."""
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize(
        "NFKC",
        value.replace("\r", " ").replace("\n", " ").replace("\t", " "),
    )
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("C")
    )
    normalized = " ".join(normalized.split())
    if not normalized:
        return None
    return normalized[:maximum_length]
