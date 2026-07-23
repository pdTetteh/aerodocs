"""Tests for PDF metadata and page extraction."""

from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest

from app.ingestion.errors import PdfExpansionLimitError
from app.ingestion.pdf import PdfExtraction, extract_pdf, normalize_metadata_field


def test_extract_pdf_preserves_metadata_and_page_numbers(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    path = tmp_path / "source.pdf"
    path.write_bytes(
        make_pdf(
            ["First page", "Second page"],
            metadata={
                "title": "Public Test Manual",
                "author": "FAA Test Author",
                "subject": "Synthetic test subject",
            },
        )
    )

    extraction = extract_pdf(
        path,
        "doc_" + "a" * 64,
        maximum_pages=10,
        maximum_characters_per_page=1_000,
        maximum_characters_total=2_000,
    )

    assert extraction.title == "Public Test Manual"
    assert extraction.author == "FAA Test Author"
    assert extraction.subject == "Synthetic test subject"
    assert [page.page_number for page in extraction.pages] == [1, 2]
    assert "First page" in extraction.pages[0].text
    assert "Second page" in extraction.pages[1].text


def test_blank_page_produces_clear_warning(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_pdf([""]))

    extraction = extract_pdf(
        path,
        "doc_" + "b" * 64,
        maximum_pages=10,
        maximum_characters_per_page=1_000,
        maximum_characters_total=2_000,
    )

    assert extraction.pages[0].text == ""
    assert extraction.pages[0].character_count == 0
    assert extraction.pages[0].warnings == ["no_extractable_text"]


def test_metadata_normalization_removes_controls_and_limits_length() -> None:
    value = "  Test\x00\r\nTitle\u202e " + "x" * 50

    assert normalize_metadata_field(value, 20) == "Test Title xxxxxxxxx"
    assert normalize_metadata_field("\x00\n", 20) is None


class FakePage:
    """Minimal page stub for bounded extraction tests."""

    def __init__(self, text: str) -> None:
        self.text = text

    def get_text(self, _format: str) -> str:
        return self.text


class FakeDocument:
    """Minimal PyMuPDF document stub with observable page loading."""

    needs_pass = False
    is_encrypted = False
    metadata: dict[str, str] = {}

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.page_count = len(texts)
        self.load_calls = 0

    def load_page(self, index: int) -> FakePage:
        self.load_calls += 1
        return FakePage(self.texts[index])

    def close(self) -> None:
        return None


def extract_stub(
    monkeypatch: pytest.MonkeyPatch,
    texts: list[str],
    *,
    maximum_pages: int,
    maximum_per_page: int,
    maximum_total: int,
) -> tuple[PdfExtraction, FakeDocument]:
    """Extract from a lightweight fake document."""
    document = FakeDocument(texts)
    monkeypatch.setattr(pymupdf, "open", lambda _path: document)
    result = extract_pdf(
        Path("unused.pdf"),
        "doc_" + "c" * 64,
        maximum_pages=maximum_pages,
        maximum_characters_per_page=maximum_per_page,
        maximum_characters_total=maximum_total,
    )
    return result, document


def test_page_count_exact_limit_and_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction, document = extract_stub(
        monkeypatch,
        ["a", "b"],
        maximum_pages=2,
        maximum_per_page=10,
        maximum_total=10,
    )
    assert len(extraction.pages) == 2
    assert document.load_calls == 2

    over_limit = FakeDocument(["a", "b", "c"])
    monkeypatch.setattr(pymupdf, "open", lambda _path: over_limit)
    with pytest.raises(PdfExpansionLimitError):
        extract_pdf(
            Path("unused.pdf"),
            "doc_" + "c" * 64,
            maximum_pages=2,
            maximum_characters_per_page=10,
            maximum_characters_total=10,
        )
    assert over_limit.load_calls == 0


def test_per_page_character_exact_limit_and_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction, _document = extract_stub(
        monkeypatch,
        ["abc"],
        maximum_pages=1,
        maximum_per_page=3,
        maximum_total=3,
    )
    assert extraction.pages[0].character_count == 3

    with pytest.raises(PdfExpansionLimitError):
        extract_stub(
            monkeypatch,
            ["abcd"],
            maximum_pages=1,
            maximum_per_page=3,
            maximum_total=10,
        )


def test_cumulative_character_exact_limit_and_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction, _document = extract_stub(
        monkeypatch,
        ["abc", "de"],
        maximum_pages=2,
        maximum_per_page=3,
        maximum_total=5,
    )
    assert sum(page.character_count for page in extraction.pages) == 5

    with pytest.raises(PdfExpansionLimitError):
        extract_stub(
            monkeypatch,
            ["abc", "def"],
            maximum_pages=2,
            maximum_per_page=3,
            maximum_total=5,
        )
