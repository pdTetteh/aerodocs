"""Focused tests for conservative heading detection."""

import pytest

from app.ingestion.headings import detect_heading, find_headings


@pytest.mark.parametrize(
    ("line", "standalone"),
    [
        ("1 Introduction", False),
        ("3.4.1 Inspection Procedure", False),
        ("SYSTEM LIMITATIONS", True),
        ("General Information", True),
    ],
)
def test_detect_heading_accepts_conservative_candidates(
    line: str,
    standalone: bool,
) -> None:
    assert detect_heading(line, standalone=standalone) == line


@pytest.mark.parametrize(
    ("line", "standalone"),
    [
        ("This is an ordinary sentence.", True),
        ("General information", False),
        ("12345", True),
        ("word " * 20, True),
        ("WARNING", True),
        ("CAUTION", True),
        ("NOTE", True),
        ("PAGE 12", False),
        ("FUEL PRESSURE", False),
        ("TABLE CELL", False),
    ],
)
def test_detect_heading_rejects_ambiguous_lines(
    line: str,
    standalone: bool,
) -> None:
    assert detect_heading(line, standalone=standalone) is None


def test_find_headings_preserves_offsets() -> None:
    text = "SYSTEM INTRODUCTION\nBody text.\n\n2 Scope\nMore text."

    headings = find_headings(text)

    assert headings == [(0, "SYSTEM INTRODUCTION"), (32, "2 Scope")]
