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


@pytest.mark.parametrize(
    "line",
    [
        "1. Remove the access panel 192CB.",
        "2. Disconnect the electrical connector from the actuator.",
        "4. Disconnect the hydraulic lines. Install blanking caps on the open lines.",
        "6. Remove the actuator.",
        "7. Install the new actuator and torque the attach bolts to 80 in-lb.",
        "10. Do a leak check at 3000 psi.",
        "5 Remove the bolts that attach the actuator",
        "3.1 Remove the nut.",
        "2) Do the following:",
    ],
)
def test_numbered_procedure_steps_are_not_headings(line: str) -> None:
    assert detect_heading(line, standalone=False) is None
    assert detect_heading(line, standalone=True) is None


@pytest.mark.parametrize(
    "line",
    [
        "1. Introduction",
        "2 Scope",
        "3.1 General description",
        "4.2.1 Landing Gear Actuator",
        "5 Removal of the Actuator",
        "6 INSPECTION REQUIREMENTS",
    ],
)
def test_numbered_section_titles_remain_headings(line: str) -> None:
    assert detect_heading(line, standalone=False) == line


@pytest.mark.parametrize(
    "line",
    [
        "CAUTION: DO NOT USE A POWER TOOL TO TORQUE THE BOLTS.",
        "WARNING: MAKE SURE THE GROUND LOCKS ARE INSTALLED.",
        "NOTE: THIS TASK CAN BE DONE WITH THE GEAR DOWN",
        "DO NOT OPERATE THE ACTUATOR WITH THE PANEL OPEN.",
    ],
)
def test_safety_statements_are_not_headings(line: str) -> None:
    assert detect_heading(line, standalone=True) is None


def test_ata_style_task_title_is_still_a_heading() -> None:
    line = "32-31-00 MAIN LANDING GEAR ACTUATOR - REMOVAL/INSTALLATION"

    assert detect_heading(line, standalone=True) == line
