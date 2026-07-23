"""Conservative best-effort section-heading detection."""

import re

NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+){0,4}[.)]?\s+\S")
MAX_HEADING_CHARACTERS = 120
MAX_HEADING_WORDS = 12
WEAK_SINGLE_WORD_CANDIDATES = {
    "CAUTION",
    "NOTE",
    "PAGE",
    "TABLE",
    "WARNING",
}


def detect_heading(line: str, *, standalone: bool) -> str | None:
    """Return a reliable heading candidate or ``None``."""
    candidate = " ".join(line.split())
    words = candidate.split()
    if (
        not candidate
        or len(candidate) > MAX_HEADING_CHARACTERS
        or len(words) > MAX_HEADING_WORDS
    ):
        return None

    alphabetic = [character for character in candidate if character.isalpha()]
    if not alphabetic:
        return None

    if NUMBERED_HEADING.match(candidate):
        return candidate

    if not standalone:
        return None

    uppercase_ratio = sum(character.isupper() for character in alphabetic) / len(
        alphabetic
    )
    if (
        len(alphabetic) >= 3
        and uppercase_ratio >= 0.85
        and len(words) >= 2
        and candidate.upper() not in WEAK_SINGLE_WORD_CANDIDATES
    ):
        return candidate

    if candidate.endswith((".", "!", "?", ";", ":")):
        return None

    significant_words = [word for word in words if any(char.isalpha() for char in word)]
    if len(significant_words) < 2:
        return None
    if all(word[0].isupper() for word in significant_words):
        return candidate
    return None


def find_headings(text: str) -> list[tuple[int, str]]:
    """Return reliable heading offsets in normalized page text."""
    lines = text.splitlines(keepends=True)
    headings: list[tuple[int, str]] = []
    offset = 0
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        previous_blank = index == 0 or not lines[index - 1].strip()
        heading = detect_heading(
            line,
            standalone=previous_blank,
        )
        if heading is not None:
            leading_characters = len(raw_line) - len(raw_line.lstrip())
            headings.append((offset + leading_characters, heading))
        offset += len(raw_line)
    return headings
