"""Conservative best-effort section-heading detection."""

import re

NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+){0,4}[.)]?\s+\S")
NUMBER_PREFIX = re.compile(r"^(?P<number>\d+(?:\.\d+){0,4})[.)]?\s+(?P<title>.+)$")
SAFETY_LABEL = re.compile(r"^(?:WARNING|CAUTION|NOTE)S?\b(?:\s*[:\-–—]|\s)")
SENTENCE_ENDINGS = (".", "!", "?", ";", ":")
SENTENCE_BREAK = re.compile(r"[.!?]\s+\S")
TITLE_CASE_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or"}
    | {"per", "the", "to", "with"}
)
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

    if SAFETY_LABEL.match(candidate):
        return None

    if NUMBERED_HEADING.match(candidate):
        return candidate if _is_numbered_section_title(candidate) else None

    if not standalone:
        return None

    uppercase_ratio = sum(character.isupper() for character in alphabetic) / len(
        alphabetic
    )
    if (
        len(alphabetic) >= 3
        and uppercase_ratio >= 0.85
        and len(words) >= 2
        and not candidate.endswith(".")
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


def _is_numbered_section_title(candidate: str) -> bool:
    """Tell a numbered section title from a numbered procedure step.

    Maintenance procedures are numbered lists of imperative sentences. Those are
    body text: treating them as headings would split a task into one chunk per
    step and detach each step from the warning that governs it.
    """
    match = NUMBER_PREFIX.match(candidate)
    if match is None:
        return False
    depth = match.group("number").count(".") + 1
    title = match.group("title").strip()
    if not title or title.endswith(SENTENCE_ENDINGS):
        return False
    if SENTENCE_BREAK.search(title):
        return False
    if depth > 1:
        return True
    return _is_title_case(title)


def _is_title_case(title: str) -> bool:
    """Accept ``Introduction``/``General Information``/``SCOPE``, not sentences."""
    for word in title.split():
        if not word[0].isalpha() or word.casefold() in TITLE_CASE_SMALL_WORDS:
            continue
        if not word[0].isupper():
            return False
    return True


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
