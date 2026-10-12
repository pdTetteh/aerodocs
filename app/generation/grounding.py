"""Deterministic semantic guards used before lexical grounding acceptance."""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from app.generation.models import EvidencePassage

WORD = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*", re.I)

_SEPARATOR = r"[ .\-·⋅]?"
# Compound torque units come first so "in-lb" is never read as inches.
UNIT_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "in_lb",
        rf"(?:in(?:ch(?:es)?)?{_SEPARATOR}(?:lbs?|lbf|pounds?)|lbf{_SEPARATOR}in)",
    ),
    (
        "ft_lb",
        rf"(?:(?:ft|foot|feet){_SEPARATOR}(?:lbs?|lbf|pounds?)|lbf{_SEPARATOR}ft)",
    ),
    ("n_m", rf"(?:n{_SEPARATOR}m|newton{_SEPARATOR}met(?:er|re)s?)"),
    ("kpa", r"kpa|kilopascals?"),
    ("mpa", r"mpa|megapascals?"),
    ("bar", r"bar"),
    ("psi", r"psi[ag]?"),
    ("c", r"(?:°\s?c|deg(?:rees?)?\s?c)"),
    ("f", r"(?:°\s?f|deg(?:rees?)?\s?f)"),
    ("mm", r"mm|millimet(?:er|re)s?"),
    ("cm", r"cm|centimet(?:er|re)s?"),
    ("m", r"m(?:et(?:er|re)s?)?"),
    ("in", r"in(?:ch(?:es)?)?"),
    ("ft", r"ft|foot|feet"),
    ("kg", r"kg|kilograms?"),
    ("lb", r"lbs?|pounds?"),
    ("hr", r"(?:flight[ -]?)?(?:hours?|hrs?|fh)"),
    ("cycle", r"(?:flight[ -]?)?(?:cycles?|fc)"),
    ("landing", r"landings?"),
)
_UNIT_ALTERNATION = "|".join(f"(?:{pattern})" for _, pattern in UNIT_PATTERNS)
_UNIT_FULL_MATCHES = tuple(
    (name, re.compile(pattern, re.I)) for name, pattern in UNIT_PATTERNS
)
NUMBER = re.compile(
    r"(?P<first>-?\d+(?:\.\d+)?)"
    r"(?:\s*(?:to|-)\s*(?P<second>-?\d+(?:\.\d+)?))?"
    rf"(?:\s*(?P<unit>{_UNIT_ALTERNATION}))?(?![a-z0-9])",
    re.I,
)
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Part numbers, AD numbers and ATA references identify things; they are not
# quantities and must never be parsed as ranges ("NAS1351-3-8" is not 1351 to 3).
IDENTIFIER = re.compile(
    r"\b[a-z]{1,6}\d[a-z0-9]*(?:-[a-z0-9]+)+\b"
    r"|\b[a-z]{1,6}\d{2,}[a-z0-9]*\b"
    r"|\b\d{1,4}(?:-\d{1,4}){2,}\b",
    re.I,
)
REVISION = re.compile(r"\brev(?:ision)?(?:\s+is)?\s+([a-z0-9.-]+)\b", re.I)
VARIANT = re.compile(r"\bvariant\s+([a-z0-9.-]+)\b", re.I)
CONDITION = re.compile(r"\b([a-z0-9.-]+)[ -]condition\b", re.I)

PHRASES: dict[str, tuple[str, ...]] = {
    "negative": (
        "not",
        "no",
        "never",
        "may not",
        "must not",
        "should not",
        "not required",
        "not approved",
    ),
    "universal": ("always", "only", "never"),
    "modality": (
        "may",
        "may not",
        "must",
        "must not",
        "should",
        "should not",
        "required",
        "not required",
    ),
    "permission": ("permitted", "prohibited"),
    "safety": ("safe", "unsafe"),
    "approval": ("approved", "unapproved", "not approved"),
    "position": ("open", "closed"),
    "serviceability": ("serviceable", "unserviceable"),
    "comparator": (
        "minimum",
        "maximum",
        "less than",
        "no less than",
        "greater than",
        "no greater than",
        "equal to",
    ),
}
OPPOSITES = (
    ("permitted", "prohibited"),
    ("safe", "unsafe"),
    ("approved", "unapproved"),
    ("approved", "not approved"),
    ("open", "closed"),
    ("serviceable", "unserviceable"),
    ("required", "not required"),
    ("may", "may not"),
    ("must", "must not"),
    ("should", "should not"),
    ("always", "never"),
)
STOP_SCOPE = {
    "a",
    "an",
    "and",
    "be",
    "is",
    "of",
    "the",
    "to",
    "value",
}
LINEAR_UNITS: dict[str, tuple[str, Decimal]] = {
    "n_m": ("torque", Decimal("1")),
    "in_lb": ("torque", Decimal("0.1129848290276")),
    "ft_lb": ("torque", Decimal("1.3558179483314")),
    "kpa": ("pressure", Decimal("1")),
    "mpa": ("pressure", Decimal("1000")),
    "bar": ("pressure", Decimal("100")),
    "psi": ("pressure", Decimal("6.894757")),
    "mm": ("length", Decimal("1")),
    "cm": ("length", Decimal("10")),
    "m": ("length", Decimal("1000")),
    "in": ("length", Decimal("25.4")),
    "ft": ("length", Decimal("304.8")),
    "kg": ("mass", Decimal("1")),
    "lb": ("mass", Decimal("0.45359237")),
}
TEMPERATURE_UNITS = frozenset({"c", "f"})
RELATIVE_CONVERSION_TOLERANCE = Decimal("0.01")
TEMPERATURE_TOLERANCE_CELSIUS = Decimal("0.6")
MONTH_OR_ABBREVIATION_EXCLUSIONS: dict[str, str] = {
    # "May 2024" / "5 May" are dates, not the modal verb.
    "may": r"(?<!\d\s)(?<!\d)\bmay\b(?!\s+\d)",
    # "Fig. No. 4" is an abbreviation, not a negation.
    "no": r"\bno\b(?!\.?\s*\d)",
}


@dataclass(frozen=True)
class Quantity:
    first: Decimal
    second: Decimal | None
    unit: str | None


@dataclass(frozen=True)
class SemanticSignature:
    markers: dict[str, frozenset[str]]
    quantities: tuple[Quantity, ...]
    dates: tuple[str, ...]
    revisions: tuple[str, ...]
    identifiers: frozenset[str] = frozenset()


def semantic_signature(text: str) -> SemanticSignature:
    """Extract only explicit safety-relevant semantics from normalized text."""
    normalized = normalize_security_text(text).casefold()
    markers = {
        category: frozenset(
            phrase for phrase in phrases if _contains_phrase(normalized, phrase)
        )
        for category, phrases in PHRASES.items()
    }
    quantities = tuple(
        Quantity(
            first=Decimal(match.group("first")),
            second=(
                Decimal(match.group("second"))
                if match.group("second") is not None
                else None
            ),
            unit=_canonical_unit(match.group("unit")),
        )
        for match in NUMBER.finditer(_mask_identifiers(normalized))
    )
    return SemanticSignature(
        markers=markers,
        quantities=quantities,
        dates=tuple(DATE.findall(normalized)),
        revisions=tuple(match.casefold() for match in REVISION.findall(normalized)),
        identifiers=frozenset(
            identifier.casefold()
            for identifier in IDENTIFIER.findall(DATE.sub(" ", normalized))
        ),
    )


def semantic_compatible(claim: str, evidence: str) -> bool:
    """Reject explicit polarity, qualifier, comparator, value, or unit drift."""
    claim_sentences = _sentences(claim)
    evidence_sentences = _sentences(evidence)
    return all(
        any(
            _scope_aligned(claim_sentence, evidence_sentence)
            and _signatures_compatible(
                semantic_signature(claim_sentence),
                semantic_signature(evidence_sentence),
            )
            for evidence_sentence in evidence_sentences
        )
        for claim_sentence in claim_sentences
    )


def _signatures_compatible(
    claim_signature: SemanticSignature,
    evidence_signature: SemanticSignature,
) -> bool:
    if bool(claim_signature.markers["negative"]) != bool(
        evidence_signature.markers["negative"]
    ):
        return False
    for category in (
        "universal",
        "modality",
        "permission",
        "safety",
        "approval",
        "position",
        "serviceability",
        "comparator",
    ):
        claimed = claim_signature.markers[category]
        if claimed and claimed != evidence_signature.markers[category]:
            return False
    if not claim_signature.identifiers <= evidence_signature.identifiers:
        return False
    if not set(claim_signature.dates) <= set(evidence_signature.dates):
        return False
    if claim_signature.quantities and not all(
        any(
            _quantity_equal(claimed, supported)
            for supported in evidence_signature.quantities
        )
        for claimed in claim_signature.quantities
    ):
        return False
    return True


def _scope_aligned(claim: str, evidence: str) -> bool:
    for pattern in (VARIANT, CONDITION):
        claim_labels = {value.casefold() for value in pattern.findall(claim)}
        evidence_labels = {value.casefold() for value in pattern.findall(evidence)}
        if (claim_labels or evidence_labels) and claim_labels != evidence_labels:
            return False
    claim_tokens = set(_scope(claim).split())
    evidence_tokens = set(_scope(evidence).split())
    if not claim_tokens or not evidence_tokens:
        return False
    overlap = claim_tokens & evidence_tokens
    return bool(overlap) and len(overlap) / len(claim_tokens | evidence_tokens) >= 0.5


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


def detect_semantic_conflicts(evidence: list[EvidencePassage]) -> list[str]:
    """Find narrow contradictions only among statements with matching scope."""
    observations: dict[str, list[tuple[SemanticSignature, str]]] = {}
    conflicts: set[str] = set()
    for passage in evidence:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", passage.text):
            if not sentence.strip():
                continue
            signature = semantic_signature(sentence)
            scope = _scope(sentence)
            if len(scope) < 3:
                continue
            for previous, evidence_id in observations.get(scope, []):
                if _signatures_conflict(previous, signature):
                    conflicts.update((evidence_id, passage.evidence_id))
            observations.setdefault(scope, []).append((signature, passage.evidence_id))
    return sorted(conflicts)


def normalize_security_text(text: str) -> str:
    """Normalize Unicode compatibility forms and remove invisible format marks."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character for character in normalized if unicodedata.category(character) != "Cf"
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = MONTH_OR_ABBREVIATION_EXCLUSIONS.get(phrase, rf"\b{re.escape(phrase)}\b")
    return re.search(pattern, text) is not None


def _mask_identifiers(text: str) -> str:
    """Blank part, AD and ATA identifiers so they are not read as quantities."""
    return IDENTIFIER.sub(" ", text)


def _canonical_unit(raw: str | None) -> str | None:
    if raw is None:
        return None
    unit = " ".join(raw.casefold().split())
    for name, pattern in _UNIT_FULL_MATCHES:
        if pattern.fullmatch(unit):
            return name
    return unit


def _quantity_equal(first: Quantity, second: Quantity) -> bool:
    if (first.second is None) != (second.second is None):
        return False
    if first.unit == second.unit:
        return first.first == second.first and first.second == second.second
    if first.unit is None or second.unit is None:
        return False
    dimension = _dimension(first.unit)
    if dimension is None or dimension != _dimension(second.unit):
        return False
    pairs = [(first.first, second.first)]
    if first.second is not None and second.second is not None:
        pairs.append((first.second, second.second))
    return all(
        _converted_close(left, first.unit, right, second.unit, dimension)
        for left, right in pairs
    )


def _dimension(unit: str) -> str | None:
    if unit in TEMPERATURE_UNITS:
        return "temperature"
    linear = LINEAR_UNITS.get(unit)
    return linear[0] if linear is not None else None


def _to_base(value: Decimal, unit: str) -> Decimal:
    """Convert to the dimension's base unit (kelvin offsets are not needed)."""
    if unit == "f":
        return (value - Decimal("32")) * Decimal("5") / Decimal("9")
    if unit == "c":
        return value
    return value * LINEAR_UNITS[unit][1]


def _converted_close(
    left: Decimal,
    left_unit: str,
    right: Decimal,
    right_unit: str,
    dimension: str,
) -> bool:
    """Compare across units, allowing only ordinary conversion rounding."""
    left_base = _to_base(left, left_unit)
    right_base = _to_base(right, right_unit)
    difference = abs(left_base - right_base)
    if dimension == "temperature":
        return difference <= TEMPERATURE_TOLERANCE_CELSIUS
    return difference <= RELATIVE_CONVERSION_TOLERANCE * max(
        abs(left_base), abs(right_base)
    )


def _scope(sentence: str) -> str:
    normalized = _mask_identifiers(normalize_security_text(sentence).casefold())
    normalized = NUMBER.sub(" value ", normalized)
    normalized = DATE.sub(" date ", normalized)
    normalized = REVISION.sub(" revision ", normalized)
    excluded = {
        token
        for phrases in PHRASES.values()
        for phrase in phrases
        for token in WORD.findall(phrase)
    }
    tokens = [
        token
        for token in WORD.findall(normalized)
        if token not in STOP_SCOPE and token not in excluded
    ]
    return " ".join(tokens)


def _signatures_conflict(
    first: SemanticSignature,
    second: SemanticSignature,
) -> bool:
    if bool(first.markers["negative"]) != bool(second.markers["negative"]):
        return True
    all_first = set().union(*first.markers.values())
    all_second = set().union(*second.markers.values())
    if any(
        (left in all_first and right in all_second)
        or (right in all_first and left in all_second)
        for left, right in OPPOSITES
    ):
        return True
    if (
        first.quantities
        and second.quantities
        and not all(
            any(_quantity_equal(value, other) for other in second.quantities)
            for value in first.quantities
        )
    ):
        return True
    if first.dates and second.dates and first.dates != second.dates:
        return True
    return bool(
        first.revisions and second.revisions and first.revisions != second.revisions
    )
