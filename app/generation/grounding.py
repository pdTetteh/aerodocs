"""Deterministic semantic guards used before lexical grounding acceptance."""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from app.generation.models import EvidencePassage

WORD = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*", re.I)
NUMBER = re.compile(
    r"(?P<first>-?\d+(?:\.\d+)?)"
    r"(?:\s*(?:to|-)\s*(?P<second>-?\d+(?:\.\d+)?))?"
    r"\s*(?P<unit>mm|cm|m|in|ft|psi|kpa|mpa|nm|lb|kg|°c|°f)?\b",
    re.I,
)
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
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
CONVERSIONS: dict[tuple[str, str], Decimal] = {
    ("in", "mm"): Decimal("25.4"),
    ("ft", "m"): Decimal("0.3048"),
    ("psi", "kpa"): Decimal("6.894757"),
    ("lb", "kg"): Decimal("0.45359237"),
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
            unit=(
                match.group("unit").casefold()
                if match.group("unit") is not None
                else None
            ),
        )
        for match in NUMBER.finditer(normalized)
    )
    return SemanticSignature(
        markers=markers,
        quantities=quantities,
        dates=tuple(DATE.findall(normalized)),
        revisions=tuple(match.casefold() for match in REVISION.findall(normalized)),
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
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _quantity_equal(first: Quantity, second: Quantity) -> bool:
    if (first.second is None) != (second.second is None):
        return False
    if first.unit == second.unit:
        return first.first == second.first and first.second == second.second
    if first.unit is None or second.unit is None:
        return False
    factor = CONVERSIONS.get((first.unit, second.unit))
    reverse = False
    if factor is None:
        factor = CONVERSIONS.get((second.unit, first.unit))
        reverse = factor is not None
    if factor is None:
        return False
    converted_first = first.first / factor if reverse else first.first * factor
    converted_second = (
        None
        if first.second is None
        else (first.second / factor if reverse else first.second * factor)
    )
    tolerance = Decimal("0.01")
    return abs(converted_first - second.first) <= tolerance and (
        converted_second is None
        or second.second is not None
        and abs(converted_second - second.second) <= tolerance
    )


def _scope(sentence: str) -> str:
    normalized = normalize_security_text(sentence).casefold()
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
