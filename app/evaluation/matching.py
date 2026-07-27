"""Deterministic evidence-gated one-to-one claim matching."""

from __future__ import annotations

import json
from functools import cache

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.benchmark import EvidenceTarget, GoldClaim
from app.generation.limits import MAX_DRAFT_CLAIMS


class ObservedSupport(BaseModel):
    """One returned support link validated against authoritative evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    target_ids: list[str] = Field(default_factory=list)
    evidence_id_valid: bool
    quotation_present: bool
    quotation_valid: bool
    provenance_valid: bool
    citation_consistent: bool
    duplicate: bool = False

    @property
    def valid(self) -> bool:
        return (
            self.evidence_id_valid
            and self.quotation_valid
            and self.provenance_valid
            and self.citation_consistent
            and not self.duplicate
        )


class ReturnedClaim(BaseModel):
    """A production claim with derived, never caller-asserted validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    text: str
    supports: list[ObservedSupport] = Field(default_factory=list)
    preserved_fact_ids: list[str] = Field(default_factory=list)
    exact_quotes: list[str] = Field(default_factory=list)
    structurally_valid: bool = True


class ClaimPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_claim_id: str
    returned_claim_id: str
    weight: int
    support_evidence_ids: list[str]


class ClaimMatching(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pairs: list[ClaimPair]
    unmatched_gold_claim_ids: list[str]
    unmatched_returned_claim_ids: list[str]
    total_weight: int


class _Edge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    weight: int
    support_evidence_ids: tuple[str, ...]
    shared_group: str | None


def target_id(target: EvidenceTarget) -> str:
    """Return the canonical identity of a validated evidence target."""
    data = target.model_dump(exclude={"grade"}, exclude_none=True)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def match_claims(
    gold_claims: list[GoldClaim],
    returned_claims: list[ReturnedClaim],
) -> ClaimMatching:
    """Maximise valid matches, then coverage/quotes, with stable ID ties."""
    if len(gold_claims) > MAX_DRAFT_CLAIMS or len(returned_claims) > MAX_DRAFT_CLAIMS:
        raise ValueError("claim_count_exceeds_structured_output_limit")
    gold = sorted(gold_claims, key=lambda item: item.claim_id)
    returned = sorted(returned_claims, key=lambda item: item.claim_id)
    edges = [
        [_edge(gold_item, returned_item) for returned_item in returned]
        for gold_item in gold
    ]

    @cache
    def solve(
        gold_index: int,
        used_returned: int,
        used_support: tuple[tuple[str, str], ...],
    ) -> tuple[int, int, tuple[tuple[int, int], ...]]:
        if gold_index == len(gold):
            return (0, 0, ())
        best = solve(gold_index + 1, used_returned, used_support)
        support_owners = dict(used_support)
        for returned_index, edge in enumerate(edges[gold_index]):
            if edge is None or used_returned & (1 << returned_index):
                continue
            if not _support_reuse_allowed(edge, support_owners):
                continue
            following = dict(support_owners)
            owner = edge.shared_group or gold[gold_index].claim_id
            for evidence_id in edge.support_evidence_ids:
                following[evidence_id] = owner
            tail_count, tail_weight, tail_pairs = solve(
                gold_index + 1,
                used_returned | (1 << returned_index),
                tuple(sorted(following.items())),
            )
            candidate = (
                tail_count + 1,
                tail_weight + edge.weight,
                ((gold_index, returned_index),) + tail_pairs,
            )
            if _better(candidate, best, gold, returned):
                best = candidate
        return best

    matched_count, total_weight, indexes = solve(0, 0, ())
    del matched_count
    pairs = [
        ClaimPair(
            gold_claim_id=gold[gold_index].claim_id,
            returned_claim_id=returned[returned_index].claim_id,
            weight=edges[gold_index][returned_index].weight,  # type: ignore[union-attr]
            support_evidence_ids=list(
                edges[gold_index][returned_index].support_evidence_ids  # type: ignore[union-attr]
            ),
        )
        for gold_index, returned_index in indexes
    ]
    matched_gold = {pair.gold_claim_id for pair in pairs}
    matched_returned = {pair.returned_claim_id for pair in pairs}
    return ClaimMatching(
        pairs=pairs,
        unmatched_gold_claim_ids=[
            item.claim_id for item in gold if item.claim_id not in matched_gold
        ],
        unmatched_returned_claim_ids=[
            item.claim_id for item in returned if item.claim_id not in matched_returned
        ],
        total_weight=total_weight,
    )


def _edge(gold: GoldClaim, returned: ReturnedClaim) -> _Edge | None:
    if not returned.structurally_valid:
        return None
    allowed = {target_id(target) for target in gold.support_targets}
    valid_supports = [
        support
        for support in returned.supports
        if support.valid and allowed.intersection(support.target_ids)
    ]
    if not gold.support_optional and not valid_supports:
        return None
    required_fact_ids = {fact.fact_id for fact in gold.facts}
    preserved = required_fact_ids.intersection(returned.preserved_fact_ids)
    if preserved != required_fact_ids:
        return None
    required_quotes = set(gold.exact_quotes)
    quote_matches = required_quotes.intersection(returned.exact_quotes)
    if quote_matches != required_quotes:
        return None
    evidence_ids = tuple(sorted({item.evidence_id for item in valid_supports}))
    support_coverage = len(
        {
            target
            for support in valid_supports
            for target in support.target_ids
            if target in allowed
        }
    )
    return _Edge(
        weight=support_coverage * 1_000_000
        + len(preserved) * 1_000
        + len(quote_matches),
        support_evidence_ids=evidence_ids,
        shared_group=gold.shared_support_group,
    )


def _support_reuse_allowed(
    edge: _Edge,
    owners: dict[str, str],
) -> bool:
    for evidence_id in edge.support_evidence_ids:
        existing = owners.get(evidence_id)
        if existing is not None and (
            edge.shared_group is None or existing != edge.shared_group
        ):
            return False
    return True


def _better(
    candidate: tuple[int, int, tuple[tuple[int, int], ...]],
    incumbent: tuple[int, int, tuple[tuple[int, int], ...]],
    gold: list[GoldClaim],
    returned: list[ReturnedClaim],
) -> bool:
    if candidate[:2] != incumbent[:2]:
        return candidate[:2] > incumbent[:2]

    def names(indexes: tuple[tuple[int, int], ...]) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted((gold[g].claim_id, returned[r].claim_id) for g, r in indexes)
        )

    return names(candidate[2]) < names(incumbent[2])
