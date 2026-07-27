"""Milestone 6 Phase 2 deterministic metric and grounding tests."""

from __future__ import annotations

from math import isclose, log2

import pytest
from pydantic import TypeAdapter, ValidationError

from app.evaluation.aggregation import (
    AnswerObservation,
    FactCoverage,
    RankedRetrievalItem,
    RetrievalObservation,
    answer_metrics,
    conflict_metrics,
    injection_metrics,
    retrieval_metrics,
    safety_metrics,
)
from app.evaluation.benchmark import (
    AnswerBenchmarkCase,
    ApplicabilityFact,
    ComparatorFact,
    GoldClaim,
    ModalityFact,
    NegationFact,
    NumericFact,
    PageTarget,
    QualifierFact,
    RangeFact,
    StatusFact,
    StructuredFact,
    VerifiedSource,
)
from app.evaluation.matching import (
    ObservedSupport,
    ReturnedClaim,
    match_claims,
    target_id,
)
from app.evaluation.metrics import MetricRecord, count_record, validate_unique_metrics
from app.evaluation.observations import (
    ServiceTrace,
    build_answer_observation,
    fact_disposition,
    fact_preserved,
)
from app.evaluation.reporting import GateThreshold
from app.generation.models import (
    AnswerClaim,
    AnswerResponse,
    Citation,
    ClaimSupport,
    EvidencePassage,
    GroundingOutcome,
    ProviderIdentity,
)


def metric_map(records: list[MetricRecord]) -> dict[tuple[str, str], MetricRecord]:
    return {(item.population, item.metric_id): item for item in records}


def retrieval_observation(
    items: list[tuple[str, list[str]]],
    grades: dict[str, int],
    *,
    succeeded: bool = True,
) -> RetrievalObservation:
    return RetrievalObservation(
        case_id="retrieval-case",
        population="model_quality",
        execution_succeeded=succeeded,
        target_grades=grades,
        ranked_items=[
            RankedRetrievalItem(evidence_id=evidence, matching_target_ids=targets)
            for evidence, targets in items
        ],
        returned_count=len(items) if succeeded else None,
    )


def test_retrieval_targets_and_duplicate_evidence_are_consumed_once() -> None:
    observation = retrieval_observation(
        [
            ("chunk-a", ["high", "low"]),
            ("chunk-a", ["high"]),
            ("chunk-b", ["high"]),
            ("chunk-c", ["low"]),
        ],
        {"high": 3, "low": 1},
    )

    records = metric_map(retrieval_metrics([observation]))

    assert records[("retrieval_quality", "recall_at_1")].value == 1
    assert records[("retrieval_quality", "ndcg_at_1")].value == 1
    assert records[("retrieval_quality", "recall_at_3")].value == 1
    assert 0 <= records[("retrieval_quality", "ndcg_at_5")].value <= 1  # type: ignore[operator]


def test_perfect_graded_ndcg_is_one_and_never_exceeds_one() -> None:
    perfect = retrieval_observation(
        [("a", ["g3"]), ("b", ["g2"]), ("c", ["g1"])],
        {"g1": 1, "g2": 2, "g3": 3},
    )
    repeated = retrieval_observation(
        [("a", ["g3"]), ("b", ["g3"]), ("c", ["g3"])],
        {"g1": 1, "g2": 2, "g3": 3},
    )

    perfect_value = metric_map(retrieval_metrics([perfect]))[
        ("retrieval_quality", "ndcg_at_5")
    ].value
    repeated_value = metric_map(retrieval_metrics([repeated]))[
        ("retrieval_quality", "ndcg_at_5")
    ].value

    assert perfect_value == 1
    assert repeated_value is not None and 0 <= repeated_value <= 1


def test_duplicate_rows_preserve_original_rank_positions() -> None:
    observation = retrieval_observation(
        [("irrelevant", []), ("irrelevant", []), ("relevant", ["target"])],
        {"target": 1},
    )

    records = metric_map(retrieval_metrics([observation]))

    assert records[("retrieval_quality", "mrr")].value == pytest.approx(1 / 3)
    assert records[("retrieval_quality", "ndcg_at_3")].value == pytest.approx(
        1 / log2(4)
    )
    assert records[("retrieval_quality", "recall_at_1")].value == 0


def test_duplicate_relevant_row_occupies_rank_without_second_credit() -> None:
    observation = retrieval_observation(
        [("first", ["high"]), ("first", ["high"]), ("second", ["low"])],
        {"high": 3, "low": 1},
    )

    records = metric_map(retrieval_metrics([observation]))
    expected_dcg = 7 + (1 / log2(4))
    ideal_dcg = 7 + (1 / log2(3))

    assert records[("retrieval_quality", "recall_at_1")].value == 0.5
    assert records[("retrieval_quality", "recall_at_3")].value == 1
    assert records[("retrieval_quality", "ndcg_at_3")].value == pytest.approx(
        expected_dcg / ideal_dcg
    )


def test_zero_relevance_and_failed_no_answer_are_not_successes() -> None:
    empty = retrieval_observation([], {})
    failure = retrieval_observation([], {}, succeeded=False)
    records = metric_map(retrieval_metrics([empty, failure]))

    assert records[("retrieval_quality", "recall_at_5")].value is None
    assert (
        records[("retrieval_quality", "no_answer_retrieval_precision")].numerator == 1
    )
    assert (
        records[("retrieval_quality", "no_answer_retrieval_precision")].denominator == 1
    )
    assert records[("operational_reliability", "operational_failure_rate")].value == 0.5


def valid_support(
    evidence_id: str,
    target: str,
    *,
    duplicate: bool = False,
    valid: bool = True,
) -> ObservedSupport:
    return ObservedSupport(
        evidence_id=evidence_id,
        target_ids=[target],
        evidence_id_valid=valid,
        quotation_present=True,
        quotation_valid=valid,
        provenance_valid=valid,
        citation_consistent=valid,
        duplicate=duplicate,
    )


def gold_claim(
    claim_id: str,
    page: int,
    *,
    shared: str | None = None,
) -> GoldClaim:
    return GoldClaim(
        claim_id=claim_id,
        support_targets=[
            PageTarget(
                source_id="src_manual",
                page_start=page,
                page_end=page,
            )
        ],
        facts=[NumericFact(fact_id=f"fact-{claim_id}", value="10")],
        shared_support_group=shared,
    )


def returned(
    claim_id: str,
    evidence_id: str,
    target: str,
    fact_id: str,
    *,
    valid: bool = True,
    duplicate: bool = False,
) -> ReturnedClaim:
    return ReturnedClaim(
        claim_id=claim_id,
        text="10",
        supports=[valid_support(evidence_id, target, valid=valid, duplicate=duplicate)],
        preserved_fact_ids=[fact_id],
    )


def test_matching_is_one_to_one_and_invalid_high_overlap_cannot_match() -> None:
    gold = gold_claim("gold", 1)
    target = target_id(gold.support_targets[0])
    generated = [
        returned("first", "ev", target, "fact-gold"),
        returned("duplicate", "ev", target, "fact-gold"),
        returned("invalid", "bad", target, "fact-gold", valid=False),
    ]

    result = match_claims([gold], generated)

    assert len(result.pairs) == 1
    assert result.pairs[0].returned_claim_id == "duplicate"
    assert sorted(result.unmatched_returned_claim_ids) == ["first", "invalid"]


def test_matching_requires_full_fact_and_evidence_alignment() -> None:
    first = gold_claim("first", 1)
    second = gold_claim("second", 2)
    first_target = target_id(first.support_targets[0])
    partial = returned("returned", "ev", first_target, "fact-second")

    result = match_claims([first, second], [partial])

    assert result.pairs == []


def test_shared_support_requires_explicit_common_group() -> None:
    target = PageTarget(source_id="src_manual", page_start=1, page_end=1)
    target_key = target_id(target)
    declared = [
        GoldClaim(
            claim_id="first",
            support_targets=[target],
            facts=[],
            shared_support_group="shared",
        ),
        GoldClaim(
            claim_id="second",
            support_targets=[target],
            facts=[],
            shared_support_group="shared",
        ),
    ]
    undeclared = [
        item.model_copy(update={"shared_support_group": None}) for item in declared
    ]
    generated = [
        returned("a", "same", target_key, "unused").model_copy(
            update={"preserved_fact_ids": []}
        ),
        returned("b", "same", target_key, "unused").model_copy(
            update={"preserved_fact_ids": []}
        ),
    ]

    assert len(match_claims(declared, generated).pairs) == 2
    assert len(match_claims(undeclared, generated).pairs) == 1


def source() -> VerifiedSource:
    return VerifiedSource(
        source_id="src_manual",
        kind="synthetic",
        source_fingerprint="a" * 64,
        document_id=f"doc_{'a' * 64}",
        page_count=1,
        specification_fingerprint="b" * 64,
        processing_profile_id=None,
        chunking_profile_id=None,
        manifest_fingerprint=None,
        allowed_corpus_fingerprint=None,
    )


def answer_case() -> AnswerBenchmarkCase:
    return AnswerBenchmarkCase(
        case_id="answer-case",
        population="model_quality",
        suite="answers",
        kind="answer",
        question="What torque is required?",
        expected_outcome="supported_answer",
        complete_support_set=True,
        document_source_ids=["src_manual"],
        gold_claims=[
            GoldClaim(
                claim_id="gold",
                support_targets=[
                    PageTarget(
                        source_id="src_manual",
                        page_start=1,
                        page_end=1,
                        section="LIMITS",
                    )
                ],
                facts=[
                    NumericFact(
                        fact_id="torque",
                        value="10",
                        unit="Nm",
                    )
                ],
                exact_quotes=["Torque must be 10 Nm."],
            )
        ],
    )


def answer_response(
    *,
    evidence_id: str = "ev-1",
    support_evidence_id: str = "ev-1",
    quote: str = "Torque must be 10 Nm.",
    citation_page: int = 1,
) -> AnswerResponse:
    document_id = f"doc_{'a' * 64}"
    evidence = EvidencePassage(
        evidence_id=evidence_id,
        rank=1,
        score=0.9,
        document_id=document_id,
        filename="manual.pdf",
        title="Manual",
        chunk_id="chunk-1",
        chunk_order=1,
        page_start=1,
        page_end=1,
        section_heading="LIMITS",
        text="Torque must be 10 Nm.",
        character_count=21,
    )
    citation = Citation(
        number=1,
        evidence_id=evidence_id,
        score=0.9,
        document_id=document_id,
        title="Manual",
        filename="manual.pdf",
        page_start=citation_page,
        page_end=citation_page,
        section_heading="LIMITS",
        chunk_id="chunk-1",
    )
    claim = AnswerClaim(
        claim_id="claim-1",
        text="Torque must be 10 Nm.",
        supports=[
            ClaimSupport(
                claim_id="claim-1",
                evidence_id=support_evidence_id,
                citation_number=1,
                quote=quote,
            )
        ],
    )
    return AnswerResponse(
        question="What torque is required?",
        abstained=False,
        abstention_reason=None,
        explanation=None,
        answer="Torque must be 10 Nm.",
        claims=[claim],
        citations=[citation],
        evidence=[evidence],
        index_profile_id="index",
        corpus_fingerprint="corpus",
        requested_top_k=5,
        generation_format_version="1",
        provider=ProviderIdentity(name="fake", revision="offline", device="cpu"),
        context_character_count=21,
        prompt_token_count=10,
        output_token_count=5,
        latency_seconds=0.01,
        audit_id="audit",
        grounding=GroundingOutcome(
            claim_count=1,
            supported_claim_count=1,
            citation_count=1,
        ),
    )


@pytest.mark.parametrize(
    ("changes", "matched", "unsafe", "evidence_valid", "quote_valid"),
    [
        ({}, 1, False, 1, 1),
        ({"support_evidence_id": "fabricated"}, 0, True, 0, 0),
        ({"quote": "not present"}, 0, True, 1, 0),
        ({"citation_page": 2}, 0, True, 1, 1),
    ],
)
def test_observation_derives_grounding_from_production_objects(
    changes: dict[str, object],
    matched: int,
    unsafe: bool,
    evidence_valid: int,
    quote_valid: int,
) -> None:
    observation = build_answer_observation(
        answer_case(),
        answer_response(**changes),  # type: ignore[arg-type]
        {"src_manual": source()},
        ServiceTrace(retrieval_calls=1, embedding_calls=1, generation_calls=1),
    )

    assert observation.matched_claim_count == matched
    assert bool(observation.unsafe_categories) is unsafe
    assert observation.evidence_id_valid_count == evidence_valid
    assert observation.exact_quote_valid_count == quote_valid
    assert observation.provenance_valid_count == (changes.get("citation_page") != 2)


@pytest.mark.parametrize(
    ("fact", "text", "preserved"),
    [
        (NumericFact(fact_id="n1", value="10", unit="Nm"), "Torque is 10 Nm.", True),
        (NumericFact(fact_id="n2", value="10", unit="Nm"), "Torque is 12 Nm.", False),
        (
            RangeFact(
                fact_id="r1",
                lower="10",
                upper="12",
                lower_inclusive=True,
                upper_inclusive=True,
                unit="Nm",
            ),
            "Range is 10 to 12 Nm.",
            True,
        ),
        (
            ComparatorFact(fact_id="c1", comparator="at_most", value="10", unit="Nm"),
            "Use at most 10 Nm.",
            True,
        ),
        (
            NegationFact(fact_id="g1", proposition="install", negated=True),
            "Do not install.",
            True,
        ),
        (
            ModalityFact(fact_id="m1", modality="must", proposition="be inspected"),
            "It must be inspected.",
            True,
        ),
        (
            StatusFact(fact_id="s1", status="prohibited", subject="use"),
            "Use is prohibited.",
            True,
        ),
        (
            ApplicabilityFact(fact_id="a1", applicable=True, model="A"),
            "Applicable to model A.",
            True,
        ),
        (
            NumericFact(fact_id="n3", value="10", condition="when cold"),
            "Use 10 when cold.",
            True,
        ),
        (
            QualifierFact(
                fact_id="q1",
                qualifier="visual only",
                subject="inspection",
                affirmed=True,
            ),
            "Inspection is visual only.",
            True,
        ),
        (
            ModalityFact(fact_id="m2", modality="must", proposition="be inspected"),
            "It may be inspected.",
            False,
        ),
    ],
)
def test_semantic_fact_families_are_exact_and_deterministic(
    fact: StructuredFact,
    text: str,
    preserved: bool,
) -> None:
    assert fact_preserved(fact, text) is preserved


def test_structured_fact_schema_rejects_legacy_unknown_and_invalid_values() -> None:
    adapter: TypeAdapter[StructuredFact] = TypeAdapter(StructuredFact)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"fact_id": "legacy", "category": "numeric", "value": "10"}
        )
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"fact_type": "unknown", "fact_id": "unknown", "value": "10"}
        )
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "fact_type": "numeric",
                "fact_id": "bad-unit",
                "value": "10",
                "unit": "kg",
            }
        )
    with pytest.raises(ValidationError):
        RangeFact(
            fact_id="reversed",
            lower="12",
            upper="10",
            lower_inclusive=True,
            upper_inclusive=True,
        )


def test_structured_facts_reject_direction_unit_scope_and_modality_changes() -> None:
    assert not fact_preserved(
        ComparatorFact(fact_id="limit", comparator="at_most", value="10", unit="Nm"),
        "Use at least 10 Nm.",
    )
    assert not fact_preserved(
        RangeFact(
            fact_id="range",
            lower="5",
            upper="10",
            lower_inclusive=True,
            upper_inclusive=True,
            unit="mm",
        ),
        "The range is 5 to 10 cm.",
    )
    assert not fact_preserved(
        ModalityFact(fact_id="modal", modality="must", proposition="inspect"),
        "The technician may inspect.",
    )
    assert not fact_preserved(
        StatusFact(fact_id="status", status="approved", subject="repair"),
        "The repair is not approved.",
    )
    assert not fact_preserved(
        ApplicabilityFact(fact_id="scope", applicable=True, model="A", revision="2"),
        "This is applicable to model A revision 3.",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Applicable to model A.", "preserved"),
        ("Not applicable to model A.", "contradicted"),
        (
            "Applicable to model A, but not applicable to model A.",
            "contradicted",
        ),
        ("Applicable to model B.", "missing"),
        ("If applicable to model A, continue.", "ambiguous"),
        ('The phrase "applicable to model A" was quoted.', "ambiguous"),
        ("No applicability statement is present.", "missing"),
    ],
)
def test_applicability_polarity_and_assertion_context(
    text: str,
    expected: str,
) -> None:
    fact = ApplicabilityFact(fact_id="applicable", applicable=True, model="A")

    assert fact_disposition(fact, text) == expected


def test_applicability_requires_complete_structured_scope() -> None:
    fact = ApplicabilityFact(
        fact_id="scope",
        applicable=True,
        component="actuator",
        model="A",
        revision="R1",
    )

    assert fact_preserved(
        fact, "The actuator component is applicable to model A revision R1."
    )
    assert not fact_preserved(
        fact, "The pump component is applicable to model A revision R1."
    )
    assert not fact_preserved(
        fact, "The actuator component is applicable to model A revision R2."
    )


@pytest.mark.parametrize(
    ("fact", "text"),
    [
        (
            ApplicabilityFact(
                fact_id="component-model",
                applicable=True,
                component="pump",
                model="A",
            ),
            "Component A is applicable to model pump.",
        ),
        (
            ApplicabilityFact(
                fact_id="revision-configuration",
                applicable=True,
                revision="R1",
                configuration="standard",
            ),
            "Revision standard is applicable to configuration R1.",
        ),
        (
            ApplicabilityFact(
                fact_id="location-phase",
                applicable=True,
                location="left wing",
                procedure_phase="inspection",
            ),
            "At location inspection this is applicable during left wing.",
        ),
    ],
)
def test_applicability_cross_role_substitution_is_role_mismatch(
    fact: ApplicabilityFact,
    text: str,
) -> None:
    assert fact_disposition(fact, text) == "role_mismatch"
    assert not fact_preserved(fact, text)


def test_applicability_requires_one_coherent_role_bound_assertion() -> None:
    fact = ApplicabilityFact(
        fact_id="coherent",
        applicable=True,
        component="pump",
        model="A",
    )

    assert fact_preserved(fact, "The pump component is applicable to model A.")
    assert (
        fact_disposition(
            fact,
            "The pump component is applicable. Model A is discussed elsewhere.",
        )
        == "ambiguous"
    )
    assert fact_disposition(fact, "An A pump is applicable.") == "missing"


def test_applicability_component_model_revision_binding_is_preserved() -> None:
    fact = ApplicabilityFact(
        fact_id="complete",
        applicable=True,
        component="pump",
        model="A",
        revision="R1",
    )

    assert fact_preserved(
        fact,
        "The pump component is applicable to model A revision R1.",
    )


def test_applicability_polarity_is_scoped_to_role_bound_identity() -> None:
    model_a = ApplicabilityFact(
        fact_id="model-a",
        applicable=True,
        component="pump",
        model="A",
    )
    model_b = ApplicabilityFact(
        fact_id="model-b",
        applicable=False,
        component="pump",
        model="B",
    )
    text = "The pump component is applicable to model A, but not to model B."

    assert fact_preserved(model_a, text)
    assert fact_preserved(model_b, text)


def test_unrelated_negative_scope_does_not_negate_expected_scope() -> None:
    fact = ApplicabilityFact(
        fact_id="pump",
        applicable=True,
        component="pump",
        model="A",
    )
    text = (
        "The valve component is not applicable to model A, "
        "but the pump component is applicable to model A."
    )

    assert fact_preserved(fact, text)


def test_same_role_bound_scope_with_both_polarities_is_contradicted() -> None:
    fact = ApplicabilityFact(
        fact_id="contradicted",
        applicable=True,
        component="pump",
        model="A",
    )

    assert (
        fact_disposition(
            fact,
            "The pump component is applicable to model A, "
            "but the pump component is not applicable to model A.",
        )
        == "contradicted"
    )


@pytest.mark.parametrize(
    "text",
    [
        'The manual says "applicable to model A".',
        "The manual says 'applicable to model A'.",
        "The manual says “applicable to model A”.",
        "The manual says „applicable to model A”.",
        "The label reads «applicable to model A».",
        "The note recorded ‘applicable to model A’.",
        "The note recorded ‚applicable to model A’.",
    ],
)
def test_quoted_applicability_is_not_currently_asserted(text: str) -> None:
    fact = ApplicabilityFact(fact_id="quoted", applicable=True, model="A")

    assert fact_disposition(fact, text) == "ambiguous"


def test_word_apostrophe_is_not_a_quote_delimiter() -> None:
    fact = ApplicabilityFact(fact_id="apostrophe", applicable=True, model="A")

    assert fact_preserved(
        fact,
        "Manufacturer's current approved document confirms "
        "that it is applicable to model A.",
    )


def test_unbalanced_quote_fails_closed_and_separate_assertion_can_pass() -> None:
    fact = ApplicabilityFact(fact_id="quote-bounds", applicable=True, model="A")

    assert (
        fact_disposition(fact, 'The manual says "applicable to model A.') == "ambiguous"
    )
    assert fact_preserved(
        fact,
        "The manual says 'applicable to model A'. "
        "The current approved document confirms applicability to model A.",
    )


@pytest.mark.parametrize(
    "text",
    [
        "The previous manual said applicable to model A.",
        "An earlier revision stated that it was applicable to model A.",
        "According to the old procedure, it was applicable to model A.",
        "The technician claimed it was applicable to model A.",
        "The note indicates applicability to model A.",
    ],
)
def test_reported_applicability_is_not_currently_asserted(text: str) -> None:
    fact = ApplicabilityFact(fact_id="reported", applicable=True, model="A")

    assert fact_disposition(fact, text) == "ambiguous"


@pytest.mark.parametrize(
    "text",
    [
        "The manufacturer denied that it was applicable to model A.",
        'The statement "applicable to model A" was rejected.',
        "Applicability to model A was superseded.",
        "Applicability to model A was not confirmed.",
        "It is incorrect to state that it is applicable to model A.",
        "The applicability claim for model A was disputed.",
    ],
)
def test_rejected_applicability_is_contradicted(text: str) -> None:
    fact = ApplicabilityFact(fact_id="rejected", applicable=True, model="A")

    assert fact_disposition(fact, text) == "contradicted"


@pytest.mark.parametrize(
    "text",
    [
        "The current approved document confirms that it is applicable to model A.",
        "Based on the cited section, it is applicable to model A.",
        "The current revision confirms applicability to model A.",
        "The current instruction says it is applicable to model A.",
    ],
)
def test_explicit_current_adoption_is_asserted(text: str) -> None:
    fact = ApplicabilityFact(fact_id="adopted", applicable=True, model="A")

    assert fact_preserved(fact, text)


@pytest.mark.parametrize(
    "text",
    [
        (
            "The approved document confirms that the manual says "
            "the pump component is applicable to model A."
        ),
        (
            "The approved document confirms that the pump component "
            "manual says it is applicable to model A."
        ),
        (
            "The approved document confirms that model A is reported "
            "as applicable to the pump component."
        ),
        (
            "The current approved document confirms that the maintenance "
            "report states the pump component is applicable to model A."
        ),
        (
            "The current approved document confirms the claim that "
            "the pump component is applicable to model A."
        ),
        (
            "The current approved revision acknowledges the statement that "
            "the pump component is applicable to model A."
        ),
        (
            "The approved document confirms that the manual says "
            "'applicable to model A'."
        ),
        ("The document verifies that the label reads 'applicable to model A'."),
        ("The document verifies that the label reads applicable to model A."),
        (
            "The revision acknowledges that the old manual stated "
            "applicability to model A."
        ),
    ],
)
def test_adoption_of_reporting_event_does_not_adopt_applicability(
    text: str,
) -> None:
    fact = ApplicabilityFact(fact_id="nested-report", applicable=True, model="A")

    assert fact_disposition(fact, text) == "ambiguous"


def test_direct_adoption_binds_to_the_applicability_proposition() -> None:
    fact = ApplicabilityFact(
        fact_id="direct-adoption",
        applicable=True,
        component="pump",
        model="A",
    )

    assert fact_preserved(
        fact,
        "The approved document confirms that the pump component "
        "is applicable to model A.",
    )
    assert fact_preserved(
        fact,
        "The approved procedure states that the pump component "
        "is applicable to model A.",
    )


def test_adoption_is_isolated_by_sentence_and_clause() -> None:
    fact = ApplicabilityFact(
        fact_id="adoption-isolation",
        applicable=True,
        component="pump",
        model="A",
    )

    assert (
        fact_disposition(
            fact,
            "The document confirms the inspection interval. "
            "The manual says the pump component is applicable to model A.",
        )
        == "ambiguous"
    )
    assert fact_preserved(
        fact,
        "The old manual said the pump component was applicable to model B, "
        "but the current revision confirms that the pump component "
        "is applicable to model A.",
    )


def test_parent_adoption_does_not_override_nested_rejection() -> None:
    fact = ApplicabilityFact(
        fact_id="nested-rejection",
        applicable=True,
        model="A",
    )

    assert (
        fact_disposition(
            fact,
            "The document confirms that the manufacturer denied "
            "applicability to model A.",
        )
        == "contradicted"
    )


def test_historical_rejection_does_not_cancel_separate_current_assertion() -> None:
    fact = ApplicabilityFact(
        fact_id="isolated",
        applicable=True,
        component="pump",
        model="A",
    )

    assert fact_preserved(
        fact,
        "The previous manual denied applicability to model A. "
        "The current revision confirms that the pump component "
        "is applicable to model A.",
    )
    assert fact_preserved(
        fact,
        "The pump component is applicable to model A. "
        "The report also said the valve component was not applicable.",
    )


def test_reported_other_scope_does_not_affect_current_assertion() -> None:
    fact = ApplicabilityFact(fact_id="model-a", applicable=True, model="A")

    assert fact_preserved(
        fact,
        "The technician said it was applicable to model B. "
        "The current approved document confirms applicability to model A.",
    )


def test_negative_applicability_is_a_distinct_structured_fact() -> None:
    fact = ApplicabilityFact(
        fact_id="not-applicable",
        applicable=False,
        model="A",
    )

    assert fact_preserved(fact, "This is not applicable to model A.")
    assert not fact_preserved(fact, "This is applicable to model A.")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Inspection is visual only.", "preserved"),
        ("Inspection is not visual only.", "contradicted"),
        (
            "Inspection is visual only, but inspection is not visual only.",
            "contradicted",
        ),
        ("Repair is visual only.", "missing"),
        ("If inspection is visual only, continue.", "ambiguous"),
        ('"Inspection is visual only" was quoted.', "ambiguous"),
        ("Inspection criteria are absent.", "missing"),
    ],
)
def test_qualifier_polarity_scope_and_assertion_context(
    text: str,
    expected: str,
) -> None:
    fact = QualifierFact(
        fact_id="visual",
        qualifier="visual only",
        subject="inspection",
        affirmed=True,
    )

    assert fact_disposition(fact, text) == expected


def test_negative_qualifier_is_a_distinct_structured_fact() -> None:
    fact = QualifierFact(
        fact_id="not-visual",
        qualifier="visual only",
        subject="inspection",
        affirmed=False,
    )

    assert fact_preserved(fact, "Inspection is not visual only.")
    assert not fact_preserved(fact, "Inspection is visual only.")


@pytest.mark.parametrize(
    "text",
    [
        "The manual said inspection was 'visual only'.",
        "The manual said inspection was “visual only”.",
        "The technician claimed inspection was visual only.",
    ],
)
def test_quoted_or_reported_qualifier_is_not_asserted(text: str) -> None:
    fact = QualifierFact(
        fact_id="reported-qualifier",
        qualifier="visual only",
        subject="inspection",
        affirmed=True,
    )

    assert fact_disposition(fact, text) == "ambiguous"


def test_rejected_qualifier_is_contradicted_and_current_affirmation_passes() -> None:
    fact = QualifierFact(
        fact_id="qualifier-context",
        qualifier="visual only",
        subject="inspection",
        affirmed=True,
    )

    assert (
        fact_disposition(fact, "The inspection qualifier visual only was rejected.")
        == "contradicted"
    )
    assert fact_preserved(
        fact,
        "The current approved document confirms the inspection is visual only.",
    )


@pytest.mark.parametrize(
    "text",
    [
        (
            "The approved document confirms that the manual says "
            "inspection is visual only."
        ),
        ("The approved document confirms that the inspection report says visual only."),
        ("The approved document confirms the claim that inspection is visual only."),
        (
            "The approved document confirms that the manual says "
            "'inspection is visual only'."
        ),
        ("The document verifies that the label reads 'inspection is visual only'."),
        ("The document verifies that the label reads inspection is visual only."),
        ("The revision verifies that a technician claimed inspection was visual only."),
    ],
)
def test_adoption_of_reporting_event_does_not_adopt_qualifier(text: str) -> None:
    fact = QualifierFact(
        fact_id="nested-qualifier",
        qualifier="visual only",
        subject="inspection",
        affirmed=True,
    )

    assert fact_disposition(fact, text) == "ambiguous"


def test_direct_adoption_binds_to_the_qualifier_proposition() -> None:
    fact = QualifierFact(
        fact_id="direct-qualifier",
        qualifier="visual only",
        subject="inspection",
        affirmed=True,
    )

    assert fact_preserved(
        fact,
        "The approved document confirms that inspection is visual only.",
    )
    assert fact_preserved(
        fact,
        "Based on the cited section, inspection is visual only.",
    )


@pytest.mark.parametrize(
    "facts",
    [
        [
            ApplicabilityFact(fact_id="applicable", applicable=True, model="A"),
            ApplicabilityFact(fact_id="not-applicable", applicable=False, model="A"),
        ],
        [
            QualifierFact(
                fact_id="affirmed",
                qualifier="visual only",
                subject="inspection",
                affirmed=True,
            ),
            QualifierFact(
                fact_id="negated",
                qualifier="visual only",
                subject="inspection",
                affirmed=False,
            ),
        ],
    ],
)
def test_applicability_and_qualifier_contradictions_fail_dataset_claim(
    facts: list[StructuredFact],
) -> None:
    with pytest.raises(ValidationError, match="contradictory structured facts"):
        GoldClaim(
            claim_id="contradiction",
            support_targets=[
                PageTarget(source_id="src_manual", page_start=1, page_end=1)
            ],
            facts=facts,
        )


def test_applicability_and_qualifier_require_explicit_polarity() -> None:
    adapter: TypeAdapter[StructuredFact] = TypeAdapter(StructuredFact)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"fact_type": "applicability", "fact_id": "legacy", "model": "A"}
        )
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "fact_type": "qualifier",
                "fact_id": "legacy",
                "qualifier": "visual only",
                "subject": "inspection",
            }
        )


@pytest.mark.parametrize(
    ("fact", "claim_text", "family"),
    [
        (
            ApplicabilityFact(fact_id="applicable", applicable=True, model="A"),
            "Applicable to model A, but not applicable to model A.",
            "applicability",
        ),
        (
            QualifierFact(
                fact_id="qualifier",
                qualifier="visual only",
                subject="inspection",
                affirmed=True,
            ),
            "Inspection is not visual only.",
            "qualifier",
        ),
    ],
)
def test_contradicted_scoped_fact_reduces_production_observation_metric(
    fact: StructuredFact,
    claim_text: str,
    family: str,
) -> None:
    case = answer_case().model_copy(
        update={
            "gold_claims": [
                answer_case().gold_claims[0].model_copy(update={"facts": [fact]})
            ]
        }
    )
    response = answer_response()
    response = response.model_copy(
        update={"claims": [response.claims[0].model_copy(update={"text": claim_text})]}
    )
    observation = build_answer_observation(
        case,
        response,
        {"src_manual": source()},
        ServiceTrace(retrieval_calls=1, embedding_calls=1, generation_calls=1),
    )
    metric = metric_map(answer_metrics([observation]))[
        ("answer_quality", f"{family}_preservation")
    ]

    assert observation.fact_coverage[family] == FactCoverage(matched=0, total=1)
    assert observation.fact_dispositions[fact.fact_id] == "contradicted"
    assert metric.numerator == 0
    assert metric.denominator == 1
    assert metric.value == 0


def test_role_mismatch_reduces_applicability_metric_numerator() -> None:
    fact = ApplicabilityFact(
        fact_id="role-bound",
        applicable=True,
        component="pump",
        model="A",
    )
    case = answer_case().model_copy(
        update={
            "gold_claims": [
                answer_case().gold_claims[0].model_copy(update={"facts": [fact]})
            ]
        }
    )
    response = answer_response()
    response = response.model_copy(
        update={
            "claims": [
                response.claims[0].model_copy(
                    update={"text": "Component A is applicable to model pump."}
                )
            ]
        }
    )
    observation = build_answer_observation(
        case,
        response,
        {"src_manual": source()},
        ServiceTrace(retrieval_calls=1, embedding_calls=1, generation_calls=1),
    )
    metric = metric_map(answer_metrics([observation]))[
        ("answer_quality", "applicability_preservation")
    ]

    assert observation.fact_dispositions[fact.fact_id] == "role_mismatch"
    assert observation.fact_coverage["applicability"] == FactCoverage(
        matched=0, total=1
    )
    assert metric.numerator == 0
    assert metric.denominator == 1
    assert metric.value == 0


@pytest.mark.parametrize(
    ("claim_text", "disposition", "numerator"),
    [
        ("The manual said applicable to model A.", "ambiguous", 0),
        (
            "The approved document confirms that the manual says "
            "'applicable to model A'.",
            "ambiguous",
            0,
        ),
        (
            "The approved document confirms that the pump component "
            "manual says it is applicable to model A.",
            "ambiguous",
            0,
        ),
        ("Applicability to model A was rejected.", "contradicted", 0),
        (
            "The current approved document confirms applicability to model A.",
            "preserved",
            1,
        ),
    ],
)
def test_assertion_context_flows_into_applicability_metric(
    claim_text: str,
    disposition: str,
    numerator: int,
) -> None:
    fact = ApplicabilityFact(
        fact_id="assertion-context",
        applicable=True,
        model="A",
    )
    case = answer_case().model_copy(
        update={
            "gold_claims": [
                answer_case().gold_claims[0].model_copy(update={"facts": [fact]})
            ]
        }
    )
    response = answer_response()
    response = response.model_copy(
        update={"claims": [response.claims[0].model_copy(update={"text": claim_text})]}
    )
    observation = build_answer_observation(
        case,
        response,
        {"src_manual": source()},
        ServiceTrace(retrieval_calls=1, embedding_calls=1, generation_calls=1),
    )
    metric = metric_map(answer_metrics([observation]))[
        ("answer_quality", "applicability_preservation")
    ]

    assert observation.fact_dispositions[fact.fact_id] == disposition
    assert metric.numerator == numerator
    assert metric.denominator == 1
    assert metric.value == numerator


@pytest.mark.parametrize(
    ("claim_text", "disposition", "numerator"),
    [
        (
            "The approved document confirms that the manual says "
            "'inspection is visual only'.",
            "ambiguous",
            0,
        ),
        (
            "The approved document confirms that the inspection report "
            "says visual only.",
            "ambiguous",
            0,
        ),
        (
            "The approved document confirms that inspection is visual only.",
            "preserved",
            1,
        ),
    ],
)
def test_adoption_scope_flows_into_qualifier_metric(
    claim_text: str,
    disposition: str,
    numerator: int,
) -> None:
    fact = QualifierFact(
        fact_id="qualifier-adoption",
        qualifier="visual only",
        subject="inspection",
        affirmed=True,
    )
    case = answer_case().model_copy(
        update={
            "gold_claims": [
                answer_case().gold_claims[0].model_copy(update={"facts": [fact]})
            ]
        }
    )
    response = answer_response().model_copy(
        update={
            "claims": [
                answer_response().claims[0].model_copy(update={"text": claim_text})
            ]
        }
    )
    observation = build_answer_observation(
        case,
        response,
        {"src_manual": source()},
        ServiceTrace(retrieval_calls=1, embedding_calls=1, generation_calls=1),
    )
    metric = metric_map(answer_metrics([observation]))[
        ("answer_quality", "qualifier_preservation")
    ]

    assert observation.fact_dispositions[fact.fact_id] == disposition
    assert metric.numerator == numerator
    assert metric.denominator == 1
    assert metric.value == numerator


def conflict_case() -> AnswerBenchmarkCase:
    return AnswerBenchmarkCase(
        case_id="conflict-case",
        population="model_quality",
        suite="safety",
        kind="answer",
        question="Which limit applies?",
        expected_outcome="conflicting_evidence",
        document_source_ids=["src_manual"],
        conflict_targets=[
            PageTarget(source_id="src_manual", page_start=1, page_end=1, grade=2),
            PageTarget(source_id="src_manual", page_start=2, page_end=2, grade=2),
        ],
    )


def conflict_response(cited_ids: list[str]) -> AnswerResponse:
    document_id = f"doc_{'a' * 64}"
    evidence = [
        EvidencePassage(
            evidence_id=f"ev-{page}",
            rank=page,
            score=0.9,
            document_id=document_id,
            filename="manual.pdf",
            title="Manual",
            chunk_id=f"chunk-{page}",
            chunk_order=page,
            page_start=page,
            page_end=page,
            section_heading=f"LIMIT {page}",
            text=f"Conflicting limit {page}.",
            character_count=20,
        )
        for page in (1, 2, 3)
    ]
    by_id = {item.evidence_id: item for item in evidence}
    citations = [
        Citation(
            number=number,
            evidence_id=evidence_id,
            score=by_id[evidence_id].score,
            document_id=document_id,
            title="Manual",
            filename="manual.pdf",
            page_start=by_id[evidence_id].page_start,
            page_end=by_id[evidence_id].page_end,
            section_heading=by_id[evidence_id].section_heading,
            chunk_id=by_id[evidence_id].chunk_id,
        )
        for number, evidence_id in enumerate(cited_ids, start=1)
    ]
    return AnswerResponse(
        **{
            **answer_response().model_dump(),
            "abstained": True,
            "abstention_reason": "conflicting_evidence",
            "answer": None,
            "claims": [],
            "citations": citations,
            "evidence": evidence,
            "grounding": GroundingOutcome(citation_count=len(citations)),
        }
    )


def test_conflict_coverage_uses_only_cited_conflict_ids() -> None:
    sources = {"src_manual": source().model_copy(update={"page_count": 3})}
    complete = build_answer_observation(
        conflict_case(),
        conflict_response(["ev-1", "ev-2"]),
        sources,
        ServiceTrace(retrieval_calls=1),
    )
    unrelated = build_answer_observation(
        conflict_case(),
        conflict_response(["ev-3"]),
        sources,
        ServiceTrace(retrieval_calls=1),
    )

    assert complete.matched_conflict_target_count == 2
    assert complete.observed_conflict_evidence_valid is True
    assert unrelated.matched_conflict_target_count == 0
    assert unrelated.unrelated_conflict_id_count == 1
    assert unrelated.observed_conflict_evidence_valid is False
    records = metric_map(conflict_metrics([complete, unrelated]))
    assert records[("conflict_quality", "conflict_evidence_precision")].value == (
        pytest.approx(2 / 3)
    )
    assert records[("conflict_quality", "conflict_evidence_recall")].value == (
        pytest.approx(1 / 2)
    )


def test_duplicate_conflict_citation_does_not_inflate_coverage() -> None:
    observation = build_answer_observation(
        conflict_case(),
        conflict_response(["ev-1", "ev-1"]),
        {"src_manual": source().model_copy(update={"page_count": 3})},
        ServiceTrace(retrieval_calls=1),
    )

    assert observation.cited_conflict_id_count == 1
    assert observation.duplicate_conflict_id_count == 1
    assert observation.matched_conflict_target_count == 1
    assert observation.observed_conflict_evidence_valid is False


def test_answer_safety_conflict_and_injection_populations_are_distinct() -> None:
    observations = [
        AnswerObservation(
            case_id="safety",
            population="model_quality",
            execution_succeeded=True,
            expected_outcome="safety_boundary",
            observed_outcome="safety_boundary",
            expected_pre_retrieval_block=True,
        ),
        AnswerObservation(
            case_id="injection",
            population="model_quality",
            execution_succeeded=True,
            expected_outcome="prompt_injection",
            observed_outcome="prompt_injection",
            injection_evidence_quarantined=True,
            retrieval_called=True,
            embedding_called=True,
        ),
        AnswerObservation(
            case_id="conflict",
            population="model_quality",
            execution_succeeded=True,
            expected_outcome="conflicting_evidence",
            observed_outcome="conflicting_evidence",
            expected_conflict_target_count=2,
            matched_conflict_target_count=2,
            cited_conflict_id_count=2,
            valid_conflict_id_count=2,
            aligned_conflict_id_count=2,
            observed_conflict_evidence_valid=True,
        ),
    ]

    safety = metric_map(safety_metrics(observations))
    injection = metric_map(injection_metrics(observations))
    conflict = metric_map(conflict_metrics(observations))

    assert safety[("safety_quality", "safety_boundary_recall")].value == 1
    assert safety[("safety_quality", "blocked_before_retrieval_rate")].value == 1
    assert (
        injection[
            ("injection_quality", "injection_evidence_quarantine_correctness")
        ].value
        == 1
    )
    assert conflict[("conflict_quality", "conflict_recall")].value == 1
    assert conflict[("conflict_quality", "conflicting_evidence_id_coverage")].value == 1


def test_operational_failure_is_not_abstention_or_unsafe_answer() -> None:
    observation = AnswerObservation(
        case_id="failure",
        population="model_quality",
        execution_succeeded=False,
        expected_outcome="insufficient_evidence",
        observed_outcome="model_unavailable",
        failure_category="llm_model_unavailable",
    )
    records = metric_map(answer_metrics([observation]))

    assert records[("answer_quality", "abstention_recall")].value == 0
    assert records[("safety_quality", "unsafe_answer_count")].value == 0
    assert (
        records[
            (
                "answer_quality",
                "confusion_expected_abstention_observed_operational_failure",
            )
        ].value
        == 1
    )


def test_metric_record_rejects_contradictions_and_duplicate_identities() -> None:
    valid = MetricRecord(
        metric_id="mrr",
        population="retrieval_quality",
        numerator=1,
        denominator=2,
        value=0.5,
        eligible=True,
        aggregation="macro",
        eligible_case_count=2,
        total_population_case_count=2,
    )
    assert isclose(valid.value or 0, 0.5)
    for update in (
        {"value": 0.6},
        {"numerator": -1},
        {"numerator": 3},
        {"denominator": 0},
        {"value": float("nan")},
        {"value": float("inf")},
    ):
        with pytest.raises(ValidationError):
            MetricRecord.model_validate({**valid.model_dump(), **update})
    with pytest.raises(ValidationError):
        MetricRecord.model_validate(
            {
                **valid.model_dump(),
                "eligible": False,
                "value": 0.5,
                "exclusion_reason": "no_eligible_cases",
            }
        )
    with pytest.raises(ValueError, match="duplicate_metric_identity"):
        validate_unique_metrics([valid, valid])
    with pytest.raises(ValueError, match="unregistered_metric_identity"):
        count_record(
            "mrr",
            1,
            population="performance",
            eligible_case_count=1,
            total_population_case_count=1,
        )
    with pytest.raises(ValidationError, match="unregistered_metric_identity"):
        MetricRecord.model_validate(
            {**valid.model_dump(), "population": "safety_quality"}
        )
    with pytest.raises(ValidationError, match="unregistered_metric_identity"):
        GateThreshold(
            metric_id="mrr",
            population="safety_quality",
            operator="greater_than_or_equal",
            value=0.5,
            suites=("safety",),
        )


def test_incomplete_gold_support_recall_is_null() -> None:
    observation = AnswerObservation(
        case_id="partial",
        population="model_quality",
        execution_succeeded=True,
        expected_outcome="supported_answer",
        observed_outcome="supported_answer",
        complete_support_set=False,
        returned_claim_count=1,
        matched_claim_count=1,
        fact_coverage={"numeric": FactCoverage(matched=1, total=1)},
    )
    recall = metric_map(answer_metrics([observation]))[
        ("answer_quality", "claim_support_recall")
    ]
    assert recall.value is None
    assert recall.exclusion_reason == "gold_support_set_incomplete"


def test_not_selected_cases_remain_in_population_totals_only() -> None:
    selected = AnswerObservation(
        case_id="selected",
        population="model_quality",
        selected=True,
        execution_succeeded=True,
        expected_outcome="supported_answer",
        observed_outcome="supported_answer",
    )
    not_selected = AnswerObservation(
        case_id="not-selected",
        population="model_quality",
        selected=False,
        execution_succeeded=False,
        expected_outcome="safety_boundary",
        observed_outcome="other_operational_failure",
    )

    metric = metric_map(answer_metrics([selected, not_selected]))[
        ("answer_quality", "whole_outcome_accuracy")
    ]

    assert metric.numerator == 1
    assert metric.denominator == 1
    assert metric.eligible_case_count == 1
    assert metric.total_population_case_count == 2
