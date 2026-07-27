"""Strict, content-bound Milestone 6 benchmark bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from app.evaluation.security import (
    MAXIMUM_EVALUATION_DATASET_BYTES,
    MAXIMUM_JSONL_LINE_BYTES,
    AuxiliaryFileError,
    ByteBudget,
    SecureDirectory,
    decode_json,
    safe_identifier,
    safe_relative_path,
)
from app.evaluation.synthetic import SyntheticPdfSpec, identity, render_pdf
from app.ingestion.service import DocumentService

MAXIMUM_SOURCES = 64
MAXIMUM_CASES = 2_000
MAXIMUM_TARGETS = 64
MAXIMUM_DOCUMENT_FILTERS = 32
MAXIMUM_GOLD_CLAIMS = 8
MAXIMUM_FACTS = 32
MAXIMUM_QUOTES = 16
MAXIMUM_SUPPORT_TARGETS = 32
MAXIMUM_TAGS = 16
MAXIMUM_SOURCE_BYTES = 25 * 1024 * 1024

OperationalFault = Literal[
    "malformed_generation",
    "model_unavailable",
    "generation_timeout",
    "audit_failure",
    "retrieval_failure",
]


class SourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^src_[a-z0-9][a-z0-9_-]{2,63}$")
    kind: Literal["synthetic", "redistributable", "manifest_only"]
    title: str = Field(min_length=1, max_length=200)
    publisher: str = Field(min_length=1, max_length=120)
    reuse_status: Literal["benchmark_authored", "public_domain", "licensed", "unknown"]
    redistribution_permitted: bool
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str | None = Field(default=None, pattern=r"^doc_[0-9a-f]{64}$")
    processing_profile_id: str | None = Field(default=None, max_length=80)
    chunking_profile_id: str | None = Field(default=None, max_length=80)
    manifest_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    allowed_corpus_fingerprint: str | None = Field(default=None, max_length=80)
    specification: str | None = Field(default=None, max_length=200)
    file: str | None = Field(default=None, max_length=200)
    source_url: str | None = Field(default=None, max_length=500)
    provenance_notes: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_source_kind(self) -> SourceManifest:
        if self.kind == "manifest_only":
            if self.file is not None or self.specification is not None:
                raise ValueError("manifest-only sources cannot reference local files")
            required = (
                self.document_id,
                self.processing_profile_id,
                self.chunking_profile_id,
                self.manifest_fingerprint,
                self.allowed_corpus_fingerprint,
            )
            if any(value is None for value in required):
                raise ValueError("manifest-only source identities must be pinned")
        elif self.kind == "synthetic":
            if self.specification is None or self.file is not None:
                raise ValueError("synthetic sources require only a specification")
        elif self.file is None or self.specification is not None:
            raise ValueError("redistributable sources require only a file")
        if self.kind != "manifest_only" and not self.redistribution_permitted:
            raise ValueError("bundled sources must permit redistribution")
        return self


class PageTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["page"] = "page"
    source_id: str = Field(pattern=r"^src_[a-z0-9][a-z0-9_-]{2,63}$")
    page_start: int | None = Field(default=None, ge=1, le=10_000)
    page_end: int | None = Field(default=None, ge=1, le=10_000)
    section: str | None = Field(default=None, min_length=1, max_length=200)
    grade: int = Field(default=1, ge=1, le=3)

    @model_validator(mode="after")
    def validate_pages(self) -> PageTarget:
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must be supplied together")
        if self.section is None and self.page_start is None:
            raise ValueError("page target requires pages or a section")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must be at least page_start")
        return self


class ChunkTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["chunk"] = "chunk"
    source_id: str = Field(pattern=r"^src_[a-z0-9][a-z0-9_-]{2,63}$")
    chunk_id: str = Field(min_length=1, max_length=100)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str = Field(pattern=r"^doc_[0-9a-f]{64}$")
    processing_profile_id: str = Field(min_length=1, max_length=80)
    chunking_profile_id: str = Field(min_length=1, max_length=80)
    manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_fingerprint: str = Field(min_length=1, max_length=80)
    grade: int = Field(default=1, ge=1, le=3)


EvidenceTarget = Annotated[PageTarget | ChunkTarget, Field(discriminator="kind")]


class CaseBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    population: Literal["model_quality", "operational_fault", "framework_validation"]
    suite: Literal["retrieval", "answers", "safety", "performance", "complete"]
    tags: list[str] = Field(default_factory=list, max_length=MAXIMUM_TAGS)
    fault: OperationalFault | None = None

    @model_validator(mode="after")
    def validate_fault_population(self) -> CaseBase:
        if self.fault is not None and self.population != "framework_validation":
            raise ValueError("operational faults require framework validation")
        if self.population == "operational_fault":
            raise ValueError("legacy operational-fault population is not supported")
        if len(self.tags) != len(set(self.tags)) or any(
            not tag or len(tag) > 40 for tag in self.tags
        ):
            raise ValueError("tags must be unique and bounded")
        return self


class RetrievalBenchmarkCase(CaseBase):
    kind: Literal["retrieval"]
    query: str = Field(min_length=1, max_length=2_000)
    document_source_ids: list[str] | None = Field(
        default=None, max_length=MAXIMUM_DOCUMENT_FILTERS
    )
    targets: list[EvidenceTarget] = Field(
        default_factory=list, max_length=MAXIMUM_TARGETS
    )

    @model_validator(mode="after")
    def unique_retrieval_references(self) -> RetrievalBenchmarkCase:
        if self.document_source_ids is not None and len(
            self.document_source_ids
        ) != len(set(self.document_source_ids)):
            raise ValueError("duplicate document source filter")
        if len([_target_key(target) for target in self.targets]) != len(
            set(_target_key(target) for target in self.targets)
        ):
            raise ValueError("duplicate retrieval target")
        return self


FactId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")]
BoundedFactText = Annotated[str, Field(min_length=1, max_length=200)]
DecimalText = Annotated[
    str,
    Field(pattern=r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$"),
]
Comparator = Literal[
    "less_than",
    "at_most",
    "greater_than",
    "at_least",
    "equal",
    "not_equal",
]
CanonicalUnit = Literal["N", "Nm", "mm", "cm", "in", "psi", "kPa", "°C", "°F", "%"]
UNIT_ALIASES: dict[str, CanonicalUnit] = {
    "n": "N",
    "newton": "N",
    "newtons": "N",
    "nm": "Nm",
    "n·m": "Nm",
    "n-m": "Nm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "in": "in",
    "inch": "in",
    "inches": "in",
    "psi": "psi",
    "kpa": "kPa",
    "°c": "°C",
    "degc": "°C",
    "°f": "°F",
    "degf": "°F",
    "%": "%",
}


class FactBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: FactId


class NumericFact(FactBase):
    fact_type: Literal["numeric"] = "numeric"
    value: DecimalText
    tolerance: DecimalText | None = None
    unit: CanonicalUnit | None = None
    comparator: Comparator | None = None
    scope: BoundedFactText | None = None
    condition: BoundedFactText | None = None

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit(cls, value: object) -> object:
        return _canonical_unit(value)

    @model_validator(mode="after")
    def validate_numeric(self) -> NumericFact:
        _finite_decimal(self.value)
        if self.tolerance is not None and _finite_decimal(self.tolerance) < 0:
            raise ValueError("numeric tolerance cannot be negative")
        return self


class RangeFact(FactBase):
    fact_type: Literal["range"] = "range"
    lower: DecimalText
    upper: DecimalText
    lower_inclusive: bool
    upper_inclusive: bool
    unit: CanonicalUnit | None = None
    scope: BoundedFactText | None = None
    condition: BoundedFactText | None = None

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit(cls, value: object) -> object:
        return _canonical_unit(value)

    @model_validator(mode="after")
    def validate_range(self) -> RangeFact:
        if _finite_decimal(self.lower) > _finite_decimal(self.upper):
            raise ValueError("range lower bound exceeds upper bound")
        return self


class ComparatorFact(FactBase):
    fact_type: Literal["comparator"] = "comparator"
    comparator: Comparator
    value: DecimalText
    unit: CanonicalUnit | None = None
    scope: BoundedFactText | None = None
    condition: BoundedFactText | None = None

    @field_validator("unit", mode="before")
    @classmethod
    def canonicalize_unit(cls, value: object) -> object:
        return _canonical_unit(value)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        _finite_decimal(value)
        return value


class ModalityFact(FactBase):
    fact_type: Literal["modality"] = "modality"
    modality: Literal["must", "must_not", "should", "should_not", "may", "may_not"]
    proposition: BoundedFactText
    scope: BoundedFactText | None = None


class StatusFact(FactBase):
    fact_type: Literal["status"] = "status"
    status: Literal[
        "required",
        "prohibited",
        "approved",
        "unapproved",
        "optional",
        "unavailable",
        "applicable",
        "not_applicable",
    ]
    subject: BoundedFactText
    scope: BoundedFactText | None = None


class NegationFact(FactBase):
    fact_type: Literal["negation"] = "negation"
    proposition: BoundedFactText
    negated: bool


class ApplicabilityFact(FactBase):
    fact_type: Literal["applicability"] = "applicability"
    applicable: bool
    component: BoundedFactText | None = None
    model: BoundedFactText | None = None
    revision: BoundedFactText | None = None
    configuration: BoundedFactText | None = None
    operating_condition: BoundedFactText | None = None
    location: BoundedFactText | None = None
    procedure_phase: BoundedFactText | None = None

    @model_validator(mode="after")
    def require_scope(self) -> ApplicabilityFact:
        if not any(
            (
                self.component,
                self.model,
                self.revision,
                self.configuration,
                self.operating_condition,
                self.location,
                self.procedure_phase,
            )
        ):
            raise ValueError("applicability requires at least one scope field")
        return self


class QualifierFact(FactBase):
    fact_type: Literal["qualifier"] = "qualifier"
    qualifier: BoundedFactText
    subject: BoundedFactText
    affirmed: bool


StructuredFact = Annotated[
    NumericFact
    | RangeFact
    | ComparatorFact
    | ModalityFact
    | StatusFact
    | NegationFact
    | ApplicabilityFact
    | QualifierFact,
    Field(discriminator="fact_type"),
]


def _canonical_unit(value: object) -> object:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("unit must be a string")
    try:
        return UNIT_ALIASES[value.strip().casefold()]
    except KeyError as error:
        raise ValueError("unknown unit") from error


def _finite_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid decimal") from error
    if not parsed.is_finite():
        raise ValueError("decimal must be finite")
    return parsed


def _facts_contradict(facts: list[StructuredFact]) -> bool:
    identities: dict[tuple[object, ...], object] = {}
    for fact in facts:
        identity: tuple[object, ...]
        if isinstance(fact, NumericFact):
            identity = ("numeric", fact.scope, fact.condition, fact.unit)
            value: object = (fact.value, fact.tolerance, fact.comparator)
        elif isinstance(fact, RangeFact):
            identity = ("range", fact.scope, fact.condition, fact.unit)
            value = (
                fact.lower,
                fact.upper,
                fact.lower_inclusive,
                fact.upper_inclusive,
            )
        elif isinstance(fact, ComparatorFact):
            identity = (
                "comparator",
                fact.scope,
                fact.condition,
                fact.value,
                fact.unit,
            )
            value = fact.comparator
        elif isinstance(fact, ModalityFact):
            identity = ("modality", fact.proposition.casefold(), fact.scope)
            value = fact.modality
        elif isinstance(fact, StatusFact):
            identity = ("status", fact.subject.casefold(), fact.scope)
            value = fact.status
        elif isinstance(fact, NegationFact):
            identity = ("negation", fact.proposition.casefold())
            value = fact.negated
        elif isinstance(fact, ApplicabilityFact):
            identity = (
                "applicability",
                _fact_text_identity(fact.component),
                _fact_text_identity(fact.model),
                _fact_text_identity(fact.revision),
                _fact_text_identity(fact.configuration),
                _fact_text_identity(fact.operating_condition),
                _fact_text_identity(fact.location),
                _fact_text_identity(fact.procedure_phase),
            )
            value = fact.applicable
        elif isinstance(fact, QualifierFact):
            identity = (
                "qualifier",
                _fact_text_identity(fact.subject),
                _fact_text_identity(fact.qualifier),
            )
            value = fact.affirmed
        else:
            continue
        existing = identities.get(identity)
        if existing is not None and existing != value:
            return True
        identities[identity] = value
    return False


def _fact_text_identity(value: str | None) -> str | None:
    return None if value is None else " ".join(value.casefold().split())


class GoldClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    support_targets: list[EvidenceTarget] = Field(
        min_length=1, max_length=MAXIMUM_SUPPORT_TARGETS
    )
    facts: list[StructuredFact] = Field(default_factory=list, max_length=MAXIMUM_FACTS)
    exact_quotes: list[str] = Field(default_factory=list, max_length=MAXIMUM_QUOTES)
    shared_support_group: str | None = Field(
        default=None, pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$"
    )
    support_optional: bool = False
    required: bool = True

    @model_validator(mode="after")
    def unique_claim_content(self) -> GoldClaim:
        fact_ids = [fact.fact_id for fact in self.facts]
        fact_values = [
            json.dumps(
                fact.model_dump(mode="json", exclude={"fact_id"}),
                sort_keys=True,
                separators=(",", ":"),
            )
            for fact in self.facts
        ]
        targets = [_target_key(target) for target in self.support_targets]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("duplicate fact_id")
        if len(fact_values) != len(set(fact_values)):
            raise ValueError("duplicate structured fact")
        if _facts_contradict(self.facts):
            raise ValueError("contradictory structured facts")
        if len(targets) != len(set(targets)):
            raise ValueError("duplicate support target")
        if len(self.exact_quotes) != len(set(self.exact_quotes)) or any(
            not quote or len(quote) > 500 for quote in self.exact_quotes
        ):
            raise ValueError("quotes must be unique and bounded")
        return self


class AnswerBenchmarkCase(CaseBase):
    kind: Literal["answer"]
    question: str = Field(min_length=1, max_length=2_000)
    expected_outcome: Literal[
        "supported_answer",
        "insufficient_evidence",
        "conflicting_evidence",
        "safety_boundary",
        "prompt_injection",
    ]
    gold_claims: list[GoldClaim] = Field(
        default_factory=list, max_length=MAXIMUM_GOLD_CLAIMS
    )
    document_source_ids: list[str] | None = Field(
        default=None, max_length=MAXIMUM_DOCUMENT_FILTERS
    )
    conflict_targets: list[EvidenceTarget] = Field(
        default_factory=list, max_length=MAXIMUM_TARGETS
    )
    block_before_retrieval: bool = False
    safety_false_positive_control: bool = False
    injection_false_positive_control: bool = False
    complete_support_set: bool = False
    prohibited_conclusions: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_answer_expectation(self) -> AnswerBenchmarkCase:
        if self.expected_outcome == "supported_answer" and not self.gold_claims:
            raise ValueError("supported answers require gold claims")
        if self.expected_outcome != "supported_answer" and self.gold_claims:
            raise ValueError("abstention cases cannot declare gold claims")
        if self.expected_outcome == "conflicting_evidence":
            if len(self.conflict_targets) < 2:
                raise ValueError("conflict cases require at least two targets")
        elif self.conflict_targets:
            raise ValueError("conflict targets require a conflict outcome")
        if self.block_before_retrieval and self.expected_outcome != "safety_boundary":
            raise ValueError("pre-retrieval blocks require a safety outcome")
        if (
            self.safety_false_positive_control or self.injection_false_positive_control
        ) and self.expected_outcome != "supported_answer":
            raise ValueError("false-positive controls require a supported answer")
        if self.document_source_ids is not None and len(
            self.document_source_ids
        ) != len(set(self.document_source_ids)):
            raise ValueError("duplicate document source filter")
        claim_ids = [claim.claim_id for claim in self.gold_claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("duplicate claim_id")
        fact_ids = [fact.fact_id for claim in self.gold_claims for fact in claim.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("duplicate fact_id")
        if any(
            not conclusion or len(conclusion) > 500
            for conclusion in self.prohibited_conclusions
        ):
            raise ValueError("prohibited conclusions must be bounded")
        groups: dict[str, list[GoldClaim]] = {}
        for claim in self.gold_claims:
            if claim.shared_support_group is not None:
                groups.setdefault(claim.shared_support_group, []).append(claim)
        for claims in groups.values():
            if len(claims) < 2:
                raise ValueError("shared support groups require at least two claims")
            common = set(_target_key(item) for item in claims[0].support_targets)
            for claim in claims[1:]:
                common &= {_target_key(item) for item in claim.support_targets}
            if not common:
                raise ValueError("shared support groups require a common target")
        return self


BenchmarkCase = Annotated[
    RetrievalBenchmarkCase | AnswerBenchmarkCase,
    Field(discriminator="kind"),
]
CASE_ADAPTER: TypeAdapter[BenchmarkCase] = TypeAdapter(BenchmarkCase)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    dataset_version: str = Field(pattern=r"^[0-9]+$")
    schema_version: Literal["1"]
    evaluation_profile_id: str = Field(min_length=1, max_length=80)
    cases_file: str = Field(min_length=1, max_length=200)
    thresholds_file: str | None = Field(default=None, max_length=200)
    sources: list[SourceManifest] = Field(min_length=1, max_length=MAXIMUM_SOURCES)

    @model_validator(mode="after")
    def unique_sources(self) -> DatasetManifest:
        identifiers = [source.source_id for source in self.sources]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate source_id")
        return self

    @field_validator("cases_file", "thresholds_file")
    @classmethod
    def validate_bundle_reference(cls, value: str | None) -> str | None:
        if value is not None:
            safe_relative_path(value)
        return value


class VerifiedSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    kind: Literal["synthetic", "redistributable", "manifest_only"]
    source_fingerprint: str
    specification_fingerprint: str | None
    document_id: str
    processing_profile_id: str | None
    chunking_profile_id: str | None
    manifest_fingerprint: str | None
    allowed_corpus_fingerprint: str | None
    page_count: int | None
    relative_file: str | None = None
    content: bytes | None = Field(default=None, exclude=True)


class BenchmarkDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: DatasetManifest
    cases: list[BenchmarkCase]
    sources: list[VerifiedSource]
    fingerprint: str


class DatasetPreflight(BaseModel):
    """Side-effect-free metadata needed to select an evaluation service graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: DatasetManifest
    cases: list[BenchmarkCase]
    source_mode: Literal["bundled", "manifest_only"]
    input_bytes: int = Field(ge=0, le=MAXIMUM_EVALUATION_DATASET_BYTES)


def preflight_dataset_by_id(
    dataset_root: Path,
    dataset_id: str,
) -> DatasetPreflight:
    """Validate manifest/cases and classify sources without resolving content."""
    safe_identifier(dataset_id)
    with SecureDirectory(dataset_root) as root:
        with root.open_directory(dataset_id) as bundle:
            budget = ByteBudget()
            try:
                manifest = DatasetManifest.model_validate(
                    decode_json(bundle.read("dataset.json", budget=budget))
                )
                if manifest.dataset_id != dataset_id:
                    raise AuxiliaryFileError("dataset_id_mismatch")
                source_kinds = {source.kind for source in manifest.sources}
                if "manifest_only" in source_kinds and len(source_kinds) > 1:
                    raise AuxiliaryFileError("mixed_source_modes_unsupported")
                cases = _decode_cases(bundle.read(manifest.cases_file, budget=budget))
            except AuxiliaryFileError:
                raise
            except ValueError as error:
                raise AuxiliaryFileError("dataset_validation_failed") from error
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise AuxiliaryFileError("duplicate_case_id")
    source_ids = {source.source_id for source in manifest.sources}
    if any(not _case_source_ids(case) <= source_ids for case in cases):
        raise AuxiliaryFileError("unknown_source_id")
    return DatasetPreflight(
        manifest=manifest,
        cases=cases,
        source_mode=(
            "manifest_only" if source_kinds == {"manifest_only"} else "bundled"
        ),
        input_bytes=budget.used_bytes,
    )


def load_dataset_by_id(
    dataset_root: Path,
    dataset_id: str,
    *,
    documents: DocumentService | None = None,
    reserved_input_bytes: int = 0,
) -> BenchmarkDataset:
    """Resolve a safe dataset ID below the configured root."""
    safe_identifier(dataset_id)
    with SecureDirectory(dataset_root) as root:
        with root.open_directory(dataset_id) as bundle:
            return _load_dataset(
                bundle,
                documents=documents,
                selected_id=dataset_id,
                reserved_input_bytes=reserved_input_bytes,
            )


def load_dataset(
    root: Path,
    manifest_name: str = "dataset.json",
    *,
    documents: DocumentService | None = None,
) -> BenchmarkDataset:
    """Compatibility loader for one already-selected bundle directory."""
    if manifest_name != "dataset.json":
        raise AuxiliaryFileError("invalid_dataset_manifest_name")
    with SecureDirectory(root) as bundle:
        return _load_dataset(bundle, documents=documents, selected_id=None)


def _load_dataset(
    bundle: SecureDirectory,
    *,
    documents: DocumentService | None,
    selected_id: str | None,
    reserved_input_bytes: int = 0,
) -> BenchmarkDataset:
    if not 0 <= reserved_input_bytes < MAXIMUM_EVALUATION_DATASET_BYTES:
        raise AuxiliaryFileError("dataset_too_large")
    budget = ByteBudget(MAXIMUM_EVALUATION_DATASET_BYTES - reserved_input_bytes)
    try:
        manifest = DatasetManifest.model_validate(
            decode_json(bundle.read("dataset.json", budget=budget))
        )
        if selected_id is not None and manifest.dataset_id != selected_id:
            raise AuxiliaryFileError("dataset_id_mismatch")
        source_kinds = {source.kind for source in manifest.sources}
        if "manifest_only" in source_kinds and len(source_kinds) > 1:
            raise AuxiliaryFileError("mixed_source_modes_unsupported")
        cases_bytes = bundle.read(manifest.cases_file, budget=budget)
        cases = _decode_cases(cases_bytes)
        verified = [
            _verify_source(bundle, source, documents=documents, budget=budget)
            for source in manifest.sources
        ]
    except AuxiliaryFileError:
        raise
    except ValueError as error:
        raise AuxiliaryFileError("dataset_validation_failed") from error
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise AuxiliaryFileError("duplicate_case_id")
    source_map = {source.source_id: source for source in verified}
    for case in cases:
        referenced = _case_source_ids(case)
        if not referenced <= set(source_map):
            raise AuxiliaryFileError("unknown_source_id")
        for target in _case_targets(case):
            source = source_map[target.source_id]
            if (
                isinstance(target, PageTarget)
                and source.page_count is not None
                and target.page_end is not None
                and target.page_end > source.page_count
            ):
                raise AuxiliaryFileError("page_target_out_of_range")
            if isinstance(target, ChunkTarget):
                _validate_declared_chunk_source(target, source)
    fingerprint_payload = {
        "manifest": manifest.model_dump(mode="json"),
        "cases": [case.model_dump(mode="json") for case in cases],
        "verified_sources": [
            source.model_dump(mode="json", exclude={"content"}) for source in verified
        ],
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return BenchmarkDataset(
        manifest=manifest,
        cases=cases,
        sources=verified,
        fingerprint=digest,
    )


def _decode_cases(raw: bytes) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        if len(line) > MAXIMUM_JSONL_LINE_BYTES:
            raise AuxiliaryFileError("jsonl_line_too_large")
        cases.append(CASE_ADAPTER.validate_python(decode_json(line)))
        if len(cases) > MAXIMUM_CASES:
            raise AuxiliaryFileError("too_many_cases")
    if not cases:
        raise AuxiliaryFileError("empty_case_file")
    return cases


def _verify_source(
    bundle: SecureDirectory,
    source: SourceManifest,
    *,
    documents: DocumentService | None,
    budget: ByteBudget,
) -> VerifiedSource:
    if source.kind == "synthetic":
        assert source.specification is not None
        specification_bytes = bundle.read(source.specification, budget=budget)
        spec = SyntheticPdfSpec.model_validate(decode_json(specification_bytes))
        if spec.source_id != source.source_id:
            raise AuxiliaryFileError("synthetic_source_id_mismatch")
        pdf = render_pdf(spec)
        rendered = identity(spec)
        if rendered.sha256 != source.source_fingerprint:
            raise AuxiliaryFileError("source_fingerprint_mismatch")
        if (
            source.document_id is not None
            and source.document_id != rendered.document_id
        ):
            raise AuxiliaryFileError("document_id_mismatch")
        return VerifiedSource(
            source_id=source.source_id,
            kind=source.kind,
            source_fingerprint=rendered.sha256,
            specification_fingerprint=hashlib.sha256(specification_bytes).hexdigest(),
            document_id=rendered.document_id,
            processing_profile_id=source.processing_profile_id,
            chunking_profile_id=source.chunking_profile_id,
            manifest_fingerprint=source.manifest_fingerprint,
            allowed_corpus_fingerprint=source.allowed_corpus_fingerprint,
            page_count=rendered.page_count,
            relative_file=None,
            content=pdf,
        )
    if source.kind == "redistributable":
        assert source.file is not None
        digest, _ = bundle.sha256(
            source.file,
            maximum_bytes=MAXIMUM_SOURCE_BYTES,
            budget=budget,
        )
        if digest != source.source_fingerprint:
            raise AuxiliaryFileError("source_fingerprint_mismatch")
        document_id = f"doc_{digest}"
        if source.document_id is not None and source.document_id != document_id:
            raise AuxiliaryFileError("document_id_mismatch")
        return VerifiedSource(
            source_id=source.source_id,
            kind=source.kind,
            source_fingerprint=digest,
            specification_fingerprint=None,
            document_id=document_id,
            processing_profile_id=source.processing_profile_id,
            chunking_profile_id=source.chunking_profile_id,
            manifest_fingerprint=source.manifest_fingerprint,
            allowed_corpus_fingerprint=source.allowed_corpus_fingerprint,
            page_count=None,
            relative_file=source.file,
            content=None,
        )
    if documents is None:
        raise AuxiliaryFileError("manifest_source_requires_document_service")
    assert source.document_id is not None
    metadata = documents.metadata(source.document_id)
    manifest = documents.store.validate_document(source.document_id).manifest
    manifest_fingerprint = documents.store.manifest_sha256(source.document_id)
    if (
        metadata.sha256 != source.source_fingerprint
        or manifest.processing_profile_id != source.processing_profile_id
        or manifest.chunking_format_version != source.chunking_profile_id
        or manifest_fingerprint != source.manifest_fingerprint
    ):
        raise AuxiliaryFileError("manifest_source_identity_mismatch")
    return VerifiedSource(
        source_id=source.source_id,
        kind=source.kind,
        source_fingerprint=metadata.sha256,
        specification_fingerprint=None,
        document_id=metadata.document_id,
        processing_profile_id=manifest.processing_profile_id,
        chunking_profile_id=manifest.chunking_format_version,
        manifest_fingerprint=manifest_fingerprint,
        allowed_corpus_fingerprint=source.allowed_corpus_fingerprint,
        page_count=manifest.page_count,
        relative_file=None,
    )


@dataclass(frozen=True)
class ResolvedDocument:
    manifest_fingerprint: str
    page_count: int
    chunk_ids: set[str]
    section_occurrences: tuple[SectionOccurrence, ...]
    processing_profile_id: str
    chunking_profile_id: str


@dataclass(frozen=True)
class SectionOccurrence:
    heading: str
    page_start: int
    page_end: int
    occurrence_id: str


def validate_resolved_targets(
    dataset: BenchmarkDataset,
    *,
    document_details: dict[str, ResolvedDocument],
    corpus_fingerprint: str,
) -> None:
    """Resolve page/chunk targets against ingested production artifacts."""
    sources = {source.source_id: source for source in dataset.sources}
    for case in dataset.cases:
        for target in _case_targets(case):
            source = sources[target.source_id]
            details = document_details.get(source.source_id)
            if details is None:
                raise AuxiliaryFileError("source_not_ingested")
            if isinstance(target, PageTarget):
                if target.page_end is not None and target.page_end > details.page_count:
                    raise AuxiliaryFileError("page_target_out_of_range")
                if target.section is not None:
                    heading = _normalize_heading(target.section)
                    page_start = target.page_start
                    page_end = target.page_end
                    matches = [
                        occurrence
                        for occurrence in details.section_occurrences
                        if occurrence.heading == heading
                        and (
                            page_start is None
                            or (
                                page_end is not None
                                and occurrence.page_start <= page_end
                                and occurrence.page_end >= page_start
                            )
                        )
                    ]
                    if not matches:
                        raise AuxiliaryFileError("section_target_not_found")
                    if len(matches) != 1:
                        raise AuxiliaryFileError("section_target_ambiguous")
            else:
                validate_chunk_target(
                    target,
                    source_sha256=source.source_fingerprint,
                    document_id=source.document_id,
                    processing_profile_id=details.processing_profile_id,
                    chunking_profile_id=details.chunking_profile_id,
                    manifest_fingerprint=details.manifest_fingerprint,
                    corpus_fingerprint=corpus_fingerprint,
                    known_chunk_ids=details.chunk_ids,
                )


def validate_chunk_target(
    target: ChunkTarget,
    *,
    source_sha256: str,
    document_id: str,
    processing_profile_id: str,
    chunking_profile_id: str,
    manifest_fingerprint: str,
    corpus_fingerprint: str,
    known_chunk_ids: set[str],
) -> None:
    """Reject non-portable pinned chunk targets on any identity mismatch."""
    actual = (
        source_sha256,
        document_id,
        processing_profile_id,
        chunking_profile_id,
        manifest_fingerprint,
        corpus_fingerprint,
    )
    expected = (
        target.source_sha256,
        target.document_id,
        target.processing_profile_id,
        target.chunking_profile_id,
        target.manifest_fingerprint,
        target.corpus_fingerprint,
    )
    if actual != expected or target.chunk_id not in known_chunk_ids:
        raise AuxiliaryFileError("chunk_target_profile_mismatch")


def _validate_declared_chunk_source(
    target: ChunkTarget, source: VerifiedSource
) -> None:
    if (
        target.source_sha256 != source.source_fingerprint
        or target.document_id != source.document_id
        or (
            source.processing_profile_id is not None
            and target.processing_profile_id != source.processing_profile_id
        )
        or (
            source.manifest_fingerprint is not None
            and target.manifest_fingerprint != source.manifest_fingerprint
        )
    ):
        raise AuxiliaryFileError("chunk_target_profile_mismatch")


def _case_source_ids(case: BenchmarkCase) -> set[str]:
    references = {target.source_id for target in _case_targets(case)}
    if isinstance(case, RetrievalBenchmarkCase):
        references.update(case.document_source_ids or [])
    else:
        references.update(case.document_source_ids or [])
    return references


def _case_targets(case: BenchmarkCase) -> list[EvidenceTarget]:
    if isinstance(case, RetrievalBenchmarkCase):
        return list(case.targets)
    return [
        *[target for claim in case.gold_claims for target in claim.support_targets],
        *case.conflict_targets,
    ]


def _target_key(target: EvidenceTarget) -> str:
    return json.dumps(
        target.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def normalize_section_heading(value: str) -> str:
    """Normalize headings by Unicode-safe whitespace folding and case folding."""
    return " ".join(value.split()).casefold()


_normalize_heading = normalize_section_heading
