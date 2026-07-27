"""Production-path tests for Milestone 6 Phase 1 trust boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from app.core.config import (
    PINNED_EMBEDDING_MODEL_NAME,
    PINNED_EMBEDDING_MODEL_REVISION,
    PINNED_GENERATION_MODEL_NAME,
    PINNED_GENERATION_MODEL_REVISION,
    Settings,
)
from app.evaluation.__main__ import (
    EXIT_INVALID,
    EXIT_OPERATIONAL,
    EXIT_REPORT,
    EXIT_SUCCESS,
    main,
)
from app.evaluation.benchmark import (
    MAXIMUM_CASES,
    MAXIMUM_TAGS,
    RetrievalBenchmarkCase,
    load_dataset,
    load_dataset_by_id,
)
from app.evaluation.offline import (
    NetworkDeniedError,
    network_denial_guard,
    real_mode_settings,
)
from app.evaluation.orchestrator import run_evaluation, validate_bundle
from app.evaluation.providers import OfflineEmbeddingProvider, OfflineLlmProvider
from app.evaluation.reporting import compare_reports, load_thresholds
from app.evaluation.reproducibility import verify_synthetic_reproducibility
from app.evaluation.security import (
    AuxiliaryFileError,
    SecureDirectory,
    safe_relative_path,
)
from app.generation.audit import GenerationAuditStore
from app.ingestion.service import DocumentService
from app.ingestion.storage import DocumentStore
from app.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from app.retrieval.models import ModelIdentity
from app.retrieval.storage import IndexStore
from app.services.runtime import create_services

FIXTURE = Path("data/evaluation/milestone-6")


class ZeroOffsetTimezone(tzinfo):
    def utcoffset(self, _value: datetime | None) -> timedelta:
        return timedelta(0)

    def dst(self, _value: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, _value: datetime | None) -> str:
        return "CUSTOM-UTC"


class MissingOffsetTimezone(tzinfo):
    def utcoffset(self, _value: datetime | None) -> None:
        return None

    def dst(self, _value: datetime | None) -> None:
        return None

    def tzname(self, _value: datetime | None) -> str:
        return "MISSING-OFFSET"


class NonZeroOffsetTimezone(tzinfo):
    def utcoffset(self, _value: datetime | None) -> timedelta:
        return timedelta(hours=2)

    def dst(self, _value: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, _value: datetime | None) -> str:
        return "CUSTOM-PLUS-TWO"


class ProductionIdentityOfflineEmbeddingProvider(OfflineEmbeddingProvider):
    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(
            name=PINNED_EMBEDDING_MODEL_NAME,
            revision=PINNED_EMBEDDING_MODEL_REVISION,
            dimension=8,
            maximum_sequence_length=512,
        )


def copy_bundle(tmp_path: Path, dataset_id: str = "bundle") -> Path:
    root = tmp_path / "datasets"
    shutil.copytree(FIXTURE, root / dataset_id)
    manifest = root / dataset_id / "dataset.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["dataset_id"] = dataset_id
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return root


def remove_default_threshold(root: Path, dataset_id: str) -> None:
    manifest_path = root / dataset_id / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["thresholds_file"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def isolated_settings(tmp_path: Path, dataset_root: Path) -> Settings:
    return Settings().model_copy(
        update={
            "document_storage_root": tmp_path / "documents",
            "index_runtime_root": tmp_path / "indexes",
            "generation_audit_root": tmp_path / "audits",
        }
    )


def configure_cli_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_root: Path,
) -> None:
    values = {
        "AERODOCS_DOCUMENT_STORAGE_ROOT": tmp_path / "documents",
        "AERODOCS_INDEX_RUNTIME_ROOT": tmp_path / "indexes",
        "AERODOCS_GENERATION_AUDIT_ROOT": tmp_path / "audits",
        "AERODOCS_EVALUATION_DATASET_ROOT": dataset_root,
        "AERODOCS_EVALUATION_REPORT_ROOT": tmp_path / "reports",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))


def tree_snapshot(root: Path) -> dict[str, tuple[str, int, str]]:
    if not root.exists():
        return {}
    snapshot: dict[str, tuple[str, int, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            target = os.readlink(path)
            snapshot[relative] = ("symlink", len(target), target)
        elif path.is_dir():
            snapshot[relative] = ("directory", 0, "")
        elif path.is_file():
            content = path.read_bytes()
            snapshot[relative] = (
                "file",
                len(content),
                hashlib.sha256(content).hexdigest(),
            )
        else:
            snapshot[relative] = ("other", path.lstat().st_size, "")
    return snapshot


def add_manifest_only_source(manifest: dict[str, object]) -> None:
    sources = manifest["sources"]
    assert isinstance(sources, list)
    sources.append(
        {
            "allowed_corpus_fingerprint": "c" * 64,
            "chunking_profile_id": "chunking-v1",
            "document_id": f"doc_{'d' * 64}",
            "file": None,
            "kind": "manifest_only",
            "manifest_fingerprint": "e" * 64,
            "processing_profile_id": "processing-v1",
            "provenance_notes": "Read-only test source.",
            "publisher": "AeroDocs benchmark authors",
            "redistribution_permitted": False,
            "reuse_status": "unknown",
            "source_fingerprint": "f" * 64,
            "source_id": "src_manifest_only",
            "source_url": None,
            "specification": None,
            "title": "Manifest-only source",
        }
    )


def test_validate_cli_is_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = copy_bundle(tmp_path, "milestone-6")
    configure_cli_environment(monkeypatch, tmp_path, root)

    assert main(["validate", "--dataset", "milestone-6"]) == EXIT_SUCCESS


@pytest.mark.parametrize(
    "suite",
    ["retrieval", "answers", "safety", "performance", "complete"],
)
def test_offline_run_cli_executes_every_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suite: str,
) -> None:
    root = copy_bundle(tmp_path, "milestone-6")
    remove_default_threshold(root, "milestone-6")
    configure_cli_environment(monkeypatch, tmp_path, root)

    assert (
        main(
            [
                "run",
                "--suite",
                suite,
                "--dataset",
                "milestone-6",
                "--mode",
                "offline",
                "--report-id",
                f"report-{suite}",
            ]
        )
        == EXIT_SUCCESS
    )
    report = json.loads(
        (tmp_path / "reports" / f"report-{suite}.json").read_text(encoding="utf-8")
    )
    assert report["mode"] == "offline"
    assert report["providers"]["provider_kind"] == "fake"
    assert report["metrics"]
    identities = {
        (metric["population"], metric["metric_id"]) for metric in report["metrics"]
    }
    assert len(identities) == len(report["metrics"])


def test_compare_cli_is_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = copy_bundle(tmp_path, "milestone-6")
    remove_default_threshold(root, "milestone-6")
    configure_cli_environment(monkeypatch, tmp_path, root)
    for report_id in ("baseline", "candidate"):
        assert (
            main(
                [
                    "run",
                    "--suite",
                    "retrieval",
                    "--dataset",
                    "milestone-6",
                    "--mode",
                    "offline",
                    "--report-id",
                    report_id,
                ]
            )
            == EXIT_SUCCESS
        )

    assert (
        main(
            [
                "compare",
                "--baseline",
                "baseline",
                "--candidate",
                "candidate",
                "--report-id",
                "comparison",
            ]
        )
        == EXIT_SUCCESS
    )


def test_cli_rejects_invalid_identifiers_and_downloads_offline() -> None:
    assert main(["validate", "--dataset", "../escape"]) == EXIT_INVALID
    assert (
        main(
            [
                "run",
                "--suite",
                "complete",
                "--dataset",
                "milestone-6",
                "--mode",
                "offline",
                "--report-id",
                "report-id",
                "--allow-model-downloads",
            ]
        )
        == EXIT_INVALID
    )


@pytest.mark.parametrize("bundled_kind", ["synthetic", "redistributable"])
@pytest.mark.parametrize("command", ["validate", "offline_run", "real_run"])
def test_mixed_source_modes_fail_before_production_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundled_kind: str,
    command: str,
) -> None:
    root = copy_bundle(tmp_path, "mixed-dataset")
    bundle = root / "mixed-dataset"
    manifest_path = bundle / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bundled_kind == "redistributable":
        loaded = load_dataset(bundle)
        content = loaded.sources[0].content
        assert content is not None
        (bundle / "source.pdf").write_bytes(content)
        source = manifest["sources"][0]
        source["kind"] = "redistributable"
        source["file"] = "source.pdf"
        source["specification"] = None
    add_manifest_only_source(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AuxiliaryFileError, match="mixed_source_modes_unsupported"):
        load_dataset_by_id(root, "mixed-dataset")

    document_root = tmp_path / "documents"
    index_root = tmp_path / "indexes"
    audit_root = tmp_path / "audits"
    report_root = tmp_path / "reports"
    for runtime_root, marker in (
        (document_root, b"document-state"),
        (index_root, b'{"active":"unchanged"}'),
        (audit_root, b"audit-state"),
    ):
        runtime_root.mkdir()
        (runtime_root / "sentinel").write_bytes(marker)
    (index_root / "active.json").write_bytes(b'{"active":"unchanged"}')
    before = {
        "documents": tree_snapshot(document_root),
        "indexes": tree_snapshot(index_root),
        "audits": tree_snapshot(audit_root),
    }
    configure_cli_environment(monkeypatch, tmp_path, root)
    arguments = ["validate", "--dataset", "mixed-dataset"]
    if command != "validate":
        arguments = [
            "run",
            "--suite",
            "complete",
            "--dataset",
            "mixed-dataset",
            "--mode",
            "offline" if command == "offline_run" else "real",
            "--report-id",
            "mixed-result",
        ]

    assert main(arguments) == EXIT_INVALID
    assert tree_snapshot(document_root) == before["documents"]
    assert tree_snapshot(index_root) == before["indexes"]
    assert tree_snapshot(audit_root) == before["audits"]
    assert tree_snapshot(report_root) == {}
    assert not any(document_root.glob("doc_*"))


@pytest.mark.parametrize("source_kind", ["synthetic", "redistributable"])
@pytest.mark.parametrize("command", ["validate", "run"])
def test_bundled_source_validation_uses_only_isolated_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    command: str,
) -> None:
    root = copy_bundle(tmp_path, "isolated-dataset")
    if command == "run":
        remove_default_threshold(root, "isolated-dataset")
    bundle = root / "isolated-dataset"
    if source_kind == "redistributable":
        loaded = load_dataset(bundle)
        content = loaded.sources[0].content
        assert content is not None
        (bundle / "source.pdf").write_bytes(content)
        manifest_path = bundle / "dataset.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["sources"][0]
        source["kind"] = "redistributable"
        source["file"] = "source.pdf"
        source["specification"] = None
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    for runtime_root in (
        tmp_path / "documents",
        tmp_path / "indexes",
        tmp_path / "audits",
    ):
        runtime_root.mkdir()
        (runtime_root / "sentinel").write_bytes(b"production-unchanged")
    before = {
        name: tree_snapshot(tmp_path / name)
        for name in ("documents", "indexes", "audits")
    }
    configure_cli_environment(monkeypatch, tmp_path, root)

    arguments = ["validate", "--dataset", "isolated-dataset"]
    if command == "run":
        arguments = [
            "run",
            "--suite",
            "retrieval",
            "--dataset",
            "isolated-dataset",
            "--mode",
            "offline",
            "--report-id",
            f"isolated-{source_kind}",
        ]
    assert main(arguments) == EXIT_SUCCESS
    assert {
        name: tree_snapshot(tmp_path / name)
        for name in ("documents", "indexes", "audits")
    } == before


@pytest.mark.parametrize("command", ["validate", "offline_run", "real_run"])
def test_mixed_preflight_does_not_create_absent_runtime_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    root = copy_bundle(tmp_path, "mixed-absent")
    manifest_path = root / "mixed-absent" / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    add_manifest_only_source(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    configure_cli_environment(monkeypatch, tmp_path, root)
    arguments = ["validate", "--dataset", "mixed-absent"]
    if command != "validate":
        arguments = [
            "run",
            "--suite",
            "complete",
            "--dataset",
            "mixed-absent",
            "--mode",
            "offline" if command == "offline_run" else "real",
            "--report-id",
            f"absent-{command}",
        ]

    assert main(arguments) == EXIT_INVALID
    for path in (
        tmp_path / "documents",
        tmp_path / "indexes",
        tmp_path / "audits",
        tmp_path / "reports",
    ):
        assert not path.exists()


def test_mixed_preflight_precedes_store_and_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = copy_bundle(tmp_path, "mixed-construction")
    manifest_path = root / "mixed-construction" / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    add_manifest_only_source(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    settings = isolated_settings(tmp_path, root)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("constructor ran before preflight rejection")

    monkeypatch.setattr(DocumentStore, "__init__", forbidden)
    monkeypatch.setattr(IndexStore, "__init__", forbidden)
    monkeypatch.setattr(GenerationAuditStore, "__init__", forbidden)
    monkeypatch.setattr(
        "app.evaluation.orchestrator.OfflineEmbeddingProvider",
        forbidden,
    )
    monkeypatch.setattr(
        "app.evaluation.orchestrator.OfflineLlmProvider",
        forbidden,
    )
    monkeypatch.setattr(
        "app.evaluation.orchestrator.SentenceTransformerEmbeddingProvider",
        forbidden,
    )
    monkeypatch.setattr(
        "app.evaluation.orchestrator.TransformersLlmProvider",
        forbidden,
    )

    with pytest.raises(AuxiliaryFileError, match="mixed_source_modes_unsupported"):
        validate_bundle(settings, root, "mixed-construction")
    for mode in ("offline", "real"):
        with pytest.raises(AuxiliaryFileError, match="mixed_source_modes_unsupported"):
            run_evaluation(
                settings,
                dataset_root=root,
                dataset_id="mixed-construction",
                suite="complete",
                mode=mode,
                report_id=f"mixed-{mode}",
                allow_model_downloads=False,
            )


def test_manifest_only_missing_roots_remain_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = copy_bundle(tmp_path, "manifest-missing")
    manifest_path = root / "manifest-missing" / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"] = []
    add_manifest_only_source(manifest)
    manifest["sources"][0]["source_id"] = "src_fastener_inspection"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    configure_cli_environment(monkeypatch, tmp_path, root)

    assert main(["validate", "--dataset", "manifest-missing"]) == EXIT_OPERATIONAL
    assert not (tmp_path / "documents").exists()
    assert not (tmp_path / "indexes").exists()
    assert not (tmp_path / "audits").exists()
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize(
    ("filename", "old", "new"),
    [
        (
            "dataset.json",
            '"dataset_version": "1"',
            '"dataset_version": "1", "dataset_version": "2"',
        ),
        (
            "cases.jsonl",
            '"page_start":1',
            '"page_start":1,"page_start":2',
        ),
        (
            "sources/fastener-inspection.json",
            '"source_id": "src_fastener_inspection"',
            ('"source_id": "src_fastener_inspection","source_id": "src_duplicate"'),
        ),
    ],
)
def test_nested_duplicate_keys_invalidate_bundle(
    tmp_path: Path,
    filename: str,
    old: str,
    new: str,
) -> None:
    root = copy_bundle(tmp_path)
    path = root / "bundle" / filename
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(AuxiliaryFileError, match="duplicate_json_key"):
        load_dataset_by_id(root, "bundle")


def test_duplicate_threshold_key_is_rejected(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    thresholds = root / "bundle" / "bad.json"
    thresholds.write_text(
        '[{"metric_id":"x","operator":"at_least","value":1,"value":0,'
        '"population":"model_quality"}]',
        encoding="utf-8",
    )

    with pytest.raises(AuxiliaryFileError, match="duplicate_json_key"):
        load_thresholds(root / "bundle", "bad")


def test_duplicate_stored_report_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "baseline.json").write_text(
        '{"report_schema_version":"1","report_schema_version":"1"}',
        encoding="utf-8",
    )
    (tmp_path / "candidate.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AuxiliaryFileError, match="duplicate_json_key"):
        compare_reports(tmp_path, "baseline", "candidate")


def test_specification_bytes_participate_in_dataset_identity(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    first = load_dataset_by_id(root, "bundle")
    spec = root / "bundle" / "sources" / "fastener-inspection.json"
    spec.write_text(spec.read_text(encoding="utf-8") + " ", encoding="utf-8")
    second = load_dataset_by_id(root, "bundle")

    assert first.fingerprint != second.fingerprint


def test_source_content_mutation_invalidates_declared_identity(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    spec = root / "bundle" / "sources" / "fastener-inspection.json"
    spec.write_text(
        spec.read_text(encoding="utf-8").replace("visible looseness", "damage"),
        encoding="utf-8",
    )

    with pytest.raises(AuxiliaryFileError, match="source_fingerprint_mismatch"):
        load_dataset_by_id(root, "bundle")


def test_redistributable_file_mutation_invalidates_identity(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    loaded = load_dataset_by_id(root, "bundle")
    source = loaded.sources[0]
    assert source.content is not None
    pdf = root / "bundle" / "source.pdf"
    pdf.write_bytes(source.content)
    manifest = root / "bundle" / "dataset.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    configured = value["sources"][0]
    configured["kind"] = "redistributable"
    configured["file"] = "source.pdf"
    configured["specification"] = None
    manifest.write_text(json.dumps(value), encoding="utf-8")
    load_dataset_by_id(root, "bundle")
    pdf.write_bytes(source.content + b"changed")

    with pytest.raises(AuxiliaryFileError, match="source_fingerprint_mismatch"):
        load_dataset_by_id(root, "bundle")


def test_manifest_only_source_identity_mismatch_is_rejected_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = copy_bundle(tmp_path)
    initial = load_dataset_by_id(root, "bundle")
    source = initial.sources[0]
    assert source.content is not None
    settings = isolated_settings(tmp_path, root)
    services = create_services(
        settings,
        embedding_provider=ProductionIdentityOfflineEmbeddingProvider(),
        llm_provider=OfflineLlmProvider(),
    )
    documents = services.documents
    ingested = documents.ingest(
        BytesIO(source.content),
        "manifest-source.pdf",
        settings.allowed_pdf_media_type,
    )
    stored = documents.store.validate_document(ingested.document.document_id)
    manifest_sha = documents.store.manifest_sha256(ingested.document.document_id)
    corpus = services.retrieval.rebuild().index.corpus_fingerprint
    manifest = root / "bundle" / "dataset.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    configured = value["sources"][0]
    configured.update(
        {
            "kind": "manifest_only",
            "specification": None,
            "file": None,
            "document_id": ingested.document.document_id,
            "processing_profile_id": stored.manifest.processing_profile_id,
            "chunking_profile_id": stored.manifest.chunking_format_version,
            "manifest_fingerprint": manifest_sha,
            "allowed_corpus_fingerprint": corpus,
        }
    )
    manifest.write_text(json.dumps(value), encoding="utf-8")
    load_dataset_by_id(root, "bundle", documents=documents)
    before_documents = tree_snapshot(settings.document_storage_root)
    before_indexes = tree_snapshot(settings.index_runtime_root)
    before_audits = tree_snapshot(settings.generation_audit_root)
    validate_bundle(settings, root, "bundle")
    configure_cli_environment(monkeypatch, tmp_path, root)
    assert main(["validate", "--dataset", "bundle"]) == EXIT_SUCCESS
    assert tree_snapshot(settings.document_storage_root) == before_documents
    assert tree_snapshot(settings.index_runtime_root) == before_indexes
    assert tree_snapshot(settings.generation_audit_root) == before_audits
    configured["allowed_corpus_fingerprint"] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AuxiliaryFileError, match="corpus_identity_mismatch"):
        validate_bundle(settings, root, "bundle")
    configured["allowed_corpus_fingerprint"] = corpus
    configured["manifest_fingerprint"] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(AuxiliaryFileError, match="manifest_source_identity_mismatch"):
        load_dataset_by_id(root, "bundle", documents=documents)


def test_production_ingestion_and_index_are_reproducible(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    settings = isolated_settings(tmp_path, root)
    dataset = load_dataset_by_id(root, "bundle")
    source = dataset.sources[0]

    identity = verify_synthetic_reproducibility(source, settings)

    assert identity.pdf_sha256 == source.source_fingerprint
    assert identity.document_id == source.document_id
    assert identity.page_count == 2
    assert identity.chunk_orders == list(range(1, len(identity.chunk_ids) + 1))
    assert identity.vector_profile_id.startswith("vpr_")
    assert identity.index_profile_id.startswith("ipr_")
    assert len(identity.stable_manifest_fingerprint) == 64
    assert len(identity.stable_corpus_fingerprint) == 64


def test_empty_sources_and_empty_cases_are_rejected(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    manifest = root / "bundle" / "dataset.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["sources"] = []
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AuxiliaryFileError):
        load_dataset_by_id(root, "bundle")

    root = copy_bundle(tmp_path / "second")
    (root / "bundle" / "cases.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(AuxiliaryFileError, match="empty_case_file"):
        load_dataset_by_id(root, "bundle")


def test_schema_boundaries_and_unknown_faults_are_rejected() -> None:
    exact = RetrievalBenchmarkCase(
        kind="retrieval",
        case_id="bounded-case",
        population="model_quality",
        suite="retrieval",
        query="bounded",
        tags=[f"tag-{index}" for index in range(MAXIMUM_TAGS)],
    )
    assert len(exact.tags) == MAXIMUM_TAGS
    with pytest.raises(ValueError):
        RetrievalBenchmarkCase(
            **exact.model_dump(exclude={"tags"}),
            tags=[f"tag-{index}" for index in range(MAXIMUM_TAGS + 1)],
        )
    with pytest.raises(ValueError):
        RetrievalBenchmarkCase.model_validate(
            {
                "kind": "retrieval",
                "case_id": "fault-case",
                "population": "operational_fault",
                "suite": "retrieval",
                "query": "fault",
                "fault": "shell_command",
            }
        )


def test_answer_unknown_source_page_and_section_fail_atomically(
    tmp_path: Path,
) -> None:
    root = copy_bundle(tmp_path)
    cases = root / "bundle" / "cases.jsonl"
    original = cases.read_text(encoding="utf-8")
    cases.write_text(
        original.replace(
            '"source_id":"src_fastener_inspection"}]}',
            '"source_id":"src_unknown"}]}',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(AuxiliaryFileError, match="unknown_source_id"):
        load_dataset_by_id(root, "bundle")

    cases.write_text(
        original.replace('"page_end":1,"page_start":1', '"page_end":3,"page_start":3'),
        encoding="utf-8",
    )
    with pytest.raises(AuxiliaryFileError, match="page_target_out_of_range"):
        load_dataset_by_id(root, "bundle")

    cases.write_text(
        original.replace(
            "SECTION 1 - INSPECTION",
            "SECTION DOES NOT EXIST",
        ),
        encoding="utf-8",
    )
    with pytest.raises(AuxiliaryFileError, match="section_target_not_found"):
        run_evaluation(
            isolated_settings(tmp_path, root),
            dataset_root=root,
            dataset_id="bundle",
            suite="answers",
            mode="offline",
            report_id="bad-section",
            allow_model_downloads=False,
        )


def test_validate_cli_resolves_section_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = copy_bundle(tmp_path, "milestone-6")
    cases = root / "milestone-6" / "cases.jsonl"
    cases.write_text(
        cases.read_text(encoding="utf-8").replace(
            "SECTION 1 - INSPECTION",
            "SECTION DOES NOT EXIST",
        ),
        encoding="utf-8",
    )
    configure_cli_environment(monkeypatch, tmp_path, root)
    before = {
        name: tree_snapshot(tmp_path / name)
        for name in ("documents", "indexes", "audits")
    }

    assert main(["validate", "--dataset", "milestone-6"]) == EXIT_INVALID
    assert {
        name: tree_snapshot(tmp_path / name)
        for name in ("documents", "indexes", "audits")
    } == before


def test_case_bound_is_enforced_before_partial_acceptance(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    case = RetrievalBenchmarkCase(
        kind="retrieval",
        case_id="case-id",
        population="model_quality",
        suite="retrieval",
        query="bounded",
    )
    cases = root / "bundle" / "cases.jsonl"
    cases.write_text(
        "\n".join(
            case.model_copy(update={"case_id": f"case-{index}"}).model_dump_json()
            for index in range(MAXIMUM_CASES + 1)
        ),
        encoding="utf-8",
    )

    with pytest.raises(AuxiliaryFileError, match="too_many_cases"):
        load_dataset_by_id(root, "bundle")


def test_real_model_policy_uses_exact_production_identities(tmp_path: Path) -> None:
    settings = isolated_settings(tmp_path, tmp_path / "datasets")
    resolved = real_mode_settings(settings, allow_model_downloads=False)

    assert resolved.embedding_model_name == PINNED_EMBEDDING_MODEL_NAME
    assert resolved.embedding_model_revision == PINNED_EMBEDDING_MODEL_REVISION
    assert resolved.generation_model_name == PINNED_GENERATION_MODEL_NAME
    assert resolved.generation_model_revision == PINNED_GENERATION_MODEL_REVISION
    assert SentenceTransformerEmbeddingProvider(
        resolved,
        local_files_only=True,
    ).local_files_only
    for update in (
        {"generation_model_name": "Qwen/Qwen2.5-0.5B-Instruct"},
        {"generation_model_revision": "a" * 40},
        {"embedding_model_revision": "main"},
    ):
        with pytest.raises(ValueError, match="model_identity_mismatch"):
            real_mode_settings(
                settings.model_copy(update=update),
                allow_model_downloads=True,
            )


def test_operational_fault_is_offline_only(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    cases = root / "bundle" / "cases.jsonl"
    cases.write_text(
        cases.read_text(encoding="utf-8")
        + '\n{"case_id":"fault-case","document_source_ids":null,'
        '"fault":"retrieval_failure","kind":"retrieval",'
        '"population":"framework_validation","query":"fault",'
        '"suite":"retrieval","tags":[],"targets":[]}\n',
        encoding="utf-8",
    )
    settings = isolated_settings(tmp_path, root)

    offline = run_evaluation(
        settings,
        dataset_root=root,
        dataset_id="bundle",
        suite="retrieval",
        mode="offline",
        report_id="offline-fault",
        allow_model_downloads=False,
    )
    assert any(
        case.requested_fault == "retrieval_failure"
        and case.failure_category == "embedding_model_unavailable"
        and case.fault_exercised
        for case in offline.cases
    )
    with pytest.raises(
        AuxiliaryFileError, match="operational_fault_not_allowed_in_real_mode"
    ):
        run_evaluation(
            settings,
            dataset_root=root,
            dataset_id="bundle",
            suite="retrieval",
            mode="real",
            report_id="real-fault",
            allow_model_downloads=False,
        )


@pytest.mark.parametrize(
    ("fault", "observed"),
    [
        ("retrieval_failure", "embedding_model_unavailable"),
        ("malformed_generation", "generation_output_invalid"),
        ("model_unavailable", "llm_model_unavailable"),
        ("generation_timeout", "generation_timeout"),
        ("audit_failure", "audit_persistence_failed"),
    ],
)
def test_each_operational_fault_is_executed(
    tmp_path: Path,
    fault: str,
    observed: str,
) -> None:
    root = copy_bundle(tmp_path)
    cases = root / "bundle" / "cases.jsonl"
    if fault == "retrieval_failure":
        value: dict[str, object] = {
            "case_id": f"fault-{fault}",
            "document_source_ids": None,
            "fault": fault,
            "kind": "retrieval",
            "population": "framework_validation",
            "query": "What inspection is stated?",
            "suite": "retrieval",
            "tags": [],
            "targets": [],
        }
        suite = "retrieval"
    else:
        value = {
            "case_id": f"fault-{fault}",
            "complete_support_set": False,
            "expected_outcome": "insufficient_evidence",
            "fault": fault,
            "gold_claims": [],
            "kind": "answer",
            "population": "framework_validation",
            "prohibited_conclusions": [],
            "question": "What inspection is stated?",
            "suite": "answers",
            "tags": [],
        }
        suite = "answers"
    cases.write_text(json.dumps(value) + "\n", encoding="utf-8")

    report = run_evaluation(
        isolated_settings(tmp_path, root),
        dataset_root=root,
        dataset_id="bundle",
        suite=suite,  # type: ignore[arg-type]
        mode="offline",
        report_id=f"report-{fault}",
        allow_model_downloads=False,
    )

    result = report.cases[0]
    assert result.requested_fault == fault
    assert result.failure_category == observed
    assert result.fault_exercised is True
    assert result.outcome == "operational_failure"


def test_network_guard_denies_socket_http_and_url_resolution() -> None:
    import urllib.request

    with network_denial_guard():
        with pytest.raises(NetworkDeniedError):
            socket.create_connection(("example.invalid", 443))
        with pytest.raises(NetworkDeniedError):
            socket.socket().connect(("127.0.0.1", 9))
        with pytest.raises(NetworkDeniedError):
            urllib.request.urlopen("https://example.invalid")
        with pytest.raises(NetworkDeniedError):
            httpx.get("https://example.invalid")


def test_held_directory_descriptor_resists_parent_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "value.json").write_text('{"safe":true}', encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "value.json").write_text('{"safe":false}', encoding="utf-8")

    with SecureDirectory(root) as directory:
        moved = tmp_path / "moved"
        os.rename(root, moved)
        root.symlink_to(outside, target_is_directory=True)
        assert directory.read("value.json") == b'{"safe":true}'


def test_evaluation_timestamp_accepts_equivalent_utc_and_canonicalizes(
    tmp_path: Path,
) -> None:
    root = copy_bundle(tmp_path)
    source = load_dataset_by_id(root, "bundle").sources[0]
    assert source.content is not None
    instant = datetime(2000, 1, 2, 3, 4, 5)
    accepted = (
        instant.replace(tzinfo=UTC),
        instant.replace(tzinfo=timezone.utc),  # noqa: UP017
        instant.replace(tzinfo=ZoneInfo("UTC")),
        instant.replace(tzinfo=ZeroOffsetTimezone()),
    )
    identities: list[tuple[str, str, datetime, str]] = []
    for index, supplied in enumerate(accepted):
        settings = isolated_settings(tmp_path / f"accepted-{index}", root)
        documents = DocumentService(settings)
        result = documents.ingest(
            BytesIO(source.content),
            "accepted.pdf",
            settings.allowed_pdf_media_type,
            ingested_at=supplied,
        )
        raw_metadata = (
            settings.document_storage_root
            / "documents"
            / result.document.document_id
            / "metadata.json"
        ).read_text(encoding="utf-8")
        identities.append(
            (
                result.document.document_id,
                documents.store.manifest_sha256(result.document.document_id),
                result.document.ingested_at,
                raw_metadata,
            )
        )

    assert len({item[0] for item in identities}) == 1
    assert len({item[1] for item in identities}) == 1
    assert all(item[2].tzinfo is UTC for item in identities)
    assert all(item[2] == accepted[0] for item in identities)
    assert all(
        json.loads(item[3])["ingested_at"] == "2000-01-02T03:04:05Z"
        for item in identities
    )


def test_evaluation_timestamp_rejects_naive_missing_and_nonzero_offsets(
    tmp_path: Path,
) -> None:
    root = copy_bundle(tmp_path)
    source = load_dataset_by_id(root, "bundle").sources[0]
    assert source.content is not None
    documents = DocumentService(isolated_settings(tmp_path, root))
    for invalid in (
        datetime(2000, 1, 1),
        datetime(2000, 1, 1, tzinfo=MissingOffsetTimezone()),
        datetime(2000, 1, 1, tzinfo=timezone(timedelta(hours=1))),
        datetime(2000, 1, 1, tzinfo=timezone(timedelta(hours=-5))),
        datetime(2000, 1, 1, tzinfo=NonZeroOffsetTimezone()),
    ):
        with pytest.raises(ValueError, match="timezone-aware UTC"):
            documents.ingest(
                BytesIO(source.content),
                "invalid.pdf",
                documents.settings.allowed_pdf_media_type,
                ingested_at=invalid,
            )


def test_normal_ingestion_timestamp_remains_current_canonical_utc(
    tmp_path: Path,
) -> None:
    root = copy_bundle(tmp_path)
    source = load_dataset_by_id(root, "bundle").sources[0]
    assert source.content is not None
    documents = DocumentService(isolated_settings(tmp_path, root))
    before = datetime.now(UTC)
    result = documents.ingest(
        BytesIO(source.content),
        "current.pdf",
        documents.settings.allowed_pdf_media_type,
    )
    after = datetime.now(UTC)
    assert before <= result.document.ingested_at <= after
    assert result.document.ingested_at.tzinfo is UTC


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/absolute",
        "sources/",
        "sources//file.json",
        "sources/./file.json",
        "sources/../file.json",
        "sources\\file.json",
        "sources/\u2215/file.json",
        "sources/\u2044/file.json",
        "sources/\uff0f/file.json",
        "sources/\x00file.json",
    ],
)
def test_raw_relative_path_aliases_are_rejected(value: str) -> None:
    with pytest.raises(AuxiliaryFileError, match="unsafe_relative_path"):
        safe_relative_path(value)


def test_existing_report_maps_real_persistence_failure_to_exit_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = copy_bundle(tmp_path, "milestone-6")
    remove_default_threshold(root, "milestone-6")
    configure_cli_environment(monkeypatch, tmp_path, root)
    arguments = [
        "run",
        "--suite",
        "retrieval",
        "--dataset",
        "milestone-6",
        "--mode",
        "offline",
        "--report-id",
        "immutable-report",
    ]
    assert main(arguments) == EXIT_SUCCESS
    assert main(arguments) == EXIT_REPORT
