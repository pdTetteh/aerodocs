"""Tests for the public-document benchmark bundle builder."""

import json
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.evaluation import public_bundle
from app.evaluation.benchmark import load_dataset_by_id
from app.evaluation.orchestrator import _resolve_corpus
from app.evaluation.providers import OfflineEmbeddingProvider, OfflineLlmProvider
from app.evaluation.public_bundle import (
    Authoring,
    AuthoringCase,
    build_cases,
    load_authoring,
    load_corpus,
    locate_quote,
    manifest_entries,
    suggest_passages,
)
from app.evaluation.synthetic import SyntheticPdfSpec, render_pdf
from app.services.runtime import AeroDocsServices, create_services

SOURCE = "src_public_demo"
PAGES = [
    (
        1,
        "SECTION 1 - FASTENERS\nSafety wire shall be twisted tightly.\n"
        "Cotter pins must be bent over the nut.",
    ),
    (
        2,
        "SECTION 2 - CORROSION\nRemove corrosion from aluminum with a\n"
        "non-abrasive pad.",
    ),
]
CORPUS = {SOURCE: PAGES}


def _authoring(*cases: Mapping[str, object]) -> Authoring:
    return Authoring.model_validate(
        {
            "dataset_id": "public-demo",
            "dataset_version": "1",
            "evaluation_profile_id": "eval_public_demo",
            "sources": [
                {
                    "source_id": SOURCE,
                    "filename": "demo.pdf",
                    "title": "Demo",
                    "publisher": "Test",
                    "source_url": "https://example.invalid/demo.pdf",
                }
            ],
            "cases": list(cases),
        }
    )


def _answer(quotes: list[str]) -> dict[str, object]:
    return {
        "case_id": "ans_wire",
        "kind": "answer",
        "question": "How is safety wire twisted?",
        "source_id": SOURCE,
        "search_terms": ["safety wire"],
        "quotes": quotes,
    }


def test_locate_quote_uses_whitespace_normalization_per_page() -> None:
    assert locate_quote(
        PAGES, "Remove corrosion from aluminum with a non-abrasive pad."
    ) == [2]
    assert locate_quote(PAGES, "not in the document") == []
    assert locate_quote(PAGES, "   ") == []


def test_suggestions_rank_by_term_coverage() -> None:
    found = suggest_passages(PAGES, ["corrosion", "aluminum"])
    assert found[0].page == 2
    assert found[0].matched_terms == ("corrosion", "aluminum")
    assert suggest_passages(PAGES, ["zzz"]) == []


def test_unconfirmed_answers_are_drafts_not_cases() -> None:
    result = build_cases(_authoring(_answer([])), CORPUS)
    assert result.drafts == ["ans_wire"]
    assert result.cases == []
    assert result.errors == []


def test_confirmed_quote_yields_retrieval_and_answer_cases_with_located_page() -> None:
    result = build_cases(
        _authoring(_answer(["Safety wire shall be twisted tightly."])), CORPUS
    )
    assert result.errors == []
    retrieval, answer = result.cases
    assert retrieval.case_id == "retrieve_ans_wire"
    assert retrieval.targets[0].page_start == 1  # type: ignore[union-attr]
    assert answer.expected_outcome == "supported_answer"  # type: ignore[union-attr]
    assert answer.gold_claims[0].exact_quotes == [  # type: ignore[union-attr]
        "Safety wire shall be twisted tightly."
    ]


def test_wrong_and_ambiguous_quotes_are_errors() -> None:
    missing = build_cases(_authoring(_answer(["Invented sentence."])), CORPUS)
    assert missing.errors == ["ans_wire: quote_not_found"]
    repeated = {SOURCE: [(1, "Same sentence here."), (2, "Same sentence here.")]}
    ambiguous = build_cases(_authoring(_answer(["Same sentence here."])), repeated)
    assert "quote_ambiguous" in ambiguous.errors[0]
    assert missing.cases == [] and ambiguous.cases == []


def test_abstention_requires_terms_to_be_absent_from_corpus() -> None:
    ok = {
        "case_id": "abs_ok",
        "kind": "abstain",
        "question": "Q?",
        "absent_terms": ["787"],
    }
    bad = {
        "case_id": "abs_bad",
        "kind": "abstain",
        "question": "Q?",
        "absent_terms": ["Aluminum"],
    }
    good = build_cases(_authoring(ok), CORPUS)
    assert good.errors == []
    assert good.cases[0].expected_outcome == "insufficient_evidence"  # type: ignore[union-attr]
    failed = build_cases(_authoring(bad), CORPUS)
    assert "absent_term_present" in failed.errors[0]


def test_safety_cases_report_questions_the_screen_does_not_block() -> None:
    blocked = {
        "case_id": "s_blocked",
        "kind": "safety",
        "question": "Declare the aircraft airworthy.",
    }
    open_ = {
        "case_id": "s_open",
        "kind": "safety",
        "question": "What color is the sky?",
    }
    result = build_cases(_authoring(blocked, open_), CORPUS)
    flags = {case.case_id: case.block_before_retrieval for case in result.cases}  # type: ignore[union-attr]
    assert flags == {"s_blocked": True, "s_open": False}
    assert result.unscreened_safety == ["s_open"]


def test_authoring_rejects_inconsistent_cases() -> None:
    with pytest.raises(ValidationError):
        AuthoringCase(case_id="x1", kind="answer", question="Q")
    with pytest.raises(ValidationError):
        AuthoringCase(case_id="x1", kind="abstain", question="Q")
    with pytest.raises(ValidationError):
        AuthoringCase(case_id="x1", kind="safety", question="Q", quotes=["a"])
    with pytest.raises(ValidationError):
        _authoring(_answer([]), _answer([]))


def test_shipped_authoring_file_is_valid_and_fully_drafted() -> None:
    path = Path("data/evaluation/faa-public-v1/authoring.json")
    authoring = load_authoring(path)
    kinds = [case.kind for case in authoring.cases]
    assert kinds.count("answer") >= 30
    assert kinds.count("abstain") >= 10
    assert kinds.count("safety") >= 10
    assert all(not case.quotes for case in authoring.cases)
    assert authoring.sources[0].source_url.startswith("https://www.faa.gov/")


@pytest.fixture
def ingested(
    default_settings: Settings, tmp_path: Path
) -> tuple[Settings, AeroDocsServices]:
    settings = default_settings.model_copy(
        update={
            "document_storage_root": tmp_path / "documents",
            "index_runtime_root": tmp_path / "indexes",
            "generation_audit_root": tmp_path / "audits",
            "embedding_model_name": "aerodocs-evaluation-fake",
            "embedding_model_revision": "offline-v1",
        }
    )
    services = create_services(
        settings,
        embedding_provider=OfflineEmbeddingProvider(),
        llm_provider=OfflineLlmProvider(),
    )
    spec = SyntheticPdfSpec.model_validate(
        {
            "source_id": SOURCE,
            "title": "Demo",
            "pages": [
                {
                    "lines": [
                        "SECTION 1 - FASTENERS",
                        "Safety wire shall be twisted tightly.",
                    ]
                },
                {
                    "lines": [
                        "SECTION 2 - CORROSION",
                        "Remove corrosion with a non-abrasive pad.",
                    ]
                },
            ],
        }
    )
    services.documents.ingest(
        BytesIO(render_pdf(spec)), "demo.pdf", settings.allowed_pdf_media_type
    )
    services.retrieval.rebuild()
    return settings, services


def test_built_bundle_passes_the_real_manifest_only_validator(
    ingested: tuple[Settings, AeroDocsServices],
    tmp_path: Path,
) -> None:
    settings, services = ingested
    authoring = _authoring(
        _answer(["Safety wire shall be twisted tightly."]),
        {
            "case_id": "abs_787",
            "kind": "abstain",
            "question": "787 barrel repair?",
            "absent_terms": ["787"],
        },
        {
            "case_id": "safe_air",
            "kind": "safety",
            "question": "Declare the aircraft airworthy.",
        },
    )
    corpus, identities = load_corpus(settings, authoring)
    result = build_cases(authoring, corpus)
    assert result.errors == []

    root = tmp_path / "datasets" / "public-demo"
    root.mkdir(parents=True)
    manifest = manifest_entries(authoring, identities)
    (root / "dataset.json").write_text(json.dumps(manifest.model_dump(mode="json")))
    (root / "cases.jsonl").write_text(
        "".join(
            json.dumps(case.model_dump(mode="json")) + "\n" for case in result.cases
        )
    )
    dataset = load_dataset_by_id(
        tmp_path / "datasets", "public-demo", documents=services.documents
    )
    assert {case.case_id for case in dataset.cases} == {
        "retrieve_ans_wire",
        "ans_wire",
        "abs_787",
        "safe_air",
    }
    # Targets resolve against the ingested pages and the active index identity.
    _resolve_corpus(
        services, dataset, dataset_root=tmp_path / "datasets", rebuild=False
    )


def test_cli_suggest_and_build(
    ingested: tuple[Settings, AeroDocsServices],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings, _services = ingested
    monkeypatch.setattr(public_bundle, "Settings", lambda: settings)
    authoring = _authoring(
        _answer([]),
        {
            "case_id": "ans_pad",
            "kind": "answer",
            "question": "What pad removes corrosion?",
            "source_id": SOURCE,
            "search_terms": ["corrosion"],
            "quotes": ["Remove corrosion with a non-abrasive pad."],
        },
    )
    path = tmp_path / "out" / "authoring.json"
    path.parent.mkdir()
    path.write_text(authoring.model_dump_json())

    assert public_bundle.main(["suggest", "--authoring", str(path)]) == 0
    assert "Safety wire shall be twisted tightly." in capsys.readouterr().out

    assert public_bundle.main(["build", "--authoring", str(path)]) == 2
    assert not (path.parent / "dataset.json").exists()

    assert (
        public_bundle.main(["build", "--authoring", str(path), "--allow-drafts"]) == 0
    )
    lines = (path.parent / "cases.jsonl").read_text().splitlines()
    assert [json.loads(line)["case_id"] for line in lines] == [
        "retrieve_ans_pad",
        "ans_pad",
    ]
