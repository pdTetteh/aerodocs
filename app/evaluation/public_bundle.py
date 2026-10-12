"""Build a manifest-only public-document benchmark from human-confirmed quotes.

Operator tooling, not part of the service. A person authors questions and
confirms verbatim quotes against their own ingested copies of public documents.
This module never fetches anything and never labels a case by itself: it only
locates each confirmed quote in the ingested pages, checks that out-of-scope
questions really are absent from the corpus, and pins the document and index
identities the evaluator requires.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import Settings
from app.evaluation.benchmark import (
    CASE_ADAPTER,
    AnswerBenchmarkCase,
    BenchmarkCase,
    DatasetManifest,
    GoldClaim,
    PageTarget,
    RetrievalBenchmarkCase,
)
from app.generation.context import requests_unsafe_decision
from app.ingestion.service import DocumentService, sanitize_filename

SOURCE_ID_PATTERN = r"^src_[a-z0-9][a-z0-9_-]{2,63}$"
CASE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{1,50}$"
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
MAXIMUM_SUGGESTION_CHARACTERS = 450
MAXIMUM_QUOTE_CHARACTERS = 500


class AuthoringSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=SOURCE_ID_PATTERN)
    filename: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    publisher: str = Field(min_length=1, max_length=120)
    source_url: str = Field(min_length=1, max_length=500)


class AuthoringCase(BaseModel):
    """One human-authored case. ``answer`` cases are drafts until quoted."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=CASE_ID_PATTERN)
    kind: Literal["answer", "abstain", "safety"]
    question: str = Field(min_length=1, max_length=2_000)
    source_id: str | None = Field(default=None, pattern=SOURCE_ID_PATTERN)
    search_terms: list[str] = Field(default_factory=list, max_length=12)
    quotes: list[str] = Field(default_factory=list, max_length=8)
    absent_terms: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_kind(self) -> AuthoringCase:
        if self.kind == "answer" and (self.source_id is None or not self.search_terms):
            raise ValueError("answer cases need source_id and search_terms")
        if self.kind == "abstain" and not self.absent_terms:
            raise ValueError("abstain cases need absent_terms")
        if self.kind != "answer" and self.quotes:
            raise ValueError("only answer cases carry quotes")
        if any(not 0 < len(quote) <= MAXIMUM_QUOTE_CHARACTERS for quote in self.quotes):
            raise ValueError("quotes must be 1-500 characters")
        return self


class Authoring(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    dataset_version: str = Field(pattern=r"^[0-9]+$")
    evaluation_profile_id: str = Field(min_length=1, max_length=80)
    sources: list[AuthoringSource] = Field(min_length=1)
    cases: list[AuthoringCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> Authoring:
        source_ids = [source.source_id for source in self.sources]
        case_ids = [case.case_id for case in self.cases]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("duplicate source_id")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("duplicate case_id")
        if any(
            case.source_id is not None and case.source_id not in source_ids
            for case in self.cases
        ):
            raise ValueError("case references an unknown source_id")
        return self


Corpus = dict[str, list[tuple[int, str]]]


@dataclass(frozen=True)
class Suggestion:
    page: int
    text: str
    matched_terms: tuple[str, ...]


@dataclass
class BuildResult:
    cases: list[BenchmarkCase]
    drafts: list[str]
    errors: list[str]
    unscreened_safety: list[str]


def canonical(text: str) -> str:
    """Production grounding whitespace rule."""
    return " ".join(text.split())


def locate_quote(pages: list[tuple[int, str]], quote: str) -> list[int]:
    """Return every page whose text contains the quote under production rules."""
    needle = canonical(quote)
    if not needle:
        return []
    return [number for number, text in pages if needle in canonical(text)]


def suggest_passages(
    pages: list[tuple[int, str]],
    terms: Sequence[str],
    *,
    limit: int = 3,
) -> list[Suggestion]:
    """Rank verbatim sentences by how many search terms they contain."""
    folded = [term.casefold() for term in terms if term.strip()]
    found: list[Suggestion] = []
    for number, text in pages:
        for sentence in SENTENCE_BREAK.split(canonical(text)):
            if not 20 <= len(sentence) <= MAXIMUM_SUGGESTION_CHARACTERS:
                continue
            lowered = sentence.casefold()
            matched = tuple(term for term in folded if term in lowered)
            if matched:
                found.append(Suggestion(number, sentence, matched))
    found.sort(key=lambda item: (-len(item.matched_terms), item.page))
    return found[:limit]


def build_cases(authoring: Authoring, corpus: Corpus) -> BuildResult:
    """Turn confirmed authoring cases into evaluator cases; never guess labels."""
    result = BuildResult(cases=[], drafts=[], errors=[], unscreened_safety=[])
    for item in authoring.cases:
        if item.kind == "answer":
            _build_answer(item, corpus, result)
        elif item.kind == "abstain":
            _build_abstention(item, corpus, result)
        else:
            _build_safety(item, result)
    return result


def _build_answer(item: AuthoringCase, corpus: Corpus, result: BuildResult) -> None:
    if not item.quotes:
        result.drafts.append(item.case_id)
        return
    assert item.source_id is not None
    pages = corpus.get(item.source_id)
    if pages is None:
        result.errors.append(f"{item.case_id}: source not ingested")
        return
    targets: list[PageTarget] = []
    for quote in item.quotes:
        located = locate_quote(pages, quote)
        if not located:
            result.errors.append(f"{item.case_id}: quote_not_found")
            return
        if len(located) > 1:
            result.errors.append(
                f"{item.case_id}: quote_ambiguous on pages {located}; lengthen it"
            )
            return
        targets.append(
            PageTarget(
                source_id=item.source_id,
                page_start=located[0],
                page_end=located[0],
                grade=2,
            )
        )
    unique = list({target.page_start: target for target in targets}.values())
    retrieval_id = f"retrieve_{item.case_id}"
    if len(retrieval_id) > 64:
        result.errors.append(f"{item.case_id}: case_id too long")
        return
    result.cases.append(
        RetrievalBenchmarkCase(
            case_id=retrieval_id,
            kind="retrieval",
            population="model_quality",
            suite="retrieval",
            tags=["public", "faa"],
            query=item.question,
            document_source_ids=[item.source_id],
            targets=list(unique),
        )
    )
    result.cases.append(
        AnswerBenchmarkCase(
            case_id=item.case_id,
            kind="answer",
            population="model_quality",
            suite="answers",
            tags=["public", "faa"],
            question=item.question,
            expected_outcome="supported_answer",
            document_source_ids=[item.source_id],
            gold_claims=[
                GoldClaim(
                    claim_id="gold_1",
                    support_targets=list(unique),
                    exact_quotes=list(dict.fromkeys(item.quotes)),
                )
            ],
        )
    )


def _build_abstention(
    item: AuthoringCase,
    corpus: Corpus,
    result: BuildResult,
) -> None:
    for term in item.absent_terms:
        lowered = term.casefold()
        hits = [
            source_id
            for source_id, pages in corpus.items()
            if any(lowered in canonical(text).casefold() for _, text in pages)
        ]
        if hits:
            result.errors.append(f"{item.case_id}: absent_term_present {term!r}")
            return
    result.cases.append(
        AnswerBenchmarkCase(
            case_id=item.case_id,
            kind="answer",
            population="model_quality",
            suite="answers",
            tags=["public", "faa", "out_of_scope"],
            question=item.question,
            expected_outcome="insufficient_evidence",
        )
    )


def _build_safety(item: AuthoringCase, result: BuildResult) -> None:
    blocked = requests_unsafe_decision(item.question)
    if not blocked:
        result.unscreened_safety.append(item.case_id)
    result.cases.append(
        AnswerBenchmarkCase(
            case_id=item.case_id,
            kind="answer",
            population="model_quality",
            suite="safety",
            tags=["public", "faa", "safety"],
            question=item.question,
            expected_outcome="safety_boundary",
            block_before_retrieval=blocked,
            prohibited_conclusions=["maintenance approval"],
        )
    )


def manifest_entries(
    authoring: Authoring,
    identities: dict[str, dict[str, str]],
) -> DatasetManifest:
    """Build the pinned manifest-only dataset manifest."""
    sources = []
    for source in authoring.sources:
        pins = identities[source.source_id]
        sources.append(
            {
                "source_id": source.source_id,
                "kind": "manifest_only",
                "title": source.title,
                "publisher": source.publisher,
                "reuse_status": "public_domain",
                "redistribution_permitted": False,
                "source_url": source.source_url,
                "provenance_notes": (
                    "Public U.S. FAA document, referenced by pinned identity only. "
                    "The PDF is not bundled; fetch it from source_url."
                ),
                **pins,
            }
        )
    return DatasetManifest.model_validate(
        {
            "dataset_id": authoring.dataset_id,
            "dataset_version": authoring.dataset_version,
            "schema_version": "1",
            "evaluation_profile_id": authoring.evaluation_profile_id,
            "cases_file": "cases.jsonl",
            "sources": sources,
        }
    )


def load_authoring(path: Path) -> Authoring:
    return Authoring.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_corpus(
    settings: Settings,
    authoring: Authoring,
) -> tuple[Corpus, dict[str, dict[str, str]]]:
    """Read pages and identities for each source from the existing corpus."""
    from app.evaluation.providers import OfflineEmbeddingProvider
    from app.retrieval.service import RetrievalService

    documents = DocumentService.open_existing(settings)
    retrieval = RetrievalService.open_existing(settings, OfflineEmbeddingProvider())
    corpus_fingerprint = retrieval.active_metadata_without_models().corpus_fingerprint
    by_filename: dict[str, str] = {}
    for stored_id in documents.store.list_document_ids():
        by_filename[documents.metadata(stored_id).filename] = stored_id
    corpus: Corpus = {}
    identities: dict[str, dict[str, str]] = {}
    for source in authoring.sources:
        document_id = by_filename.get(sanitize_filename(source.filename))
        if document_id is None:
            raise SystemExit(f"{source.source_id}: {source.filename} is not ingested")
        validated = documents.store.validate_document(
            document_id, page_limit=settings.max_pdf_pages
        )
        corpus[source.source_id] = [
            (page.page_number, page.text) for page in validated.pages
        ]
        identities[source.source_id] = {
            "source_fingerprint": validated.metadata.sha256,
            "document_id": document_id,
            "processing_profile_id": validated.manifest.processing_profile_id,
            "chunking_profile_id": validated.manifest.chunking_format_version,
            "manifest_fingerprint": documents.store.manifest_sha256(document_id),
            "allowed_corpus_fingerprint": corpus_fingerprint,
        }
    return corpus, identities


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation.public_bundle")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("suggest", "build"):
        command = commands.add_parser(name)
        command.add_argument("--authoring", type=Path, required=True)
    commands.choices["build"].add_argument(
        "--allow-drafts",
        action="store_true",
        help="skip answer cases that have no confirmed quote yet",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    authoring = load_authoring(parsed.authoring)
    corpus, identities = load_corpus(Settings(), authoring)

    if parsed.command == "suggest":
        for item in authoring.cases:
            if item.kind != "answer" or item.quotes:
                continue
            assert item.source_id is not None
            print(f"\n== {item.case_id}: {item.question}")
            found = suggest_passages(corpus[item.source_id], item.search_terms)
            if not found:
                print("   no candidates; revise search_terms or drop the case")
            for suggestion in found:
                print(f"   p.{suggestion.page} {list(suggestion.matched_terms)}")
                print(f"   {json.dumps(suggestion.text)}")
        return 0

    result = build_cases(authoring, corpus)
    if result.drafts and not parsed.allow_drafts:
        print(f"{len(result.drafts)} draft cases have no confirmed quote:")
        print("  " + ", ".join(result.drafts))
        print("Run `suggest`, confirm quotes, or pass --allow-drafts.")
        return 2
    if result.errors:
        print("Authoring errors:\n  " + "\n  ".join(result.errors))
        return 2
    manifest = manifest_entries(authoring, identities)
    for case in result.cases:
        CASE_ADAPTER.validate_python(case.model_dump(mode="json"))
    output = parsed.authoring.parent
    (output / "dataset.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "cases.jsonl").write_text(
        "".join(
            json.dumps(case.model_dump(mode="json"), sort_keys=True) + "\n"
            for case in result.cases
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(result.cases)} cases to {output}")
    print(f"Skipped {len(result.drafts)} unconfirmed drafts")
    if result.unscreened_safety:
        print(
            "Not blocked before retrieval by the question screen: "
            + ", ".join(result.unscreened_safety)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
