"""Offline protocol tests for the read-oriented AeroDocs MCP adapter."""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult
from numpy.typing import NDArray
from pydantic import AnyUrl

from app.core.config import Settings
from app.generation.models import PreparedGenerationInput, ProviderIdentity
from app.generation.service import AnswerService
from app.ingestion.models import (
    DocumentListResponse,
    DocumentSummary,
    ProcessingProfile,
)
from app.ingestion.service import DocumentService
from app.mcp_server.results import serialize_tool_result
from app.mcp_server.server import (
    _bounded_tool_failure,
    _find_trusted_tool_error,
    _TrustedMcpToolError,
    create_mcp_server,
)
from app.mcp_server.validation import (
    InvalidMcpArgumentsError,
    decode_document_resource_id,
)
from app.retrieval.models import ModelIdentity
from app.retrieval.service import RetrievalService


class FakeEmbeddingProvider:
    """Deterministic offline embedding provider."""

    def __init__(self) -> None:
        self.identity_calls = 0
        self.document_calls = 0
        self.query_calls = 0

    def model_identity(self) -> ModelIdentity:
        self.identity_calls += 1
        return ModelIdentity(
            name="fake-mcp-embedding",
            revision="mcp-test-revision",
            dimension=4,
            maximum_sequence_length=512,
        )

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.document_calls += 1
        return np.asarray([_vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.query_calls += 1
        return np.asarray([_vector(text)], dtype=np.float32)


class FakeLlmProvider:
    """Deterministic offline structured generation provider."""

    def __init__(self, output: str | None = None) -> None:
        self.prepare_calls = 0
        self.generate_calls = 0
        self.last_user = ""
        self.output = output

    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            name="fake-mcp-generator",
            revision="mcp-test-revision",
            device="cpu",
        )

    def prepare(
        self,
        _system_prompt: str,
        user_prompt: str,
    ) -> PreparedGenerationInput:
        self.prepare_calls += 1
        self.last_user = user_prompt
        return PreparedGenerationInput(
            input_ids=object(),
            attention_mask=object(),
            prompt_token_count=20,
            model_context_limit=1_000,
            maximum_new_tokens=64,
            provider=self.identity(),
            response_prefix="",
            deadline_monotonic=float("inf"),
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, _prepared: PreparedGenerationInput) -> str:
        self.generate_calls += 1
        if self.output is not None:
            return self.output
        evidence = json.loads(self.last_user)["untrusted_evidence"][0]
        return json.dumps(
            {
                "status": "answer",
                "claims": [
                    {
                        "text": "Inspect corrosion carefully.",
                        "supports": [
                            {
                                "evidence_id": evidence["evidence_id"],
                                "quote": "Inspect corrosion carefully.",
                            }
                        ],
                    }
                ],
            }
        )


def _vector(text: str) -> list[float]:
    lowered = text.lower()
    return [
        float(lowered.count("corrosion")),
        float(lowered.count("inspect")),
        float(lowered.count("engine")),
        1.0,
    ]


def _settings(default: Settings, tmp_path: Path) -> Settings:
    return default.model_copy(
        update={
            "document_storage_root": tmp_path / "documents",
            "index_runtime_root": tmp_path / "indexes",
            "generation_audit_root": tmp_path / "audits",
            "embedding_model_name": "fake-mcp-embedding",
            "embedding_model_revision": "mcp-test-revision",
            "minimum_similarity_score": 0.0,
            "answer_minimum_similarity_score": 0.0,
            "mcp_default_pagination_limit": 1,
            "mcp_maximum_pagination_limit": 2,
        }
    )


def _ingest_and_index(
    settings: Settings,
    provider: FakeEmbeddingProvider,
    make_pdf: Callable[..., bytes],
    *,
    filename: str = "public.pdf",
) -> str:
    document = DocumentService(settings).ingest(
        BytesIO(make_pdf(["CORROSION INSPECTION\nInspect corrosion carefully."])),
        filename,
        "application/pdf",
    )
    RetrievalService(settings, provider).rebuild()
    return document.document.document_id


@pytest.mark.asyncio
async def test_discovery_has_exact_allowlists_and_annotations(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    server = create_mcp_server(
        _settings(default_settings, tmp_path),
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLlmProvider(),
    )
    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
        resources = await session.list_resources()
        templates = await session.list_resource_templates()
        prompts = await session.list_prompts()

    assert [tool.name for tool in tools.tools] == [
        "list_documents",
        "get_document_metadata",
        "get_active_index",
        "search_evidence",
        "answer_from_documents",
    ]
    expected = {
        "list_documents": (True, False, True, False),
        "get_document_metadata": (True, False, True, False),
        "get_active_index": (True, False, True, False),
        "search_evidence": (True, False, True, False),
        "answer_from_documents": (False, False, False, False),
    }
    for tool in tools.tools:
        expected_properties = {
            "list_documents": {"offset", "limit"},
            "get_document_metadata": {"document_id"},
            "get_active_index": set(),
            "search_evidence": {"query", "top_k", "document_ids"},
            "answer_from_documents": {"question", "top_k", "document_ids"},
        }
        assert tool.inputSchema["additionalProperties"] is False
        assert set(tool.inputSchema["properties"]) == expected_properties[tool.name]
        assert tool.annotations is not None
        assert (
            tool.annotations.readOnlyHint,
            tool.annotations.destructiveHint,
            tool.annotations.idempotentHint,
            tool.annotations.openWorldHint,
        ) == expected[tool.name]
    answer_description = next(
        tool.description for tool in tools.tools if tool.name == "answer_from_documents"
    )
    assert answer_description is not None
    assert "sanitized local audit records" in answer_description
    assert [str(resource.uri) for resource in resources.resources] == [
        "aerodocs://status",
        "aerodocs://index",
    ]
    assert [template.uriTemplate for template in templates.resourceTemplates] == [
        "aerodocs://documents/{document_id}/metadata"
    ]
    assert prompts.prompts == []


@pytest.mark.asyncio
async def test_all_tools_reject_unknown_arguments_before_services(
    default_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(default_settings, tmp_path)
    embedding = FakeEmbeddingProvider()
    generator = FakeLlmProvider()
    service_calls: list[str] = []

    def fail(name: str) -> Callable[..., NoReturn]:
        def rejected(*_args: object, **_kwargs: object) -> NoReturn:
            service_calls.append(name)
            raise AssertionError("service must not be invoked")

        return rejected

    monkeypatch.setattr(DocumentService, "list_documents", fail("list_documents"))
    monkeypatch.setattr(DocumentService, "metadata", fail("metadata"))
    monkeypatch.setattr(
        RetrievalService,
        "active_metadata_without_models",
        fail("active_index"),
    )
    monkeypatch.setattr(RetrievalService, "search", fail("search"))
    monkeypatch.setattr(AnswerService, "answer", fail("answer"))

    valid_id = f"doc_{'a' * 64}"
    rejected_calls: list[tuple[str, dict[str, Any]]] = [
        ("list_documents", {"evil": "x"}),
        ("get_document_metadata", {"document_id": valid_id, "path": "/tmp"}),
        ("get_active_index", {"rebuild": True}),
        ("search_evidence", {"query": "x", "minimum_score": -1}),
        ("search_evidence", {"query": "x", "topp_k": 5}),
        ("answer_from_documents", {"question": "x", "minimum_score": -1}),
        ("answer_from_documents", {"question": "x", "system_prompt": "..."}),
        ("list_documents", {"command": "rebuild"}),
        ("get_active_index", {"environment": "PATH"}),
        ("search_evidence", {"query": "x", "filename": "../../manual.pdf"}),
    ]
    server = create_mcp_server(
        settings,
        embedding_provider=embedding,
        llm_provider=generator,
    )
    async with create_connected_server_and_client_session(server) as session:
        for name, arguments in rejected_calls:
            result = await session.call_tool(name, arguments)
            assert result.isError is True
            assert "invalid_arguments" in result.content[0].text  # type: ignore[union-attr]
            assert "minimum_score" not in result.content[0].text  # type: ignore[union-attr]
            assert service_calls == []
            assert embedding.identity_calls == 0
            assert embedding.document_calls == 0
            assert embedding.query_calls == 0
            assert generator.prepare_calls == 0
            assert generator.generate_calls == 0
            assert list(settings.generation_audit_root.glob("*.json")) == []
            assert len((await session.list_tools()).tools) == 5


@pytest.mark.asyncio
async def test_unexpected_tool_failures_are_safe_internal_errors(
    default_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(default_settings, tmp_path)
    raw_failure = (
        "RuntimeError /private/cache/model prompt=secret raw-model-output=secret"
    )

    def unexpected(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError(raw_failure)

    original_list_documents = DocumentService.list_documents
    monkeypatch.setattr(DocumentService, "list_documents", unexpected)
    monkeypatch.setattr(DocumentService, "metadata", unexpected)
    monkeypatch.setattr(RetrievalService, "search", unexpected)
    monkeypatch.setattr(AnswerService, "answer", unexpected)

    server = create_mcp_server(
        settings,
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLlmProvider(),
    )
    valid_id = f"doc_{'a' * 64}"
    calls: list[tuple[str, dict[str, Any]]] = [
        ("list_documents", {}),
        ("get_document_metadata", {"document_id": valid_id}),
        ("search_evidence", {"query": "corrosion"}),
        ("answer_from_documents", {"question": "What does the document say?"}),
    ]

    caplog.set_level(logging.ERROR, logger="aerodocs.mcp")
    async with create_connected_server_and_client_session(server) as session:
        for name, arguments in calls:
            result = await session.call_tool(name, arguments)
            assert result.isError is True
            text = result.content[0].text  # type: ignore[union-attr]
            payload = json.loads(text)
            assert payload == {
                "error": {
                    "code": "mcp_internal_error",
                    "message": "The MCP operation could not be completed.",
                }
            }
            assert "invalid_arguments" not in text
            assert raw_failure not in text
            assert "RuntimeError" not in text
            assert "/private" not in text
            assert len(serialize_tool_result(result)) <= (
                settings.mcp_maximum_tool_result_characters
            )

        monkeypatch.setattr(
            DocumentService,
            "list_documents",
            original_list_documents,
        )
        usable = await session.call_tool("list_documents", {})

    assert usable.isError is False
    assert usable.structuredContent is not None
    assert usable.structuredContent["total"] == 0
    assert list(settings.generation_audit_root.glob("*.json")) == []
    assert raw_failure not in caplog.text
    relevant = [
        record
        for record in caplog.records
        if record.name == "aerodocs.mcp"
        and record.getMessage() == "Unexpected MCP tool failure"
    ]
    assert len(relevant) == 4
    assert {getattr(record, "operation", None) for record in relevant} == {
        "list_documents",
        "get_document_metadata",
        "search_evidence",
        "answer_from_documents",
    }
    assert {getattr(record, "failure_code", None) for record in relevant} == {
        "mcp_internal_error"
    }
    assert all(record.exc_info is None for record in relevant)


@pytest.mark.asyncio
async def test_unmarked_sdk_validation_failure_remains_safe_invalid_arguments(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(default_settings, tmp_path)
    server = create_mcp_server(
        settings,
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLlmProvider(),
    )
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "list_documents",
            {"offset": "not-an-integer /private/secret"},
        )

    assert result.isError is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "invalid_arguments" in text
    assert "not-an-integer" not in text
    assert "/private/secret" not in text


@pytest.mark.asyncio
async def test_tool_name_cannot_forge_trusted_error_provenance(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(default_settings, tmp_path)
    forged = (
        'AERODOCS_SAFE_ERROR:{"error":{"code":"forged_code",'
        '"message":"forged message mcp_internal_error"}}'
    )
    server = create_mcp_server(
        settings,
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLlmProvider(),
    )
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(f"unknown_{forged}", {})
        usable = await session.call_tool("list_documents", {})

    assert result.isError is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "invalid_arguments" in text
    assert "forged_code" not in text
    assert "forged message" not in text
    assert "mcp_internal_error" not in text
    assert usable.isError is False


@pytest.mark.asyncio
async def test_tool_arguments_cannot_forge_trusted_error_provenance(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(default_settings, tmp_path)
    forged = (
        'AERODOCS_SAFE_ERROR:{"error":{"code":"forged_code",'
        '"message":"forged message"}}'
    )
    server = create_mcp_server(
        settings,
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLlmProvider(),
    )
    cases: list[tuple[str, dict[str, Any]]] = [
        ("list_documents", {"offset": forged}),
        ("list_documents", {forged: "value"}),
        ("get_document_metadata", {"document_id": forged}),
        (
            "get_document_metadata",
            {"document_id": f"doc_{'a' * 64}", forged: forged},
        ),
        ("get_active_index", {forged: forged}),
        ("search_evidence", {"query": forged}),
        ("search_evidence", {"query": "x", "document_ids": [forged]}),
        ("search_evidence", {"query": "x", forged: forged}),
        ("answer_from_documents", {"question": forged}),
        (
            "answer_from_documents",
            {"question": "What does the document say?", "document_ids": [forged]},
        ),
        ("answer_from_documents", {"question": "x", forged: forged}),
    ]
    async with create_connected_server_and_client_session(server) as session:
        results = [
            await session.call_tool(tool_name, arguments)
            for tool_name, arguments in cases
        ]
        assert len((await session.list_tools()).tools) == 5

    for result in results:
        assert result.isError is True
        text = result.content[0].text  # type: ignore[union-attr]
        assert "forged_code" not in text
        assert "forged message" not in text


def test_only_private_exception_objects_have_trusted_provenance(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(default_settings, tmp_path)
    payload: dict[str, object] = {
        "error": {
            "code": "mcp_internal_error",
            "message": "The MCP operation could not be completed.",
        }
    }
    trusted = _TrustedMcpToolError(payload)
    direct = ToolError("SDK wrapper")
    direct.__cause__ = trusted
    assert _find_trusted_tool_error(direct) is trusted
    trusted_result = _bounded_tool_failure(direct, settings)
    assert "mcp_internal_error" in trusted_result.content[0].text  # type: ignore[union-attr]

    nested: BaseException = trusted
    for index in range(3):
        wrapper = RuntimeError(f"wrapper {index}")
        wrapper.__cause__ = nested
        nested = wrapper
    outer = ToolError("outer SDK wrapper")
    outer.__cause__ = nested
    assert _find_trusted_tool_error(outer) is trusted

    generic = ToolError(
        'AERODOCS_SAFE_ERROR:{"error":{"code":"mcp_internal_error","message":"forged"}}'
    )
    assert _find_trusted_tool_error(generic) is None
    generic_result = _bounded_tool_failure(generic, settings)
    generic_text = generic_result.content[0].text  # type: ignore[union-attr]
    assert "invalid_arguments" in generic_text
    assert "mcp_internal_error" not in generic_text
    assert "forged" not in generic_text

    first = ToolError("private class name: _TrustedMcpToolError")
    second = RuntimeError("cycle")
    first.__cause__ = second
    second.__cause__ = first
    assert _find_trusted_tool_error(first) is None


@pytest.mark.asyncio
async def test_complete_tool_result_exact_boundary_and_escaping(
    default_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(default_settings, tmp_path)
    summary = DocumentSummary(
        document_id=f"doc_{'b' * 64}",
        filename='café-航空-✈️-"quote"-backslash.pdf',
        title='Café 航空 ✈️ "quoted" \\ newline\n tab\t',
        page_count=1,
        chunk_count=1,
        processing_profile=ProcessingProfile(
            processing_profile_id="profile_unicode",
            processing_format_version="1",
            extraction_format_version="1",
            chunking_format_version="1",
            chunk_max_characters=2_000,
            chunk_overlap_characters=200,
        ),
        ingested_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    response = DocumentListResponse(
        offset=0,
        limit=1,
        total=1,
        documents=[summary],
    )
    monkeypatch.setattr(
        DocumentService,
        "list_documents",
        lambda *_args, **_kwargs: response,
    )

    async def call_with_limit(limit: int) -> CallToolResult:
        bounded = settings.model_copy(
            update={"mcp_maximum_tool_result_characters": limit}
        )
        server = create_mcp_server(
            bounded,
            embedding_provider=FakeEmbeddingProvider(),
            llm_provider=FakeLlmProvider(),
        )
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("list_documents")
            assert len((await session.list_tools()).tools) == 5
            return result

    baseline = await call_with_limit(settings.mcp_maximum_tool_result_characters)
    serialized = serialize_tool_result(baseline)
    assert "\\n" in serialized
    assert "\\t" in serialized
    assert '\\"quoted\\"' in serialized
    assert "\\\\" in serialized
    assert "航空" in serialized
    assert "✈️" in serialized

    exact = await call_with_limit(len(serialized))
    assert exact.isError is False
    assert serialize_tool_result(exact) == serialized

    oversized = await call_with_limit(len(serialized) - 1)
    assert oversized.isError is True
    oversized_text = oversized.content[0].text  # type: ignore[union-attr]
    assert "mcp_result_too_large" in oversized_text
    assert "Café" not in oversized_text
    assert "航空" not in oversized_text


@pytest.mark.asyncio
async def test_document_tools_resources_and_pagination(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    settings = _settings(default_settings, tmp_path)
    provider = FakeEmbeddingProvider()
    document_id = _ingest_and_index(settings, provider, make_pdf)
    server = create_mcp_server(
        settings,
        embedding_provider=provider,
        llm_provider=FakeLlmProvider(),
    )

    async with create_connected_server_and_client_session(server) as session:
        listed = await session.call_tool("list_documents", {})
        metadata = await session.call_tool(
            "get_document_metadata",
            {"document_id": document_id},
        )
        status = await session.read_resource(AnyUrl("aerodocs://status"))
        resource = await session.read_resource(
            AnyUrl(f"aerodocs://documents/{document_id}/metadata")
        )
        invalid_page = await session.call_tool(
            "list_documents",
            {"offset": -1},
        )

    assert listed.isError is False
    assert listed.structuredContent is not None
    assert listed.structuredContent["total"] == 1
    assert listed.structuredContent["documents"][0]["document_id"] == document_id
    assert "source.pdf" not in json.dumps(listed.structuredContent)
    assert metadata.isError is False
    assert metadata.structuredContent is not None
    assert metadata.structuredContent["document_id"] == document_id
    assert json.loads(status.contents[0].text)["status"] == "healthy"  # type: ignore[union-attr]
    assert json.loads(resource.contents[0].text)["document_id"] == document_id  # type: ignore[union-attr]
    assert invalid_page.isError is True
    assert "invalid_arguments" in invalid_page.content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_index_metadata_search_and_answer_use_existing_services(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    settings = _settings(default_settings, tmp_path)
    embeddings = FakeEmbeddingProvider()
    generator = FakeLlmProvider()
    document_id = _ingest_and_index(settings, embeddings, make_pdf)
    embeddings.identity_calls = 0
    embeddings.document_calls = 0
    server = create_mcp_server(
        settings,
        embedding_provider=embeddings,
        llm_provider=generator,
    )

    async with create_connected_server_and_client_session(server) as session:
        active = await session.call_tool("get_active_index")
        index_resource = await session.read_resource(AnyUrl("aerodocs://index"))
        assert embeddings.identity_calls == 0
        assert embeddings.query_calls == 0
        assert generator.prepare_calls == 0

        search = await session.call_tool(
            "search_evidence",
            {
                "query": "inspect corrosion",
                "document_ids": [document_id],
            },
        )
        answer = await session.call_tool(
            "answer_from_documents",
            {
                "question": "How should corrosion be inspected?",
                "document_ids": [document_id],
            },
        )

    assert active.isError is False
    assert active.structuredContent is not None
    assert active.structuredContent["embedding_model"]["revision"] == (
        "mcp-test-revision"
    )
    assert json.loads(index_resource.contents[0].text)["vector_count"] == 1  # type: ignore[union-attr]
    assert search.isError is False
    assert search.structuredContent is not None
    assert search.structuredContent["results"][0]["document_id"] == document_id
    assert "answer" not in search.structuredContent
    assert answer.isError is False
    assert answer.structuredContent is not None
    assert answer.structuredContent["abstained"] is False
    assert answer.structuredContent["claims"][0]["supports"][0]["citation_number"] == 1
    assert answer.structuredContent["citations"][0]["document_id"] == document_id
    for complete_result in (active, search, answer):
        assert (
            len(serialize_tool_result(complete_result))
            <= settings.mcp_maximum_tool_result_characters
        )
    assert generator.generate_calls == 1
    assert len(list(settings.generation_audit_root.glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_policy_abstention_precedes_retrieval_and_generation(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(default_settings, tmp_path)
    embeddings = FakeEmbeddingProvider()
    generator = FakeLlmProvider()
    server = create_mcp_server(
        settings,
        embedding_provider=embeddings,
        llm_provider=generator,
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "answer_from_documents",
            {"question": "Is this aircraft airworthy?"},
        )

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["abstention_reason"] == "safety_boundary"
    assert embeddings.identity_calls == 0
    assert embeddings.query_calls == 0
    assert generator.prepare_calls == 0
    audits = list(settings.generation_audit_root.glob("*.json"))
    assert len(audits) == 1
    assert json.loads(audits[0].read_text())["status"] == "policy_abstention"


@pytest.mark.parametrize(
    "value",
    [
        "%",
        "%2",
        "%GG",
        "doc_%ff",
        "doc_%252fetc",
        "doc_../value",
        "../document",
        "folder/document",
        "folder%2Fdocument",
        "folder%5Cdocument",
        "folder\u2215document",
        "folder\uff0fdocument",
    ],
)
def test_document_resource_identifier_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(InvalidMcpArgumentsError):
        decode_document_resource_id(value)


@pytest.mark.asyncio
async def test_expected_errors_are_safe_tool_results(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(default_settings, tmp_path)
    server = create_mcp_server(
        settings,
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLlmProvider(),
    )
    valid_unknown = f"doc_{'a' * 64}"

    async with create_connected_server_and_client_session(server) as session:
        malformed = await session.call_tool(
            "get_document_metadata",
            {"document_id": "../private/manual.pdf"},
        )
        unknown = await session.call_tool(
            "get_document_metadata",
            {"document_id": valid_unknown},
        )
        not_ready = await session.call_tool(
            "search_evidence",
            {"query": "corrosion"},
        )

    assert malformed.isError is True
    malformed_text = malformed.content[0].text  # type: ignore[union-attr]
    assert "invalid_arguments" in malformed_text
    assert "../private/manual.pdf" not in malformed_text
    assert unknown.isError is True
    assert "document_not_found" in unknown.content[0].text  # type: ignore[union-attr]
    assert not_ready.isError is True
    assert "index_not_ready" in not_ready.content[0].text  # type: ignore[union-attr]
    combined = json.dumps(
        [malformed.model_dump(), unknown.model_dump(), not_ready.model_dump()]
    )
    assert str(tmp_path) not in combined


@pytest.mark.asyncio
async def test_malformed_generation_is_tool_error_and_audited(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    settings = _settings(default_settings, tmp_path)
    embeddings = FakeEmbeddingProvider()
    _ingest_and_index(settings, embeddings, make_pdf)
    generator = FakeLlmProvider("not-json")
    server = create_mcp_server(
        settings,
        embedding_provider=embeddings,
        llm_provider=generator,
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "answer_from_documents",
            {"question": "How should corrosion be inspected?"},
        )

    assert result.isError is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "generation_output_invalid" in text
    assert "not-json" not in text
    audits = list(settings.generation_audit_root.glob("*.json"))
    assert len(audits) == 1
    audit_text = audits[0].read_text()
    assert '"status":"invalid_output_failure"' in audit_text
    assert "not-json" not in audit_text


@pytest.mark.asyncio
async def test_result_limit_rejects_without_truncation(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    settings = _settings(default_settings, tmp_path).model_copy(
        update={"mcp_maximum_tool_result_characters": 32}
    )
    DocumentService(settings).ingest(
        BytesIO(make_pdf(["Public metadata."])),
        "public.pdf",
        "application/pdf",
    )
    server = create_mcp_server(
        settings,
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLlmProvider(),
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("list_documents")

    assert result.isError is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "mcp_result_too_large" in text
    assert "Public metadata" not in text


def test_stdio_protocol_stdout_is_json_only(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repository),
            "AERODOCS_DOCUMENT_STORAGE_ROOT": str(tmp_path / "documents"),
            "AERODOCS_INDEX_RUNTIME_ROOT": str(tmp_path / "indexes"),
            "AERODOCS_GENERATION_AUDIT_ROOT": str(tmp_path / "audits"),
        }
    )
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "offline-test", "version": "1"},
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 99, "method": "ping"},
    ]
    result = subprocess.run(
        [sys.executable, "-m", "app.mcp_server"],
        cwd=tmp_path,
        env=environment,
        input="{bad-json\n"
        + "".join(json.dumps(message) + "\n" for message in messages),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0
    frames = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert {1, 2} <= {frame.get("id") for frame in frames}
    assert all(frame.get("jsonrpc") == "2.0" for frame in frames)
    initialization = next(frame for frame in frames if frame.get("id") == 1)
    assert initialization["result"]["serverInfo"] == {
        "name": "AeroDocs",
        "version": "0.1.0",
    }
    assert "Processing request" not in result.stdout
