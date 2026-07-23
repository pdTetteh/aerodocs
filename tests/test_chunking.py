"""Tests for deterministic chunking and provenance."""

from app.ingestion.chunking import create_chunks
from app.ingestion.models import PageRecord


def page(document_id: str, number: int, text: str) -> PageRecord:
    """Create a normalized page record for chunk tests."""
    return PageRecord(
        document_id=document_id,
        page_number=number,
        text=text,
        character_count=len(text),
    )


def test_chunk_ids_and_order_are_deterministic() -> None:
    pages = [
        page("doc_" + "a" * 64, 1, "paragraph one " * 15),
        page("doc_" + "a" * 64, 2, "paragraph two " * 15),
    ]

    first = create_chunks(
        pages,
        processing_profile_id="prf_test",
        maximum_characters=100,
        overlap_characters=20,
    )
    second = create_chunks(
        pages,
        processing_profile_id="prf_test",
        maximum_characters=100,
        overlap_characters=20,
    )

    assert first == second
    assert [chunk.chunk_order for chunk in first] == list(range(1, len(first) + 1))
    assert all(chunk.chunk_id.startswith("chk_") for chunk in first)
    assert all(chunk.character_count <= 100 for chunk in first)


def test_chunk_overlap_and_text_coverage() -> None:
    source = "alpha beta gamma delta epsilon " * 20
    chunks = create_chunks(
        [page("doc_" + "b" * 64, 1, source)],
        processing_profile_id="prf_test",
        maximum_characters=90,
        overlap_characters=15,
    )

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.text[-15:] == current.text[:15]
    reconstructed = chunks[0].text + "".join(chunk.text[15:] for chunk in chunks[1:])
    assert reconstructed == source


def test_page_transitions_and_heading_context_are_preserved() -> None:
    document_id = "doc_" + "c" * 64
    chunks = create_chunks(
        [
            page(document_id, 1, "1 Scope\n" + "scope text " * 8),
            page(document_id, 2, "continued text " * 8),
            page(document_id, 3, ""),
        ],
        processing_profile_id="prf_test",
        maximum_characters=180,
        overlap_characters=20,
    )

    assert chunks[0].section_heading == "1 Scope"
    assert chunks[0].page_start == 1
    assert any(chunk.page_end == 2 for chunk in chunks)
    assert all(chunk.page_end != 3 for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_heading_starts_a_new_chunk() -> None:
    document_id = "doc_" + "d" * 64
    chunks = create_chunks(
        [
            page(
                document_id,
                1,
                "SYSTEM INTRODUCTION\nIntroductory text.\n\n"
                "2 LIMITATIONS\nRestricted text.",
            )
        ],
        processing_profile_id="prf_test",
        maximum_characters=200,
        overlap_characters=20,
    )

    assert len(chunks) == 2
    assert chunks[0].section_heading == "SYSTEM INTRODUCTION"
    assert chunks[1].section_heading == "2 LIMITATIONS"
    assert chunks[1].text.startswith("2 LIMITATIONS")


def test_repeated_running_header_is_not_section_context() -> None:
    document_id = "doc_" + "e" * 64
    chunks = create_chunks(
        [
            page(document_id, 1, "PUBLIC TEST MANUAL\nPage one text."),
            page(document_id, 2, "PUBLIC TEST MANUAL\nPage two text."),
        ],
        processing_profile_id="prf_test",
        maximum_characters=200,
        overlap_characters=20,
    )

    assert chunks
    assert all(chunk.section_heading is None for chunk in chunks)
