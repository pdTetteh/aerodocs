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


MAINTENANCE_PROCEDURE = (
    "32-31-00 MAIN LANDING GEAR ACTUATOR - REMOVAL/INSTALLATION\n\n"
    "WARNING: MAKE SURE THE LANDING GEAR GROUND LOCKS ARE INSTALLED. IF NOT, "
    "THE LANDING GEAR CAN RETRACT AND CAUSE INJURY TO PERSONS.\n\n"
    "1. Remove the access panel 192CB.\n"
    "2. Disconnect the electrical connector from the actuator.\n"
    "3. Put a container below the actuator to catch the hydraulic fluid.\n"
    "4. Disconnect the hydraulic lines. Install blanking caps on the open lines.\n"
    "5. Remove the bolts that attach the actuator to the structure.\n"
    "6. Remove the actuator.\n"
    "7. Install the new actuator and torque the attach bolts to 80 in-lb.\n"
    "8. Connect the hydraulic lines and torque the B-nuts to 135 in-lb.\n"
    "9. Remove the blanking caps and connect the electrical connector.\n"
    "10. Do a leak check at 3000 psi.\n\n"
    "CAUTION: DO NOT USE A POWER TOOL TO TORQUE THE BOLTS."
)


def test_numbered_procedure_stays_in_one_chunk_with_its_warnings() -> None:
    document_id = "doc_" + "f" * 64
    chunks = create_chunks(
        [page(document_id, 1, MAINTENANCE_PROCEDURE)],
        processing_profile_id="prf_test",
        maximum_characters=2_000,
        overlap_characters=200,
    )

    assert len(chunks) == 1
    assert "WARNING: MAKE SURE" in chunks[0].text
    assert "7. Install the new actuator" in chunks[0].text
    assert "CAUTION: DO NOT USE A POWER TOOL" in chunks[0].text
    assert chunks[0].section_heading == (
        "32-31-00 MAIN LANDING GEAR ACTUATOR - REMOVAL/INSTALLATION"
    )


def test_long_procedure_keeps_every_step_under_the_task_heading() -> None:
    document_id = "doc_" + "1" * 64
    chunks = create_chunks(
        [page(document_id, 1, MAINTENANCE_PROCEDURE)],
        processing_profile_id="prf_test",
        maximum_characters=400,
        overlap_characters=40,
    )

    assert len(chunks) < 6
    assert all(
        chunk.section_heading
        == "32-31-00 MAIN LANDING GEAR ACTUATOR - REMOVAL/INSTALLATION"
        for chunk in chunks
    )
    assert all(chunk.character_count >= 40 for chunk in chunks)


def test_chunk_never_ends_on_a_safety_paragraph() -> None:
    document_id = "doc_" + "2" * 64
    body = "Inspect the panel fasteners for looseness. " * 6
    text = (
        f"{body.strip()}\n\n"
        "WARNING: DO NOT OPERATE THE ACTUATOR WITH THE PANEL REMOVED.\n\n"
        "Operate the actuator through one full cycle."
    )
    cut = text.index("Operate the actuator") + 5

    chunks = create_chunks(
        [page(document_id, 1, text)],
        processing_profile_id="prf_test",
        maximum_characters=cut,
        overlap_characters=20,
    )

    warning = "WARNING: DO NOT OPERATE"
    last_with_warning = [chunk for chunk in chunks if warning in chunk.text][-1]
    assert "Operate the actuator through" in last_with_warning.text
    assert not chunks[0].text.rstrip().endswith("REMOVED.")
