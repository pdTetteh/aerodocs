"""Deterministic character-based chunking with page provenance."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass

from app.ingestion.headings import find_headings
from app.ingestion.models import ChunkRecord, PageRecord


@dataclass(frozen=True)
class _PageSpan:
    start: int
    end: int
    page_number: int


def create_chunks(
    pages: list[PageRecord],
    *,
    processing_profile_id: str,
    maximum_characters: int,
    overlap_characters: int,
) -> list[ChunkRecord]:
    """Create deterministic ordered chunks without dropping extracted text."""
    if maximum_characters <= 0:
        raise ValueError("maximum_characters must be positive")
    if not 0 <= overlap_characters < maximum_characters:
        raise ValueError("overlap_characters must be smaller than maximum_characters")

    document_id = pages[0].document_id if pages else ""
    page_headings = {
        page.page_number: find_headings(page.text) for page in pages if page.text
    }
    running_headers = {
        heading
        for heading, count in Counter(
            heading
            for headings in page_headings.values()
            for offset, heading in headings
            if offset == 0
        ).items()
        if count > 1
    }
    stream_parts: list[str] = []
    page_spans: list[_PageSpan] = []
    heading_offsets: list[tuple[int, str]] = []
    stream_length = 0

    for page in pages:
        if not page.text:
            continue
        if stream_parts:
            separator = "\n\n"
            stream_parts.append(separator)
            stream_length += len(separator)
        page_start = stream_length
        stream_parts.append(page.text)
        stream_length += len(page.text)
        page_spans.append(_PageSpan(page_start, stream_length, page.page_number))
        heading_offsets.extend(
            (page_start + offset, heading)
            for offset, heading in page_headings[page.page_number]
            if not (offset == 0 and heading in running_headers)
        )

    stream = "".join(stream_parts)
    if not stream:
        return []

    chunks: list[ChunkRecord] = []
    start = 0
    while start < len(stream):
        target_end = min(start + maximum_characters, len(stream))
        end, heading_boundary = _choose_end(
            stream,
            start,
            target_end,
            heading_offsets,
            maximum_characters,
        )
        if end <= start:
            end = target_end
            heading_boundary = False

        text = stream[start:end]
        if not text.strip():
            start = end
            continue

        intersecting_pages = [
            span.page_number
            for span in page_spans
            if span.start < end and span.end > start
        ]
        if not intersecting_pages:
            start = end
            continue

        section_heading = _heading_at_offset(heading_offsets, start)
        chunk_order = len(chunks) + 1
        page_start = min(intersecting_pages)
        page_end = max(intersecting_pages)
        chunk_id = _chunk_identifier(
            document_id=document_id,
            processing_profile_id=processing_profile_id,
            maximum_characters=maximum_characters,
            overlap_characters=overlap_characters,
            chunk_order=chunk_order,
            page_start=page_start,
            page_end=page_end,
            section_heading=section_heading,
            text=text,
        )
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                page_start=page_start,
                page_end=page_end,
                section_heading=section_heading,
                text=text,
                character_count=len(text),
                chunk_order=chunk_order,
            )
        )

        if end == len(stream):
            break
        next_start = end if heading_boundary else max(0, end - overlap_characters)
        start = end if next_start <= start else next_start

    return chunks


def _choose_end(
    stream: str,
    start: int,
    target_end: int,
    headings: list[tuple[int, str]],
    maximum_characters: int,
) -> tuple[int, bool]:
    for offset, _heading in headings:
        if start < offset <= target_end:
            return offset, True

    if target_end == len(stream):
        return target_end, False

    minimum_break = start + int(maximum_characters * 0.6)
    for separator in ("\n\n", "\n", " "):
        boundary = stream.rfind(separator, minimum_break, target_end)
        if boundary != -1:
            return boundary + len(separator), False
    return target_end, False


def _heading_at_offset(
    headings: list[tuple[int, str]],
    offset: int,
) -> str | None:
    current: str | None = None
    for heading_offset, heading in headings:
        if heading_offset > offset:
            break
        current = heading
    return current


def _chunk_identifier(
    *,
    document_id: str,
    processing_profile_id: str,
    maximum_characters: int,
    overlap_characters: int,
    chunk_order: int,
    page_start: int,
    page_end: int,
    section_heading: str | None,
    text: str,
) -> str:
    identity = {
        "document_id": document_id,
        "processing_profile_id": processing_profile_id,
        "maximum_characters": maximum_characters,
        "overlap_characters": overlap_characters,
        "chunk_order": chunk_order,
        "page_start": page_start,
        "page_end": page_end,
        "section_heading": section_heading,
        "text": text,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"chk_{hashlib.sha256(encoded).hexdigest()}"
