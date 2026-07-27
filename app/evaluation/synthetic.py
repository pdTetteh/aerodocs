"""Deterministic evaluation-only PDF source generation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.evaluation.security import AuxiliaryFileError


class SyntheticPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lines: list[str] = Field(min_length=1, max_length=80)

    @field_validator("lines")
    @classmethod
    def safe_lines(cls, lines: list[str]) -> list[str]:
        if any(not line or len(line) > 180 for line in lines):
            raise ValueError("lines must contain 1-180 characters")
        return lines


class SyntheticPdfSpec(BaseModel):
    """Portable source specification with no host-derived fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^src_[a-z0-9][a-z0-9_-]{2,63}$")
    title: str = Field(min_length=1, max_length=120)
    pages: list[SyntheticPage] = Field(min_length=1, max_length=100)


class SyntheticIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str
    document_id: str
    page_count: int


def render_pdf(spec: SyntheticPdfSpec) -> bytes:
    """Serialize a byte-identical, minimal PDF 1.4 for the same specification."""
    objects: list[bytes] = []
    page_count = len(spec.pages)
    page_ids = [4 + index * 2 for index in range(page_count)]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = b" ".join(f"{page_id} 0 R".encode() for page_id in page_ids)
    objects.append(
        b"<< /Type /Pages /Count "
        + str(page_count).encode()
        + b" /Kids ["
        + kids
        + b"] >>"
    )
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>"
    )
    for index, page in enumerate(spec.pages):
        page_id = page_ids[index]
        content_id = page_id + 1
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> "
            + f"/Contents {content_id} 0 R >>".encode()
        )
        commands = [b"BT", b"/F1 11 Tf", b"72 720 Td", b"14 TL"]
        for line_index, line in enumerate(page.lines):
            if line_index:
                commands.append(b"T*")
            commands.append(b"(" + _pdf_string(line) + b") Tj")
        commands.append(b"ET")
        stream = b"\n".join(commands) + b"\n"
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"endstream"
        )
    title = _pdf_string(spec.title)
    objects.append(
        b"<< /Title (" + title + b") /Producer (AeroDocs deterministic evaluator) >>"
    )
    info_id = len(objects)
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, value in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode())
        output.extend(value)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R /Info "
        + str(info_id).encode()
        + b" 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )
    return bytes(output)


def identity(spec: SyntheticPdfSpec) -> SyntheticIdentity:
    """Return stable source identity derived only from serialized bytes."""
    digest = hashlib.sha256(render_pdf(spec)).hexdigest()
    return SyntheticIdentity(
        sha256=digest,
        document_id=f"doc_{digest}",
        page_count=len(spec.pages),
    )


def write_pdf(root: Path, name: str, spec: SyntheticPdfSpec) -> Path:
    """Write a generated PDF to a new contained regular destination."""
    if Path(name).name != name or not name.endswith(".pdf"):
        raise AuxiliaryFileError("unsafe_destination")
    base = root.absolute()
    base.mkdir(parents=True, exist_ok=True)
    if base.is_symlink() or not base.is_dir():
        raise AuxiliaryFileError("unsafe_root")
    path = base / name
    try:
        with path.open("xb") as output:
            output.write(render_pdf(spec))
    except OSError as error:
        raise AuxiliaryFileError("destination_unavailable") from error
    return path


def _pdf_string(value: str) -> bytes:
    encoded = value.encode("cp1252", errors="replace")
    return (
        encoded.replace(b"\\", b"\\\\")
        .replace(b"(", b"\\(")
        .replace(b")", b"\\)")
        .replace(b"\r", b" ")
        .replace(b"\n", b" ")
    )
