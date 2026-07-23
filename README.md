# AeroDocs

AeroDocs is a safety-conscious aviation engineering documentation assistant.
It is designed to support evidence retrieval, citation-grounded question
answering, synthetic defect workflows, and human-reviewed work-order drafting
using public FAA and EASA material.

Milestone 2 provides the FastAPI foundation, health reporting, secure local PDF
upload, page-level extraction, metadata preservation, and deterministic text
chunking. It does not make aircraft-maintenance decisions or approve
maintenance or airworthiness.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

## Local development

Install the locked dependencies:

```shell
uv sync
```

Copy `.env.example` to `.env` if local configuration overrides are needed.
Settings use the `AERODOCS_` environment-variable prefix. Never place secrets in
either file.

Start the API:

```shell
uv run uvicorn app.main:app --reload
```

Verify the service:

```shell
curl http://127.0.0.1:8000/health
```

The default response is:

```json
{"name":"AeroDocs","version":"0.1.0","status":"healthy"}
```

## PDF ingestion

Only public or otherwise authorized PDF documents may be ingested. Upload a
document with:

```shell
curl -X POST http://127.0.0.1:8000/documents \
  -H "accept: application/json" \
  -F "file=@public-manual.pdf;type=application/pdf"
```

The response contains a deterministic document identifier and normalized
metadata. Use that identifier to retrieve stored representations:

```shell
curl http://127.0.0.1:8000/documents/doc_<sha256>
curl "http://127.0.0.1:8000/documents/doc_<sha256>/pages?offset=0&limit=100"
curl "http://127.0.0.1:8000/documents/doc_<sha256>/chunks?offset=0&limit=100"
```

Accepted PDFs are stored under the configured
`AERODOCS_DOCUMENT_STORAGE_ROOT` (default `data/runtime`) using atomic
publication. Each completed document contains the source PDF, metadata, pages,
chunks, and a versioned integrity manifest with artifact hashes and sizes.
Reads and duplicate checks validate that manifest. Runtime files remain local
and are ignored by Git.

Uploads must be unencrypted, use `application/pdf`, begin with a PDF signature,
and contain at least one page. The PDF-file limit defaults to 25 MiB; a separate
27 MiB request-body limit allows multipart overhead and is enforced before
multipart parsing. Extraction defaults to at most 1,000 pages, 500,000
characters per page, and 10,000,000 characters total. Limits are configurable
with the `AERODOCS_` settings shown in `.env.example`; content is rejected, not
truncated, when a limit is exceeded.

Duplicate source bytes with the same processing profile return the existing
document. A profile consists of the processing, extraction, and chunking format
versions plus chunk size and overlap. The same bytes with a different profile
return HTTP 409 and identify the stored profile. Automatic reprocessing and
version migration are intentionally unsupported in Milestone 2.

OCR is not implemented; scanned or image-only pages can have empty text and a
warning. PDF metadata and detected headings are best-effort context, not
authoritative engineering data. Local no-follow and containment checks reduce
symlink risks, but filesystem checks and operations retain an unavoidable local
TOCTOU window. The runtime root must therefore be writable only by the AeroDocs
process or a trusted operator.

## Verification

```shell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

Only public, synthetic, or explicitly authorized data may be used. Do not
commit `.env` files, credentials, private manuals, runtime documents, or
generated data. AeroDocs does not replace approved manufacturer maintenance
documentation or qualified engineering judgment.

## Document data

Intentionally tracked public FAA and EASA documents may eventually be placed
under `data/public/`. Private, temporary, or generated material must be placed
under `data/private/`, `data/temporary/`, or `data/generated/`; these locations
are ignored by Git. Private or proprietary aviation manuals must never be
committed.
