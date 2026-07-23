# AeroDocs

AeroDocs is a safety-conscious aviation engineering documentation assistant.
It is designed to support evidence retrieval, citation-grounded question
answering, synthetic defect workflows, and human-reviewed work-order drafting
using public FAA and EASA material.

This first milestone provides only the FastAPI application foundation and a
health endpoint. It does not make aircraft-maintenance decisions or approve
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

## Verification

```shell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

Only public or synthetic data may be used. Do not commit `.env` files,
credentials, private manuals, or generated data.

## Document data

Intentionally tracked public FAA and EASA documents may eventually be placed
under `data/public/`. Private, temporary, or generated material must be placed
under `data/private/`, `data/temporary/`, or `data/generated/`; these locations
are ignored by Git. Private or proprietary aviation manuals must never be
committed.
