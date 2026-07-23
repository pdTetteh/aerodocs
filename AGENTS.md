# Repository Guidelines

## Purpose and Safety

AeroDocs is a safety-conscious aviation engineering documentation assistant built from public FAA and EASA documents. It will support evidence retrieval, citation-grounded answers, synthetic defect workflows, and human-reviewed work-order drafting.

Use only public or synthetic data. Never create autonomous aircraft-maintenance decisions or imply that generated content approves maintenance or airworthiness. Future answers must preserve source document title, page, and section. Future work-order approval must require explicit human confirmation.

## Architecture and Scope

Use Python 3.11 with FastAPI, Pydantic, and `pydantic-settings`. Keep API routes, configuration, services, ingestion, retrieval, evaluation, database, work-order, and MCP code separated. Prefer the simplest architecture appropriate to the current milestone.

Do not implement future milestones unless explicitly requested. Planned technologies—PyMuPDF, sentence-transformers, FAISS, SQLite/SQLAlchemy, local LLMs, MCP, Docker, and Azure—are not current requirements. Explain why any new dependency is necessary before adding it. Modify only files relevant to the task.

## Coding and Security

Use type hints for every public function and Pydantic models for API inputs and outputs. Do not silently suppress exceptions; handle them explicitly or allow them to surface with useful context. Never place credentials or secrets in source code. Do not commit `.env` files, credentials, private manuals, generated data, or local caches.

## Development Commands

- `uv sync` — install locked dependencies.
- `uv run uvicorn app.main:app --reload` — start the development API.
- `uv run pytest` — run automated tests.
- `uv run ruff check .` — run lint checks.
- `uv run ruff format --check .` — verify formatting.
- `uv run mypy .` — run static type checks.

Use pytest, pytest-asyncio, and httpx as appropriate. Add automated tests for every implemented feature. Run pytest and Ruff before declaring a task complete.

## Definition of Done

A task is complete when relevant tests, Ruff linting, and formatting checks pass; public functions have appropriate type hints; README instructions remain accurate; and no unrelated files are modified. Report limitations, skipped checks, and unresolved issues honestly.
