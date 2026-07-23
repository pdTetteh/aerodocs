"""Tests for application creation."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings


def test_create_app_returns_configured_fastapi_application(
    default_settings: Settings,
) -> None:
    from app.factory import create_app

    application = create_app(default_settings)

    assert isinstance(application, FastAPI)
    assert application.title == "AeroDocs"
    assert application.version == "0.1.0"
    assert "/health" in application.openapi()["paths"]


def test_application_metadata_matches_health_response(
    default_settings: Settings,
) -> None:
    from app.factory import create_app

    application = create_app(default_settings)

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.json()["name"] == application.title
    assert response.json()["version"] == application.version


def test_application_factories_use_isolated_storage_roots(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    from app.factory import create_app

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    create_app(
        default_settings.model_copy(update={"document_storage_root": first_root})
    )
    create_app(
        default_settings.model_copy(update={"document_storage_root": second_root})
    )

    assert (first_root / "documents").is_dir()
    assert (second_root / "documents").is_dir()
