"""Tests for application creation."""

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
