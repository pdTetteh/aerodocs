"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from app.api.health import HealthResponse


def test_health_returns_required_response(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "name": "AeroDocs",
        "version": "0.1.0",
        "status": "healthy",
    }


def test_health_response_schema_and_content_type(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers["content-type"].split(";")[0] == "application/json"
    assert HealthResponse.model_validate(response.json()).status == "healthy"

    openapi = client.get("/openapi.json").json()
    response_schema = openapi["paths"]["/health"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert response_schema == {
        "$ref": "#/components/schemas/HealthResponse",
    }
