"""The container entry point must fail closed without valid OIDC settings."""

import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI

OIDC_ENVIRONMENT = {
    "AERODOCS_SECURITY_MODE": "oidc",
    "AERODOCS_OIDC_ISSUER": "https://identity.example.test/",
    "AERODOCS_OIDC_AUDIENCE": "urn:aerodocs:deployment:alpha:api",
    "AERODOCS_OIDC_JWKS_URL": "https://identity.example.test/jwks.json",
    "AERODOCS_OIDC_VIEWER_GROUP": "aerodocs-deployment-alpha-evidence-readers",
    "AERODOCS_OIDC_OPERATOR_GROUP": "aerodocs-deployment-alpha-document-operators",
}


def _import_secure_main() -> object:
    sys.modules.pop("app.secure_main", None)
    return importlib.import_module("app.secure_main")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(OIDC_ENVIRONMENT) + ["AERODOCS_SECURITY_MODE"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def test_secure_entry_point_refuses_to_start_without_oidc() -> None:
    with pytest.raises(ValueError, match="requires oidc mode"):
        _import_secure_main()


def test_secure_entry_point_refuses_incomplete_oidc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AERODOCS_SECURITY_MODE", "oidc")

    with pytest.raises(ValueError, match="complete OIDC settings"):
        _import_secure_main()


def test_secure_entry_point_builds_with_valid_oidc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in OIDC_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    module = _import_secure_main()

    assert isinstance(module.app, FastAPI)  # type: ignore[attr-defined]
    assert module.app.openapi_url is None  # type: ignore[attr-defined]
