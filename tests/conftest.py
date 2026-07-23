"""Shared pytest fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings

SETTING_ENVIRONMENT_VARIABLES = (
    "AERODOCS_APP_NAME",
    "AERODOCS_APP_VERSION",
    "AERODOCS_LOG_LEVEL",
)


@pytest.fixture
def default_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Settings:
    """Return settings isolated from local environment and dotenv files."""
    monkeypatch.chdir(tmp_path)
    for variable in SETTING_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    # `_env_file` is supported at runtime but absent from the generated signature.
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def client(default_settings: Settings) -> Iterator[TestClient]:
    """Return an HTTP client for an isolated application instance."""
    from app.factory import create_app

    with TestClient(create_app(default_settings)) as test_client:
        yield test_client
