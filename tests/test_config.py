"""Tests for application settings."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_default_settings(default_settings: Settings) -> None:
    assert default_settings.app_name == "AeroDocs"
    assert default_settings.app_version == "0.1.0"
    assert default_settings.log_level == "INFO"


def test_environment_variables_override_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AERODOCS_APP_NAME", "AeroDocs Test")
    monkeypatch.setenv("AERODOCS_APP_VERSION", "9.9.9")
    monkeypatch.setenv("AERODOCS_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.app_name == "AeroDocs Test"
    assert settings.app_version == "9.9.9"
    assert settings.log_level == "DEBUG"


def test_settings_are_immutable(default_settings: Settings) -> None:
    with pytest.raises(ValidationError):
        default_settings.app_name = "Changed"


def test_isolated_settings_import_ignores_dotenv_but_respects_environment(
    tmp_path: Path,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "AERODOCS_APP_NAME=Contaminated\nAERODOCS_APP_VERSION=9.9.9\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    for variable in (
        "AERODOCS_APP_NAME",
        "AERODOCS_APP_VERSION",
        "AERODOCS_LOG_LEVEL",
    ):
        environment.pop(variable, None)
    environment["AERODOCS_LOG_LEVEL"] = "DEBUG"
    environment["PYTHONPATH"] = str(Path(__file__).parents[1])

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from app.core.config import Settings; "
                "settings = Settings(_env_file=None); "
                "print(json.dumps(settings.model_dump()))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "app_name": "AeroDocs",
        "app_version": "0.1.0",
        "log_level": "DEBUG",
    }
