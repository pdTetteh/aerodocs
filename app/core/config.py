"""Application configuration loaded from environment variables."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


class Settings(BaseSettings):
    """AeroDocs settings with safe local defaults."""

    model_config = SettingsConfigDict(
        env_prefix="AERODOCS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "AeroDocs"
    app_version: str = "0.1.0"
    log_level: LogLevel = "INFO"
