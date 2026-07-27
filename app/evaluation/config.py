"""Evaluation-only path configuration kept outside production settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationSettings(BaseSettings):
    """Configured roots for ID-based evaluation artifact resolution."""

    model_config = SettingsConfigDict(
        env_prefix="AERODOCS_EVALUATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    dataset_root: Path = Path("data/evaluation")
    report_root: Path = Path("data/runtime/evaluation/reports")
    performance_warm_repetitions: int = Field(default=3, ge=1, le=100)
