"""Application settings, loaded from environment variables and the .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> parents[0]=core, [1]=app, [2]=repo root.
# Derived from __file__ so it is absolute regardless of the current working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "Nexa"
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    host: str = "127.0.0.1"
    port: int = 8765

    # Audio (consumed in Phase 2)
    sample_rate: int = 16000
    channels: int = 1
    frame_size: int = 1280
    ring_buffer_seconds: float = 10.0
    wake_word_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    input_device_index: int | None = 8

    # Paths
    allowed_file_roots: list[Path] = Field(default_factory=list)
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    logs_dir: Path = PROJECT_ROOT / "logs"

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v

    @field_validator("project_root", "data_dir", "logs_dir")
    @classmethod
    def _absolute_path(cls, v: Path) -> Path:
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @field_validator("allowed_file_roots")
    @classmethod
    def _absolute_paths(cls, v: list[Path]) -> list[Path]:
        return [p if p.is_absolute() else (PROJECT_ROOT / p).resolve() for p in v]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings
