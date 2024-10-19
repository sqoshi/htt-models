import os

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    workers: int = Field(
        default_factory=lambda: os.cpu_count() or 4,
        description="Number of workers for data loading",
    )
    models_path: str = Field("models", description="Path to save models")


_settings = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
