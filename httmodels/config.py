"""Configuration module for the httmodels package."""

import os

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for the httmodels package."""

    workers: int = Field(
        default_factory=lambda: os.cpu_count() // 2 or 4,
        description="Number of workers for data loading",
    )
    models_path: str = Field("models", description="Path to save models")
    asl_data_path: str = Field(
        "/home/piotr/Documents/htt/images", description="Path to ASL hands dataset"
    )


_settings = None


def settings() -> Settings:
    """Get the settings instance.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
