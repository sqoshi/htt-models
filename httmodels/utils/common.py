"""Utility functions for the httmodels package."""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def setup_logging(
    level: int = logging.INFO, log_file: Optional[str] = None, console: bool = True
) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level
        log_file: Path to log file
        console: Whether to log to console
    """
    handlers = []

    if console:
        handlers.append(logging.StreamHandler())

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def get_device() -> torch.device:
    """Get PyTorch device.

    Returns:
        PyTorch device (cuda or cpu)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model_info(model_path: str, model_info: Dict[str, Any]) -> None:
    """Save model information to file.

    Args:
        model_path: Path to save model info
        model_info: Model information dictionary
    """
    import json

    # Get model directory and base name
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).split(".")[0]

    # Create info file path
    info_path = os.path.join(model_dir, f"{model_name}_info.json")

    # Save info to file
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    logging.info(f"Model info saved to {info_path}")


def load_model_info(model_path: str) -> Dict[str, Any]:
    """Load model information from file.

    Args:
        model_path: Path to model file

    Returns:
        Model information dictionary
    """
    import json

    # Get model directory and base name
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).split(".")[0]

    # Create info file path
    info_path = os.path.join(model_dir, f"{model_name}_info.json")

    # Load info from file
    with open(info_path, "r") as f:
        model_info = json.load(f)

    return model_info
