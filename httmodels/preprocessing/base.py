"""Base data processing interfaces for all data processors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np


class DataProcessor(ABC):
    """Base class for all data processors."""

    @abstractmethod
    def load(self, source: str) -> Any:
        """Load data from a source."""
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> Dict[str, Any]:
        """Preprocess the data."""
        pass

    @abstractmethod
    def save(self, data: Dict[str, Any], destination: str) -> None:
        """Save the processed data to a destination."""
        pass

    @abstractmethod
    def split(
        self, data: np.ndarray, labels: np.ndarray, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the data into training and testing sets."""
        pass


class ImageProcessor(DataProcessor):
    """Base class for image data processors."""

    def split(
        self, data: np.ndarray, labels: np.ndarray, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the data into training and testing sets."""
        from sklearn.model_selection import train_test_split

        return train_test_split(data, labels, test_size=test_size, random_state=42)
