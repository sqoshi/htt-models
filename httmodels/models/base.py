"""Base model interfaces for all models in the project."""

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(ABC):
    """Base class for all models in the project."""

    @abstractmethod
    def forward(self, x):
        """Forward pass through the model."""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save the model to a file."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load the model from a file."""
        pass


class PyTorchModel(BaseModel, nn.Module):
    """Base class for PyTorch models."""

    def __init__(self):
        nn.Module.__init__(self)

    def save(self, filepath: str):
        """Save the PyTorch model to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str):
        """Load the PyTorch model from a file."""
        self.load_state_dict(torch.load(filepath, map_location=torch.device("cpu")))
        self.eval()


class SklearnModel(BaseModel):
    """Base class for scikit-learn models."""

    def __init__(self, model):
        self.model = model

    def forward(self, x):
        """Forward pass for sklearn models."""
        return self.model.predict(x)

    def save(self, filepath: str):
        """Save the sklearn model to a file."""
        import pickle

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str):
        """Load the sklearn model from a file."""
        import pickle

        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
