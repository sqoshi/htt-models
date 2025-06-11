"""Training context for model training."""

import logging
import os

import numpy as np

from httmodels.config import settings
from httmodels.trainers.base import BaseTrainer


class TrainingContext:
    """Context for training models.

    This class provides a consistent interface for training and evaluating
    models, regardless of the underlying model type.
    """

    def __init__(self, trainer: BaseTrainer):
        """Initialize training context.

        Args:
            trainer: Model trainer to use
        """
        self.trainer = trainer
        self.models_path = settings().models_path
        os.makedirs(self.models_path, exist_ok=True)

    def set_trainer(self, trainer: BaseTrainer):
        """Set a new trainer.

        Args:
            trainer: New model trainer to use
        """
        self.trainer = trainer

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Train the model.

        Args:
            x_train: Training data features
            y_train: Training data labels
            **kwargs: Additional training parameters
        """
        logging.info("Starting model training...")
        self.trainer.fit(x_train, y_train, **kwargs)
        logging.info("Model training completed")

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        """Evaluate the model.

        Args:
            x_test: Test data features
            y_test: Test data labels

        Returns:
            Model accuracy (percentage)
        """
        logging.info("Evaluating model...")
        accuracy = self.trainer.evaluate(x_test, y_test)
        logging.info(f"Model accuracy: {accuracy:.2f}%")
        return accuracy

    def save(self, filepath: str):
        """Save the model.

        Args:
            filepath: Path to save the model
        """
        full_path = os.path.join(self.models_path, filepath)
        logging.info(f"Saving model to {full_path}")
        self.trainer.save(full_path)

    def load(self, filepath: str):
        """Load the model.

        Args:
            filepath: Path to load the model from
        """
        full_path = os.path.join(self.models_path, filepath)
        logging.info(f"Loading model from {full_path}")
        self.trainer.load(full_path)
