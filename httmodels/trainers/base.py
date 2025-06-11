"""Base trainer interfaces for all trainers."""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


class BaseTrainer(ABC):
    """Base class for all model trainers."""

    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions with the trained model."""
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate the model on the test data."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Save the trained model to a file."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Load a trained model from a file."""
        pass


class MLTrainer(BaseTrainer):
    """Trainer for scikit-learn models."""

    def __init__(self, model):
        """Initialize the trainer with a scikit-learn model."""
        self.model = model

    def fit(self, X_train, y_train, **kwargs):
        """Train the scikit-learn model on the given data."""
        if hasattr(X_train, "numpy") and hasattr(y_train, "numpy"):
            X_train = X_train.numpy()
            y_train = y_train.numpy()

        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        """Make predictions with the trained scikit-learn model."""
        if hasattr(X, "numpy"):
            X = X.numpy()

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate the scikit-learn model on the test data."""
        if hasattr(X_test, "numpy") and hasattr(y_test, "numpy"):
            X_test = X_test.numpy()
            y_test = y_test.numpy()

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy * 100

    def save(self, filepath):
        """Save the trained scikit-learn model to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        logging.debug(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load a trained scikit-learn model from a file."""
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        logging.debug(f"Model loaded from {filepath}")


class DLTrainer(BaseTrainer):
    """Trainer for PyTorch deep learning models."""

    def __init__(
        self,
        model,
        learning_rate=0.001,
        step_size=7,
        gamma=0.1,
        device=None,
    ):
        """Initialize the trainer with a PyTorch model and training parameters."""
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

    def fit(self, X_train, y_train, **kwargs):
        """Train the PyTorch model on the given data."""
        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 32)

        # Handle both tensor and numpy inputs
        if not torch.is_tensor(X_train):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not torch.is_tensor(y_train):
            y_train = torch.tensor(y_train, dtype=torch.long)

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        # Create DataLoader for batching
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            running_corrects = 0

            for images, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            self.scheduler.step()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            logging.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}"
            )

        return self.model

    def predict(self, X):
        """Make predictions with the trained PyTorch model."""
        self.model.eval()

        # Handle numpy input
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            _, preds = torch.max(outputs, 1)

        return preds.cpu().numpy()

    def evaluate(self, X_test, y_test):
        """Evaluate the PyTorch model on the test data."""
        # Handle both tensor and numpy inputs
        if not torch.is_tensor(X_test):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        if not torch.is_tensor(y_test):
            y_test = torch.tensor(y_test, dtype=torch.long)

        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        # Create DataLoader for batching
        dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        test_loss = running_loss / len(test_loader.dataset)
        test_acc = running_corrects.double() / len(test_loader.dataset)
        test_acc = test_acc.item() * 100

        logging.info(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.2f}%")
        return test_acc

    def save(self, filepath):
        """Save the trained PyTorch model to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        logging.debug(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load a trained PyTorch model from a file."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        logging.debug(f"Model loaded from {filepath}")
