"""MNIST dataset processor."""

import logging

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from httmodels.preprocessing.base import ImageProcessor


class MNISTProcessor(ImageProcessor):
    """Processor for MNIST dataset."""

    def __init__(self, root="./data", download=True):
        """Initialize MNIST processor.

        Args:
            root: Root directory for data
            download: Whether to download the dataset if not present
        """
        self.root = root
        self.download = download
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def load(self, source=None):
        """Load MNIST dataset.

        Args:
            source: Optional source path (uses self.root if not provided)

        Returns:
            MNIST dataset
        """
        source = source or self.root
        logging.info(f"Loading MNIST dataset from {source}")

        try:
            train_dataset = datasets.MNIST(
                root=source,
                train=True,
                download=self.download,
                transform=self.transform,
            )
            test_dataset = datasets.MNIST(
                root=source,
                train=False,
                download=self.download,
                transform=self.transform,
            )

            return {"train": train_dataset, "test": test_dataset}
        except Exception as e:
            logging.error(f"Error loading MNIST dataset: {e}")
            raise

    def preprocess(self, data):
        """Preprocess MNIST dataset.

        Args:
            data: MNIST dataset dictionary with 'train' and 'test' keys

        Returns:
            Dictionary with preprocessed data and labels
        """
        logging.info("Preprocessing MNIST dataset")

        # Extract train data
        train_data = []
        train_labels = []
        for img, label in data["train"]:
            train_data.append(img.numpy())
            train_labels.append(label)

        # Extract test data
        test_data = []
        test_labels = []
        for img, label in data["test"]:
            test_data.append(img.numpy())
            test_labels.append(label)

        return {
            "train_data": np.array(train_data),
            "train_labels": np.array(train_labels),
            "test_data": np.array(test_data),
            "test_labels": np.array(test_labels),
        }

    def save(self, data, destination):
        """Save preprocessed data.

        Args:
            data: Preprocessed data dictionary
            destination: Destination path

        Returns:
            None
        """
        logging.info(f"Saving preprocessed MNIST data to {destination}")
        np.savez(
            destination,
            train_data=data["train_data"],
            train_labels=data["train_labels"],
            test_data=data["test_data"],
            test_labels=data["test_labels"],
        )

    def split(self, data, labels, test_size=0.2):
        """Split data into train and test sets.

        Args:
            data: Input data
            labels: Input labels
            test_size: Proportion of data to use for testing

        Returns:
            x_train, x_test, y_train, y_test
        """
        logging.info(f"Splitting data with test_size={test_size}")
        return train_test_split(
            data, labels, test_size=test_size, random_state=42, stratify=labels
        )
