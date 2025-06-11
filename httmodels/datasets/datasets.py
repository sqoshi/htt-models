"""Unified dataset module for MNIST and ASL datasets."""

from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """PyTorch Dataset for MNIST digit dataset."""

    def __init__(self, csv_path: str, transform: Optional[Callable] = None):
        """Initialize MNIST dataset.

        Args:
            csv_path: Path to CSV file containing MNIST data
            transform: Optional transform to apply to images
        """
        self.csv_path = csv_path
        self.transform = transform

        # Count number of samples (excluding header)
        with open(csv_path, "r") as f:
            self.data_length = sum(1 for _ in f) - 1

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of samples in dataset
        """
        return self.data_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of sample to get

        Returns:
            Tuple of (image, label)
        """
        # Read the specific row
        chunk = pd.read_csv(
            self.csv_path,
            skiprows=idx + 1,
            nrows=1,
            header=None,
        )

        # Extract label and image
        label = int(chunk.iloc[0, 0])
        image = np.array(chunk.iloc[0, 1:]).reshape(28, 28).astype(np.uint8)

        # Convert to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        # Apply transform if available
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label


class ASLDataset(Dataset):
    """PyTorch Dataset for ASL hands dataset."""

    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        """Initialize ASL dataset.

        Args:
            data_path: Path to directory containing ASL images
            transform: Optional transform to apply to images
        """
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Get all image paths and labels
        import os
        from pathlib import Path

        all_images = [
            str(file) for file in Path(data_path).rglob("*") if file.is_file()
        ]
        for img_path in all_images:
            label = img_path.split(os.path.sep)[-2]
            self.image_paths.append(img_path)
            self.labels.append(label)

        # Create label mapping
        self.unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of samples in dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of sample to get

        Returns:
            Tuple of (image, label)
        """
        # Load image
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]

        # Read and process image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # If image can't be loaded, return a blank image
            img = np.zeros((28, 28), dtype=np.uint8)

        # Resize if needed
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28))

        # Convert to tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        # Apply transform if available
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_idx
