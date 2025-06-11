"""Dataset classes for MNIST and ASL datasets."""

from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    """PyTorch Dataset for MNIST dataset.

    This dataset loads the MNIST digit dataset from a CSV file.
    """

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
            idx: Index of sample

        Returns:
            Tuple of (image, label)
        """
        # Read a single row from CSV
        chunk = pd.read_csv(
            self.csv_path,
            skiprows=idx + 1,
            nrows=1,
            header=None,
        )

        # Extract label and image
        label = int(chunk.iloc[0, 0])
        image = np.array(chunk.iloc[0, 1:]).reshape(28, 28).astype(np.uint8)

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, label


class ASLHandsDataset(Dataset):
    """PyTorch Dataset for ASL hands dataset.

    This dataset loads the ASL hands dataset from a CSV file.
    """

    def __init__(
        self, csv_path: str, transform: Optional[Callable] = None, rgb: bool = True
    ):
        """Initialize ASL hands dataset.

        Args:
            csv_path: Path to CSV file containing ASL data
            transform: Optional transform to apply to images
            rgb: Whether to convert grayscale images to RGB
        """
        self.csv_path = csv_path
        self.transform = transform
        self.rgb = rgb

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
            idx: Index of sample

        Returns:
            Tuple of (image, label)
        """
        # Read a single row from CSV
        chunk = pd.read_csv(
            self.csv_path,
            skiprows=idx + 1,
            nrows=1,
            header=None,
        )

        # Extract label and image
        label = int(chunk.iloc[0, 0])
        image = np.array(chunk.iloc[0, 1:]).reshape(28, 28).astype(np.uint8)

        # Convert to RGB if needed
        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor
            if self.rgb:
                # (H, W, C) -> (C, H, W)
                image = (
                    torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
            else:
                image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, label
