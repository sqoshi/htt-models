"""Data loaders for different model types."""

import logging
import os
import random
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from httmodels.dataloaders.landmarkstransformers import HandLandmarkTransformer


def _get_sample_indices(dataset, sample_ratio):
    """Get indices for a sampled subset of data.

    Args:
        dataset: Dataset to sample from
        sample_ratio: Ratio of data to use

    Returns:
        List of indices
    """
    total_samples = len(dataset)
    sample_size = int(total_samples * sample_ratio)
    return random.sample(range(total_samples), sample_size)


class PyTorchDataLoader:
    """Data loader for PyTorch models."""

    @staticmethod
    def create_dataloaders(
        dataset: Dataset,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders from a dataset.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size for dataloaders
            train_ratio: Ratio of data to use for training
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for dataloaders
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Calculate sizes
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size

        # Set number of workers
        if num_workers is None:
            num_workers = os.cpu_count() or 4

        # Split dataset
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader


class MLDataLoader(DataLoader):
    """Data loader for machine learning models."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        sample_ratio: float = 1.0,
        augmentation: bool = False,
        num_workers: Optional[int] = None,
    ):
        """Initialize ML data loader.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size for data loading
            shuffle: Whether to shuffle the data
            sample_ratio: Ratio of data to use
            augmentation: Whether to apply data augmentation
            num_workers: Number of workers for data loading
        """
        self.augmentation = augmentation
        self.transformer = HandLandmarkTransformer() if augmentation else None

        # Set number of workers
        if num_workers is None:
            num_workers = os.cpu_count() or 4

        # Sample subset of data if needed
        sampled_dataset = (
            Subset(dataset, _get_sample_indices(dataset, sample_ratio))
            if sample_ratio < 1.0
            else dataset
        )

        super().__init__(
            sampled_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def __iter__(self):
        """Iterate over batches of data.

        Yields:
            Tuple of (X_batch, y_batch)
        """
        for image, label in super().__iter__():
            if self.augmentation and self.transformer:
                landmarks = self.transformer.transform(image)
                if landmarks:
                    yield landmarks, np.int32(label)
                else:
                    continue
            else:
                yield image, label


class LeNetDataLoader(DataLoader):

    def __init__(
        self, dataset, batch_size=32, shuffle=True, augmentation=True, sample_ratio=1.0
    ):
        self.augmentation = augmentation
        sampled_dataset = (
            torch.utils.data.Subset(dataset, _get_sample_indices(dataset, sample_ratio))
            if sample_ratio < 1.0
            else dataset
        )
        super().__init__(
            sampled_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=os.cpu_count(),
        )

        base_transforms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]

        if augmentation:
            aug_transforms = [
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
            ]
            self.transform = transforms.Compose(aug_transforms + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __iter__(self):
        for image, label in super().__iter__():
            image = self.transform(image)
            yield image, label


class ResNetDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_size=32, shuffle=True, augmentation=True, sample_ratio=1.0
    ):
        self.augmentation = augmentation
        sampled_dataset = (
            torch.utils.data.Subset(dataset, _get_sample_indices(dataset, sample_ratio))
            if sample_ratio < 1.0
            else dataset
        )
        super().__init__(
            sampled_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=os.cpu_count(),
        )

        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

        if augmentation:
            aug_transforms = [
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
            ]
            self.transform = transforms.Compose(aug_transforms + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __iter__(self):
        for image, label in super().__iter__():
            image = self.transform(image)
            yield image, label
