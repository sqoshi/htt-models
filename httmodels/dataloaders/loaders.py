"""Data loaders for different model types."""

import os
from typing import Callable, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms

from httmodels.dataloaders.landmarkstransformers import HandLandmarkTransformer


def get_dataloader(
    dataset: Union[Dataset, Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """Create a PyTorch DataLoader from a dataset or tensor data.

    Args:
        dataset: PyTorch Dataset or tuple of (X, y) tensors
        batch_size: Batch size for the data loader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for loading data

    Returns:
        PyTorch DataLoader
    """
    # Set number of workers if not specified
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 4)

    # Handle tensor inputs by creating a TensorDataset
    if isinstance(dataset, tuple) and len(dataset) == 2:
        X, y = dataset
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X, y)

    # Create and return DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def split_dataset(
    dataset: Dataset, train_ratio: float = 0.8, seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """Split a dataset into training and validation sets.

    Args:
        dataset: PyTorch Dataset to split
        train_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Calculate sizes
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Split dataset
    return random_split(dataset, [train_size, val_size])


def create_train_val_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from a dataset.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of data to use for training
        shuffle: Whether to shuffle the training data
        num_workers: Number of workers for dataloaders
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_dataset, val_dataset = split_dataset(dataset, train_ratio, seed)

    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# Transformation factories for different model types
def get_mnist_transforms(augmentation: bool = False) -> Callable:
    """Get transformations for MNIST dataset.

    Args:
        augmentation: Whether to apply data augmentation

    Returns:
        Composed transforms
    """
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]

    if augmentation:
        aug_transforms = [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
        return transforms.Compose(aug_transforms + base_transforms)

    return transforms.Compose(base_transforms)


def get_asl_transforms(
    model_type: str = "resnet", augmentation: bool = False
) -> Callable:
    """Get transformations for ASL hands dataset.

    Args:
        model_type: Type of model to create transforms for ("resnet" or "lenet")
        augmentation: Whether to apply data augmentation

    Returns:
        Composed transforms
    """
    if model_type.lower() == "resnet":
        # ResNet expects 3 channels and 224x224 images
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
            return transforms.Compose(aug_transforms + base_transforms)

        return transforms.Compose(base_transforms)

    else:  # LeNet or default
        # LeNet expects 1 channel and 28x28 images
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
            return transforms.Compose(aug_transforms + base_transforms)

        return transforms.Compose(base_transforms)


def get_hand_landmark_transforms() -> HandLandmarkTransformer:
    """Get a transformer for extracting hand landmarks.

    Returns:
        HandLandmarkTransformer instance
    """
    return HandLandmarkTransformer()
