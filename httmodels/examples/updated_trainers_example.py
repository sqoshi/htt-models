"""Example implementation using the updated trainers with dataloaders.

This example shows how to train models using the updated trainers
with the new dataloader functionality.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch

from httmodels.config import settings
from httmodels.dataloaders import get_mnist_transforms, get_asl_transforms
from httmodels.preprocessing.aslhands import ASLHandsProcessor
from httmodels.preprocessing.mnist import MNISTProcessor
from httmodels.trainers.adaboost import AdaBoostTrainer
from httmodels.trainers.context import TrainingContext
from httmodels.trainers.lenet import LeNetTrainer
from httmodels.trainers.resnet import ResNetTrainer
from httmodels.trainers.rf import RandomForestTrainer
from httmodels.utils import get_device, save_model_info, setup_logging


def train_lenet_mnist(
    data_path: str,
    epochs: int = 10,
    batch_size: int = 64,
    save_path: Optional[str] = None
) -> float:
    """Train LeNet model on MNIST dataset using the updated trainers.

    Args:
        data_path: Path to MNIST data
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_path: Path to save model (if None, will use default path)

    Returns:
        Model accuracy on test set
    """
    device = get_device()
    logging.info(f"Using device: {device}")

    # Process data
    processor = MNISTProcessor(root=data_path, apply_augmentation=True)
    data = processor.load()
    processed_data = processor.preprocess(data)

    x_train = processed_data["train_data"]
    y_train = processed_data["train_labels"]
    x_test = processed_data["test_data"]
    y_test = processed_data["test_labels"]

    # Initialize trainer
    trainer = LeNetTrainer(input_shape=(1, 28, 28), num_classes=10, device=device)

    # Train and evaluate with context (uses the updated dataloader methods)
    context = TrainingContext(trainer)
    context.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    accuracy = context.evaluate(x_test, y_test)

    # Save model
    if save_path is None:
        save_path = "lenet_mnist.pth"

    context.save(save_path)
    logging.info(f"Model saved with accuracy: {accuracy:.2f}%")

    # Save model info
    save_model_info(
        os.path.join(settings().models_path, save_path),
        {
            "model_type": "LeNet",
            "dataset": "MNIST",
            "input_shape": [1, 28, 28],
            "num_classes": 10,
            "accuracy": float(accuracy),
            "epochs": epochs,
            "batch_size": batch_size,
        },
    )

    return accuracy


def train_resnet_asl(
    data_path: str,
    epochs: int = 15,
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> float:
    """Train ResNet model on ASL hands dataset using the updated trainers.

    Args:
        data_path: Path to ASL hands data
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_path: Path to save model (if None, will use default path)

    Returns:
        Model accuracy on test set
    """
    device = get_device()
    logging.info(f"Using device: {device}")

    # Process data
    processor = ASLHandsProcessor(apply_augmentation=True)
    img_paths = processor.load(data_path)
    data = processor.preprocess(img_paths)

    # Get processed data
    x_train, x_test, y_train, y_test = processor.split(
        data["data"], data["labels"]
    )

    # Get label mapping
    label_to_int = data["label_mapping"]
    
    # Convert labels to integers
    y_train = np.array([label_to_int[label] for label in y_train])
    y_test = np.array([label_to_int[label] for label in y_test])

    # Reshape for CNN and convert to RGB (3 channels for ResNet)
    x_train = np.array(x_train).reshape(-1, 1, 28, 28)
    x_test = np.array(x_test).reshape(-1, 1, 28, 28)
    
    # Expand channels for ResNet (requires 3 channels)
    x_train = np.repeat(x_train, 3, axis=1)
    x_test = np.repeat(x_test, 3, axis=1)

    # Initialize trainer
    trainer = ResNetTrainer(num_classes=len(label_to_int), device=device)

    # Train and evaluate
    context = TrainingContext(trainer)
    context.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    accuracy = context.evaluate(x_test, y_test)

    # Save model
    if save_path is None:
        save_path = "resnet_asl.pth"

    context.save(save_path)
    logging.info(f"Model saved with accuracy: {accuracy:.2f}%")

    # Save model info
    save_model_info(
        os.path.join(settings().models_path, save_path),
        {
            "model_type": "ResNet",
            "dataset": "ASL Hands",
            "input_shape": [3, 28, 28],
            "num_classes": len(label_to_int),
            "class_mapping": label_to_int,
            "accuracy": float(accuracy),
            "epochs": epochs,
            "batch_size": batch_size,
        },
    )

    return accuracy


def train_rf_asl(
    data_path: str,
    n_estimators: int = 100,
    save_path: Optional[str] = None
) -> float:
    """Train Random Forest model on ASL hands dataset.

    Args:
        data_path: Path to ASL hands data
        n_estimators: Number of trees in the forest
        save_path: Path to save model (if None, will use default path)

    Returns:
        Model accuracy on test set
    """
    # Process data
    processor = ASLHandsProcessor(apply_augmentation=False)
    img_paths = processor.load(data_path)
    data = processor.preprocess(img_paths)

    # Get label mapping
    label_to_int = data["label_mapping"]
    
    # Split data
    x_train, x_test, y_train, y_test = processor.split(
        data["data"], data["labels"]
    )
    
    # Convert labels to integers
    y_train = [label_to_int[label] for label in y_train]
    y_test = [label_to_int[label] for label in y_test]
    
    # Flatten images for Random Forest
    x_train = np.array([img.flatten() for img in x_train])
    x_test = np.array([img.flatten() for img in x_test])

    # Initialize trainer
    trainer = RandomForestTrainer(n_estimators=n_estimators)

    # Train and evaluate
    context = TrainingContext(trainer)
    context.fit(x_train, y_train)
    accuracy = context.evaluate(x_test, y_test)

    # Save model
    if save_path is None:
        save_path = "rf_asl.pickle"

    context.save(save_path)
    logging.info(f"Model saved with accuracy: {accuracy:.2f}%")

    # Save model info
    save_model_info(
        os.path.join(settings().models_path, save_path),
        {
            "model_type": "RandomForest",
            "dataset": "ASL Hands",
            "input_shape": [x_train.shape[1]],
            "num_classes": len(label_to_int),
            "class_mapping": label_to_int,
            "n_estimators": n_estimators,
            "accuracy": float(accuracy),
        },
    )

    return accuracy


def main():
    """Main function to demonstrate the updated trainers."""
    setup_logging()
    
    # Configure settings
    if not hasattr(settings(), "asl_data_path"):
        settings().asl_data_path = os.path.join(os.getcwd(), "data", "asl")
    
    # Train models
    train_lenet_mnist(data_path=os.path.join(os.getcwd(), "data"))
    train_resnet_asl(data_path=settings().asl_data_path)
    train_rf_asl(data_path=settings().asl_data_path)


if __name__ == "__main__":
    main()
