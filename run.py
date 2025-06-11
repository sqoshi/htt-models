#!/usr/bin/env python
"""Demo script for training models with the restructured project."""

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import torch

from httmodels.config import settings
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
    save_path: Optional[str] = None,
) -> float:
    """Train LeNet model on MNIST dataset.

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
    processor = MNISTProcessor(root=data_path)
    data = processor.load()
    processed_data = processor.preprocess(data)

    x_train = processed_data["train_data"]
    y_train = processed_data["train_labels"]
    x_test = processed_data["test_data"]
    y_test = processed_data["test_labels"]

    logging.info(f"Training data shape: {x_train.shape}")
    logging.info(f"Training labels shape: {y_train.shape}")

    # Initialize trainer
    trainer = LeNetTrainer(input_shape=(1, 28, 28), num_classes=10, device=device)

    # Train and evaluate
    context = TrainingContext(trainer)
    context.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    accuracy = context.evaluate(x_test, y_test)

    # Save model
    if save_path is None:
        save_path = "lenet_mnist.pth"

    context.save(save_path)
    logging.info(f"Model saved to {save_path} with accuracy: {accuracy:.2f}%")

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
    save_path: Optional[str] = None,
) -> float:
    """Train ResNet model on ASL hands dataset.

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
    processor = ASLHandsProcessor()
    img_paths = processor.load(data_path)
    data = processor.preprocess(img_paths)

    x_train, x_test, y_train, y_test = processor.split(data["data"], data["labels"])

    # Convert labels to integers
    label_to_int = {label: idx for idx, label in enumerate(sorted(set(y_train)))}
    y_train = [label_to_int[label] for label in y_train]
    y_test = [label_to_int[label] for label in y_test]

    # Reshape for CNN
    x_train = np.array(x_train).reshape(-1, 1, 28, 28)
    x_test = np.array(x_test).reshape(-1, 1, 28, 28)

    # Expand channels for ResNet (requires 3 channels)
    x_train = np.repeat(x_train, 3, axis=1)
    x_test = np.repeat(x_test, 3, axis=1)

    logging.info(f"Training data shape: {x_train.shape}")
    logging.info(f"Training labels shape: {np.array(y_train).shape}")
    logging.info(f"Number of classes: {len(label_to_int)}")

    # Initialize trainer
    trainer = ResNetTrainer(num_classes=len(label_to_int), device=device)

    # Train and evaluate
    context = TrainingContext(trainer)
    context.fit(x_train, np.array(y_train), epochs=epochs, batch_size=batch_size)
    accuracy = context.evaluate(x_test, np.array(y_test))

    # Save model
    if save_path is None:
        save_path = "resnet_asl.pth"

    context.save(save_path)
    logging.info(f"Model saved to {save_path} with accuracy: {accuracy:.2f}%")

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
    data_path: str, n_estimators: int = 100, save_path: Optional[str] = None
) -> float:
    """Train Random Forest model on ASL hands dataset.

    Args:
        data_path: Path to ASL hands data
        n_estimators: Number of trees in the forest
        save_path: Path to save model (if None, will use default path)

    Returns:
        Model accuracy on test set
    """
    logging.info("Training Random Forest model on ASL hands dataset")

    # Process data
    processor = ASLHandsProcessor()
    img_paths = processor.load(data_path)
    data = processor.preprocess(img_paths)

    x_train, x_test, y_train, y_test = processor.split(data["data"], data["labels"])

    # Convert labels to integers
    label_to_int = {label: idx for idx, label in enumerate(sorted(set(y_train)))}
    y_train = [label_to_int[label] for label in y_train]
    y_test = [label_to_int[label] for label in y_test]

    # Flatten images for Random Forest
    x_train = np.array([img.flatten() for img in x_train])
    x_test = np.array([img.flatten() for img in x_test])

    logging.info(f"Training data shape: {x_train.shape}")
    logging.info(f"Training labels shape: {np.array(y_train).shape}")

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
    logging.info(f"Model saved to {save_path} with accuracy: {accuracy:.2f}%")

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
    """Main function."""
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["lenet", "resnet", "rf", "all"],
        default="all",
        help="Model to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "asl", "all"],
        default="all",
        help="Dataset to use",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to dataset")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for deep learning models",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for deep learning models"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of estimators for Random Forest",
    )
    parser.add_argument(
        "--save-path", type=str, default=None, help="Path to save models"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)

    # Configure settings
    if args.data_path is not None:
        if args.dataset == "asl" or args.dataset == "all":
            settings().asl_data_path = args.data_path

    # Train models based on arguments
    if args.model == "lenet" or args.model == "all":
        if args.dataset == "mnist" or args.dataset == "all":
            data_path = args.data_path or "./data"
            train_lenet_mnist(
                data_path=data_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_path=args.save_path,
            )

    if args.model == "resnet" or args.model == "all":
        if args.dataset == "asl" or args.dataset == "all":
            data_path = args.data_path or settings().asl_data_path
            train_resnet_asl(
                data_path=data_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_path=args.save_path,
            )

    if args.model == "rf" or args.model == "all":
        if args.dataset == "asl" or args.dataset == "all":
            data_path = args.data_path or settings().asl_data_path
            train_rf_asl(
                data_path=data_path,
                n_estimators=args.n_estimators,
                save_path=args.save_path,
            )


if __name__ == "__main__":
    main()
