"""Main module for training and evaluating models."""

import argparse
import logging
import os

import numpy as np
import torch

from httmodels.config import settings
from httmodels.dataloaders import (
    get_asl_transforms,
    get_dataloader,
    get_mnist_transforms,
)
from httmodels.datasets import ASLDataset
from httmodels.preprocessing.mnist import MNISTProcessor
from httmodels.trainers.adaboost import AdaBoostTrainer
from httmodels.trainers.context import TrainingContext
from httmodels.trainers.lenet import LeNetTrainer
from httmodels.trainers.resnet import ResNetTrainer
from httmodels.trainers.rf import RandomForestTrainer
from httmodels.utils import get_device


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def get_model_path(model_name, dataset_name):
    """Get the path to save the model.

    Args:
        model_name: Name of the model (lenet, resnet, rf, adaboost)
        dataset_name: Name of the dataset (mnist, asl)

    Returns:
        Path to save the model
    """
    # Create models directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Generate filename based on model and dataset
    if model_name in ["rf", "adaboost"]:
        extension = ".pickle"
    else:
        extension = ".pth"

    filename = f"{model_name}_{dataset_name}{extension}"
    return os.path.join(models_dir, filename)


def train_lenet_mnist(data_path=None, epochs=10, batch_size=64):
    """Train LeNet model on MNIST dataset using the dataloader API.

    Args:
        data_path: Path to MNIST data directory
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Model accuracy on test set
    """
    device = get_device()
    logging.info(f"Using device: {device}")

    # Use default data path if not provided
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_dir, "data")

    # Process data
    processor = MNISTProcessor(root=data_path, apply_augmentation=True)
    data = processor.load()

    # Get transforms
    train_transform = get_mnist_transforms(augmentation=True)
    test_transform = get_mnist_transforms(augmentation=False)

    # Apply transforms to datasets
    train_dataset = data["train"]
    train_dataset.transform = train_transform

    test_dataset = data["test"]
    test_dataset.transform = test_transform

    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize trainer and training context
    trainer = LeNetTrainer(input_shape=(1, 28, 28), num_classes=10, device=device)
    context = TrainingContext(trainer)  # Train model
    model = trainer.model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        logging.info(
            f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} "
            f"Acc: {epoch_acc:.4f}"
        )

    # Evaluate model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f"Test Accuracy: {accuracy:.2f}%")

    # Save model
    model_path = get_model_path("lenet", "mnist")
    context.save(model_path)
    logging.info(f"Model saved to {model_path}")

    return accuracy


def train_resnet_asl(data_path=None, epochs=15, batch_size=32):
    """Train ResNet model on ASL hands dataset using the dataloader API.

    Args:
        data_path: Path to ASL data directory
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Model accuracy on test set
    """
    device = get_device()
    logging.info(f"Using device: {device}")

    # Use default data path if not provided
    if data_path is None:
        data_path = settings().asl_data_path

    # Create dataset
    dataset = ASLDataset(data_path=data_path)

    # Get transforms
    transform = get_asl_transforms(model_type="resnet", augmentation=True)

    # Create train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Apply transforms
    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = transform

    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize trainer
    num_classes = len(dataset.unique_labels)
    trainer = ResNetTrainer(num_classes=num_classes, device=device)
    context = TrainingContext(trainer)

    # Train model
    model = trainer.model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        logging.info(
            f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
        )

    # Evaluate model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f"Validation Accuracy: {accuracy:.2f}%")

    # Save model
    model_path = get_model_path("resnet", "asl")
    context.save(model_path)
    logging.info(f"Model saved to {model_path}")

    return accuracy


def train_rf_asl(data_path=None, n_estimators=100, max_depth=20):
    """Train Random Forest model on ASL hands dataset using the dataloader API.

    Args:
        data_path: Path to ASL data directory
        n_estimators: Number of trees in the random forest
        max_depth: Maximum depth of the trees

    Returns:
        Model accuracy on test set
    """
    logging.info("Training Random Forest model on ASL dataset")

    # Use default data path if not provided
    if data_path is None:
        data_path = settings().asl_data_path

    # Create dataset
    dataset = ASLDataset(data_path=data_path)

    # Create train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=128,  # Larger batch size for faster processing
        shuffle=True,
    )

    val_loader = get_dataloader(val_dataset, batch_size=128, shuffle=False)

    # Collect data from dataloaders
    x_train, y_train = [], []
    for images, labels in train_loader:
        for img, label in zip(images, labels):
            # Convert tensor to numpy and flatten
            img_np = img.numpy().flatten()
            x_train.append(img_np)
            y_train.append(label.item())

    x_val, y_val = [], []
    for images, labels in val_loader:
        for img, label in zip(images, labels):
            img_np = img.numpy().flatten()
            x_val.append(img_np)
            y_val.append(label.item())

    # Convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    # Initialize trainer
    trainer = RandomForestTrainer(n_estimators=n_estimators, max_depth=max_depth)
    context = TrainingContext(trainer)

    # Train model
    context.fit(x_train, y_train)

    # Evaluate model
    accuracy = context.evaluate(x_val, y_val)
    logging.info(f"Validation Accuracy: {accuracy:.2f}%")

    # Save model
    model_path = get_model_path("rf", "asl")
    context.save(model_path)
    logging.info(f"Model saved to {model_path}")

    return accuracy


def train_adaboost_asl(data_path=None, n_estimators=50, learning_rate=1.0):
    """Train AdaBoost model on ASL hands dataset using the dataloader API.

    Args:
        data_path: Path to ASL data directory
        n_estimators: Number of estimators
        learning_rate: Learning rate

    Returns:
        Model accuracy on test set
    """
    logging.info("Training AdaBoost model on ASL dataset")

    # Use default data path if not provided
    if data_path is None:
        data_path = settings().asl_data_path

    # Create dataset
    dataset = ASLDataset(data_path=data_path)

    # Create train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=128,  # Larger batch size for faster processing
        shuffle=True,
    )

    val_loader = get_dataloader(val_dataset, batch_size=128, shuffle=False)

    # Collect data from dataloaders
    x_train, y_train = [], []
    for images, labels in train_loader:
        for img, label in zip(images, labels):
            # Convert tensor to numpy and flatten
            img_np = img.numpy().flatten()
            x_train.append(img_np)
            y_train.append(label.item())

    x_val, y_val = [], []
    for images, labels in val_loader:
        for img, label in zip(images, labels):
            img_np = img.numpy().flatten()
            x_val.append(img_np)
            y_val.append(label.item())

    # Convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    # Initialize trainer
    trainer = AdaBoostTrainer(n_estimators=n_estimators, learning_rate=learning_rate)
    context = TrainingContext(trainer)

    # Train model
    context.fit(x_train, y_train)

    # Evaluate model
    accuracy = context.evaluate(x_val, y_val)
    logging.info(f"Validation Accuracy: {accuracy:.2f}%")

    # Save model
    model_path = get_model_path("adaboost", "asl")
    context.save(model_path)
    logging.info(f"Model saved to {model_path}")

    return accuracy


def main():
    """Main function for training models."""
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["lenet", "resnet", "rf", "adaboost", "all"],
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
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs for training (neural networks only)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training (neural networks only)",
    )

    args = parser.parse_args()
    setup_logging()

    # Configure settings
    if args.data_path and "asl" in [args.dataset, "all"]:
        settings().asl_data_path = args.data_path
    elif not hasattr(settings(), "asl_data_path"):
        settings().asl_data_path = "/home/piotr/Documents/htt/images"

    # Extract optional parameters
    nn_kwargs = {}
    if args.epochs:
        nn_kwargs["epochs"] = args.epochs
    if args.batch_size:
        nn_kwargs["batch_size"] = args.batch_size

    # Train models based on arguments
    if args.model in ["lenet", "all"] and args.dataset in ["mnist", "all"]:
        train_lenet_mnist(data_path=args.data_path, **nn_kwargs)

    if args.model in ["resnet", "all"] and args.dataset in ["asl", "all"]:
        train_resnet_asl(data_path=args.data_path, **nn_kwargs)

    if args.model in ["rf", "all"] and args.dataset in ["asl", "all"]:
        train_rf_asl(data_path=args.data_path)

    if args.model in ["adaboost", "all"] and args.dataset in ["asl", "all"]:
        train_adaboost_asl(data_path=args.data_path)


if __name__ == "__main__":
    main()
