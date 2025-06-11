"""Main module for training and evaluating models."""

import argparse
import logging

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


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def train_lenet_mnist():
    """Train LeNet model on MNIST dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Process data
    processor = MNISTProcessor()
    data = processor.load()
    processed_data = processor.preprocess(data)

    x_train = processed_data["train_data"]
    y_train = processed_data["train_labels"]
    x_test = processed_data["test_data"]
    y_test = processed_data["test_labels"]

    # Initialize trainer
    trainer = LeNetTrainer(input_shape=(1, 28, 28), num_classes=10, device=device)

    # Train and evaluate
    context = TrainingContext(trainer)
    context.fit(x_train, y_train, epochs=10, batch_size=64)
    accuracy = context.evaluate(x_test, y_test)

    # Save model
    context.save("lenet_mnist.pth")
    logging.info(f"Model saved with accuracy: {accuracy:.2f}%")

    return accuracy


def train_resnet_asl():
    """Train ResNet model on ASL hands dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Process data
    processor = ASLHandsProcessor()
    img_paths = processor.load(settings().asl_data_path)
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

    # Initialize trainer
    trainer = ResNetTrainer(num_classes=len(label_to_int), device=device)

    # Train and evaluate
    context = TrainingContext(trainer)
    context.fit(x_train, np.array(y_train), epochs=15, batch_size=32)
    accuracy = context.evaluate(x_test, np.array(y_test))

    # Save model
    context.save("resnet_asl.pth")
    logging.info(f"Model saved with accuracy: {accuracy:.2f}%")

    return accuracy


def train_rf_asl():
    """Train Random Forest model on ASL hands dataset."""
    # Process data
    processor = ASLHandsProcessor()
    img_paths = processor.load(settings().asl_data_path)
    data = processor.preprocess(img_paths)

    x_train, x_test, y_train, y_test = processor.split(data["data"], data["labels"])

    # Flatten images for Random Forest
    x_train = np.array([img.flatten() for img in x_train])
    x_test = np.array([img.flatten() for img in x_test])

    # Initialize trainer
    trainer = RandomForestTrainer(n_estimators=100, max_depth=20)

    # Train and evaluate
    context = TrainingContext(trainer)
    context.fit(x_train, y_train)
    accuracy = context.evaluate(x_test, y_test)

    # Save model
    context.save("rf_asl.pickle")
    logging.info(f"Model saved with accuracy: {accuracy:.2f}%")

    return accuracy


def train_adaboost_asl():
    """Train AdaBoost model on ASL hands dataset."""
    # Process data
    processor = ASLHandsProcessor()
    img_paths = processor.load(settings().asl_data_path)
    data = processor.preprocess(img_paths)

    x_train, x_test, y_train, y_test = processor.split(data["data"], data["labels"])

    # Flatten images for AdaBoost
    x_train = np.array([img.flatten() for img in x_train])
    x_test = np.array([img.flatten() for img in x_test])

    # Initialize trainer
    trainer = AdaBoostTrainer(n_estimators=50, learning_rate=1.0)

    # Train and evaluate
    context = TrainingContext(trainer)
    context.fit(x_train, y_train)
    accuracy = context.evaluate(x_test, y_test)

    # Save model
    context.save("adaboost_asl.pickle")
    logging.info(f"Model saved with accuracy: {accuracy:.2f}%")

    return accuracy


def main():
    """Main function."""
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

    args = parser.parse_args()
    setup_logging()

    # Configure settings
    if not hasattr(settings(), "asl_data_path"):
        settings().asl_data_path = "/home/piotr/Documents/htt/images"

    # Train models based on arguments
    if args.model == "lenet" or args.model == "all":
        if args.dataset == "mnist" or args.dataset == "all":
            train_lenet_mnist()

    if args.model == "resnet" or args.model == "all":
        if args.dataset == "asl" or args.dataset == "all":
            train_resnet_asl()

    if args.model == "rf" or args.model == "all":
        if args.dataset == "asl" or args.dataset == "all":
            train_rf_asl()

    if args.model == "adaboost" or args.model == "all":
        if args.dataset == "asl" or args.dataset == "all":
            train_adaboost_asl()


if __name__ == "__main__":
    main()


def main_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug("device: %s", device)
    processor = MNISTDataProcessor()
    train_data, test_data = processor.load(
        "data/mnist/sign_mnist_train/sign_mnist_train.csv",
        "data/mnist/sign_mnist_test/sign_mnist_test.csv",
        apply_augmentation=True,  # Apply augmentation on training data
    )
    train_loader = processor.create_dataloader(*train_data, batch_size=64)
    test_loader = processor.create_dataloader(*test_data, batch_size=64, shuffle=False)
    trainer = LeNetTrainer(input_shape=(1, 28, 28), device=device, num_classes=25)
    trainer.fit(train_loader, epochs=10)
    accuracy = trainer.evaluate(test_loader)
    logging.info(f"Final accuracy on test set: {accuracy:.2f}%")
    trainer.save("cnn_v2.pth")
    trainer.load("cnn_v2.pth")


def main_adaboost():
    processor = ASLHandsProcessor()
    img_paths = processor.load("/home/piotr/Documents/htt/images")
    data = processor.preprocess(img_paths)
    x_train, x_test, y_train, y_test = processor.split(data["data"], data["labels"])
    trainer = AdaBoostTrainer()
    context = TrainingContext(trainer)
    context.fit(x_train, y_train)
    accuracy = context.evaluate(x_test, y_test)
    logging.info(f"Accuracy after reloading model: {accuracy * 100:.2f}%")
    context.save("ada.pickle")
    context.load("ada.pickle")


def main_rf():
    processor = ASLHandsProcessor()
    img_paths = processor.load("/home/piotr/Documents/htt/images")
    data = processor.preprocess(img_paths)
    x_train, x_test, y_train, y_test = processor.split(data["data"], data["labels"])
    trainer = RandomForestTrainer()
    context = TrainingContext(trainer)
    context.fit(x_train, y_train)
    accuracy = context.evaluate(x_test, y_test)
    logging.info(f"Final accuracy on test set: {accuracy * 100:.2f}%")
    context.save("rf.pickle")
    context.load("rf.pickle")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.debug(settings().model_dump())
    main_cnn()
