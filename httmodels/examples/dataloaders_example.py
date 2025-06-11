"""Example implementation using the streamlined dataloaders.

This example shows how to train models using the refactored dataloader
functionality in combination with the existing trainers and processors.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from httmodels.config import settings
from httmodels.dataloaders import (
    create_train_val_dataloaders,
    get_asl_transforms,
    get_dataloader,
    get_mnist_transforms
)
from httmodels.preprocessing.aslhands import ASLHandsProcessor
from httmodels.preprocessing.mnist import MNISTProcessor
from httmodels.trainers.adaboost import AdaBoostTrainer
from httmodels.trainers.context import TrainingContext
from httmodels.trainers.lenet import LeNetTrainer
from httmodels.trainers.resnet import ResNetTrainer
from httmodels.trainers.rf import RandomForestTrainer
from httmodels.utils import get_device, save_model_info


def train_lenet_mnist_with_dataloader(
    data_path: str,
    epochs: int = 10,
    batch_size: int = 64,
    save_path: Optional[str] = None
) -> float:
    """Train LeNet model on MNIST dataset using the dataloader API.

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
    
    # Create train and test loaders
    train_transform = get_mnist_transforms(augmentation=True)
    test_transform = get_mnist_transforms(augmentation=False)
    
    # Apply transforms to datasets
    train_dataset = data["train"]
    train_dataset.transform = train_transform
    
    test_dataset = data["test"]
    test_dataset.transform = test_transform
    
    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = get_dataloader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Initialize trainer and training context
    trainer = LeNetTrainer(
        input_shape=(1, 28, 28), 
        num_classes=10, 
        device=device
    )
    context = TrainingContext(trainer)
    
    # Train with dataloader
    logging.info("Starting model training...")
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
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        logging.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}"
        )
    
    # Evaluate model
    model.eval()
    running_corrects = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    
    accuracy = running_corrects.double() / len(test_loader.dataset)
    accuracy = accuracy.item() * 100
    
    logging.info(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save model
    if save_path is None:
        save_path = "lenet_mnist.pth"
    
    full_path = os.path.join(settings().models_path, save_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    torch.save(model.state_dict(), full_path)
    
    logging.info(f"Model saved to {full_path} with accuracy: {accuracy:.2f}%")
    
    # Save model info
    save_model_info(
        full_path,
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


def train_resnet_asl_with_dataloader(
    data_path: str,
    epochs: int = 15,
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> float:
    """Train ResNet model on ASL hands dataset using the dataloader API.

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
    
    # Process data with the processor
    processor = ASLHandsProcessor(apply_augmentation=False)  # No augmentation here, we'll use transforms
    img_paths = processor.load(data_path)
    data = processor.preprocess(img_paths)
    
    # Get label mapping
    label_to_int = data["label_mapping"]
    int_to_label = {v: k for k, v in label_to_int.items()}
    
    # Split data
    x_train, x_test, y_train, y_test = processor.split(
        data["data"], data["labels"]
    )
    
    # Reshape for CNN and convert to RGB (3 channels for ResNet)
    x_train = np.array(x_train).reshape(-1, 28, 28, 1)
    x_test = np.array(x_test).reshape(-1, 28, 28, 1)
    
    # Convert labels to integers
    y_train = np.array([label_to_int[label] for label in y_train])
    y_test = np.array([label_to_int[label] for label in y_test])
    
    # Create datasets with transformations
    train_transform = get_asl_transforms(model_type="resnet", augmentation=True)
    test_transform = get_asl_transforms(model_type="resnet", augmentation=False)
    
    # Create custom dataset class for applying transforms
    class TransformDataset(TensorDataset):
        def __init__(self, x, y, transform=None):
            self.x = torch.tensor(x, dtype=torch.float32) / 255.0
            self.y = torch.tensor(y, dtype=torch.long)
            self.transform = transform
            
        def __len__(self):
            return len(self.y)
            
        def __getitem__(self, idx):
            x = self.x[idx]
            y = self.y[idx]
            
            if self.transform:
                # Convert to PIL for torchvision transforms
                img = (x.numpy() * 255).astype(np.uint8)
                img = np.repeat(img, 3, axis=2)  # Expand to 3 channels
                img = torch.tensor(img).permute(2, 0, 1)  # HWC to CHW
                img = train_transform(img)
                return img, y
            
            # Repeat grayscale to 3 channels
            x = torch.repeat_interleave(x.unsqueeze(0), 3, dim=0)
            return x, y
    
    # Create datasets with transforms
    train_dataset = TransformDataset(x_train, y_train, transform=train_transform)
    test_dataset = TransformDataset(x_test, y_test, transform=test_transform)
    
    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = get_dataloader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Initialize trainer with model
    trainer = ResNetTrainer(
        num_classes=len(label_to_int), 
        device=device
    )
    
    # Train model
    model = trainer.model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        logging.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}"
        )
    
    # Evaluate model
    model.eval()
    running_corrects = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    
    accuracy = running_corrects.double() / len(test_loader.dataset)
    accuracy = accuracy.item() * 100
    
    logging.info(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save model
    if save_path is None:
        save_path = "resnet_asl.pth"
    
    full_path = os.path.join(settings().models_path, save_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    torch.save(model.state_dict(), full_path)
    
    logging.info(f"Model saved to {full_path} with accuracy: {accuracy:.2f}%")
    
    # Save model info
    save_model_info(
        full_path,
        {
            "model_type": "ResNet",
            "dataset": "ASL Hands",
            "input_shape": [3, 224, 224],
            "num_classes": len(label_to_int),
            "class_mapping": label_to_int,
            "accuracy": float(accuracy),
            "epochs": epochs,
            "batch_size": batch_size,
        },
    )
    
    return accuracy


def train_rf_asl_with_dataloader(
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
    logging.info(f"Training Random Forest with {n_estimators} estimators")
    
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
