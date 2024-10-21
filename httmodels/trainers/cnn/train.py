import logging

import torch
import torch.nn as nn
import torch.optim as optim


class CNNTrainer:
    def __init__(self, input_shape, device, num_classes=25):
        self.device = device
        self.model = self.build_model(input_shape, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-4
        )

    def build_model(self, input_shape, num_classes):
        return nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * (input_shape[1] // 4) * (input_shape[2] // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def fit(self, train_loader, epochs=10):
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            logging.debug(
                f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}"
            )

    def evaluate(self, test_loader):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logging.debug(f"Accuracy on test set: {accuracy:.2f}%")
        return accuracy

    def save(self, filepath: str):
        torch.save(self.model, filepath)
        logging.debug(f"Model saved to {filepath}")

    def load(self, filepath: str):
        self.model = torch.load(filepath)
        self.model.to(self.device)
        logging.debug(f"Model loaded from {filepath}")
