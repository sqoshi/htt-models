import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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

    def fit(self, x_train, y_train, epochs=10, batch_size=64):
        self.model.train()

        x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)

        dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

    def evaluate(self, x_test, y_test):
        self.model.eval()

        x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)

        dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

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
        """Save the model to a file."""
        torch.save(self.model.state_dict(), filepath)
        logging.debug(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load a model from a file."""
        self.model.load_state_dict(torch.load(filepath), weights_only=True)
        self.model.to(self.device)
        logging.debug(f"Model loaded from {filepath}")
