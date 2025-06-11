"""LeNet-5 CNN model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from httmodels.models.base import PyTorchModel


class LeNet(PyTorchModel):
    """LeNet-5 Convolutional Neural Network for image classification.

    Architecture:
    - 2 convolutional layers
    - 3 fully connected layers
    - ReLU activations
    - Max pooling after conv layers
    """

    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """Initialize LeNet model.

        Args:
            input_shape: Tuple of (channels, height, width)
            num_classes: Number of output classes
        """
        super(LeNet, self).__init__()

        channels, height, width = input_shape

        # Feature extraction layers
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Calculate size after convolutions and pooling
        conv1_output_size = (height - 5 + 2 * 2) + 1  # After conv1
        pool1_output_size = conv1_output_size // 2  # After pool1
        conv2_output_size = (pool1_output_size - 5) + 1  # After conv2
        pool2_output_size = conv2_output_size // 2  # After pool2

        # Flatten size for FC layers
        self.flatten_size = 16 * pool2_output_size * pool2_output_size

        # Classification layers
        self.fc1 = nn.Linear(self.flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers with max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, self.flatten_size)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
