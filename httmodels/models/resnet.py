"""ResNet model implementation."""

import torch
import torch.nn as nn
from torchvision import models

from httmodels.models.base import PyTorchModel


class ResNet(PyTorchModel):
    """ResNet model for image classification.

    Uses a pretrained ResNet18 model from torchvision with a modified
    final fully connected layer for the specific number of classes.
    """

    def __init__(self, num_classes=26, pretrained=True):
        """Initialize ResNet model.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(ResNet, self).__init__()

        # Load pretrained ResNet18 model
        self.model = models.resnet18(pretrained=pretrained)

        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
