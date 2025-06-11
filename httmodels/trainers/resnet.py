"""ResNet model trainer."""

from httmodels.models.resnet import ResNet
from httmodels.trainers.base import DLTrainer


class ResNetTrainer(DLTrainer):
    """Trainer for ResNet model."""

    def __init__(
        self,
        num_classes=26,
        pretrained=True,
        learning_rate=0.001,
        step_size=7,
        gamma=0.1,
        device=None,
    ):
        """Initialize ResNet trainer.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            learning_rate: Learning rate for optimizer
            step_size: Step size for learning rate scheduler
            gamma: Gamma factor for learning rate scheduler
            device: Device to use for training (cpu or cuda)
        """
        model = ResNet(num_classes=num_classes, pretrained=pretrained)
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            step_size=step_size,
            gamma=gamma,
            device=device,
        )
