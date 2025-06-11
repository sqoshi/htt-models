"""LeNet model trainer."""

from httmodels.models.lenet import LeNet
from httmodels.trainers.base import DLTrainer


class LeNetTrainer(DLTrainer):
    """Trainer for LeNet CNN model."""

    def __init__(
        self,
        input_shape=(1, 28, 28),
        num_classes=10,
        learning_rate=0.001,
        step_size=7,
        gamma=0.1,
        device=None,
    ):
        """Initialize LeNet trainer.

        Args:
            input_shape: Input shape of images (channels, height, width)
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            step_size: Step size for learning rate scheduler
            gamma: Gamma factor for learning rate scheduler
            device: Device to use for training (cpu or cuda)
        """
        model = LeNet(input_shape=input_shape, num_classes=num_classes)
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            step_size=step_size,
            gamma=gamma,
            device=device,
        )
