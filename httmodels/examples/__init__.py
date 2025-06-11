"""Example implementations for the httmodels package."""

from httmodels.examples.dataloaders_example import (
    train_lenet_mnist_with_dataloader,
    train_resnet_asl_with_dataloader,
    train_rf_asl_with_dataloader
)

from httmodels.examples.updated_trainers_example import (
    train_lenet_mnist,
    train_resnet_asl,
    train_rf_asl
)

__all__ = [
    # Dataloader examples
    "train_lenet_mnist_with_dataloader",
    "train_resnet_asl_with_dataloader",
    "train_rf_asl_with_dataloader",
    
    # Updated trainer examples
    "train_lenet_mnist",
    "train_resnet_asl",
    "train_rf_asl"
]
