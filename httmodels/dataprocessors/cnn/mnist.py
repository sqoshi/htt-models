import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_pil_image

from httmodels.config import settings


# Define augmentation transforms
def get_train_transforms():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.RandomRotation(10),  # Randomly rotate by 10 degrees
            transforms.RandomResizedCrop(
                (28, 28), scale=(0.8, 1.0)
            ),  # Random crop and resize
            transforms.ToTensor(),  # Convert image to PyTorch Tensor
        ]
    )


def get_test_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),  # Only convert test data to tensor (no augmentation)
        ]
    )


class MNISTDataProcessor:
    def load(self, train_source, test_source, apply_augmentation=False):
        # Load and preprocess training data
        train_data = pd.read_csv(train_source)
        train_labels = train_data.iloc[:, 0].values
        train_images = train_data.iloc[:, 1:].values / 255.0
        train_images = train_images.reshape(-1, 1, 28, 28)

        if apply_augmentation:
            train_transforms = get_train_transforms()
            augmented_images = []
            for img in train_images:
                pil_img = to_pil_image(torch.tensor(img, dtype=torch.float32))
                augmented_img = train_transforms(pil_img)
                augmented_images.append(augmented_img)
            train_images = torch.stack(augmented_images)
        else:
            train_images = torch.tensor(train_images, dtype=torch.float32)

        train_labels = torch.tensor(train_labels, dtype=torch.long)

        # Load and preprocess testing data
        test_data = pd.read_csv(test_source)
        test_labels = test_data.iloc[:, 0].values
        test_images = test_data.iloc[:, 1:].values / 255.0
        test_images = test_images.reshape(-1, 1, 28, 28)
        test_images = torch.tensor(test_images, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        return (train_images, train_labels), (test_images, test_labels)

    def create_dataloader(
        self, x_data, y_data, batch_size=64, shuffle=True
    ) -> DataLoader:
        dataset = TensorDataset(x_data, y_data)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=settings().workers,
        )
