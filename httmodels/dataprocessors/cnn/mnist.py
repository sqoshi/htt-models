import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from httmodels.config import settings


class MNISTDataProcessor:
    def load(self, train_source, test_source):
        # Load and preprocess training data
        train_data = pd.read_csv(train_source)
        train_labels = train_data.iloc[:, 0].values
        train_images = train_data.iloc[:, 1:].values / 255.0
        train_images = train_images.reshape(-1, 1, 28, 28)
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
