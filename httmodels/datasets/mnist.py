import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MnistASLDataset(Dataset):
    def __init__(self, csv_path="../../data/mnist/sign_mnist_train.csv"):
        self.csv_path = csv_path
        with open(csv_path, "r") as f:
            self.data_length = sum(1 for _ in f) - 1

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        chunk = pd.read_csv(
            self.csv_path,
            skiprows=idx + 1,
            nrows=1,
            header=None,
        )
        label = int(chunk.iloc[0, 0])
        image = np.array(chunk.iloc[0, 1:]).reshape(28, 28).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), label
