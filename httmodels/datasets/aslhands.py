import os

import cv2
from torch.utils.data import Dataset


class ASLHandsDataset(Dataset):
    def __init__(self, dataset_path="../../data/asl_hands/"):
        self.dataset_path = dataset_path
        self.image_paths = []
        self.labels = []

        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append(os.path.basename(root))
        self.data_length = len(self.image_paths)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        return image, label
