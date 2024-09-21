import os

import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = "./data_mnist"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

csv_file_path = "/home/piotr/Downloads/archive/sign_mnist_train.csv"
df = pd.read_csv(csv_file_path)

labels = df["label"].values
images = df.drop("label", axis=1).values

images = images.reshape(-1, 28, 28).astype(np.uint8)

unique_labels = np.unique(labels)
for label in unique_labels:
    label_dir = os.path.join(DATA_DIR, str(label))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

for idx, (image, label) in enumerate(zip(images, labels)):
    img = Image.fromarray(image)
    img.save(os.path.join(DATA_DIR, str(label), f"{idx}.png"))

print("MNIST Sign Language dataset images saved successfully.")
