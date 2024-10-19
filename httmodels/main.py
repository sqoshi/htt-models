# import pickle

# import numpy as np
# from sklearn.model_selection import train_test_split

# from httmodels.trainers.context import TrainingContext
# from httmodels.trainers.randomforest.train import RandomForestTrainer


# def main():
#     with open("/home/piotr/Documents/htt/pickles/data.pickle", "rb") as f:
#         data_dict = pickle.load(f)
#         data = np.asarray(data_dict["data"])
#         labels = np.asarray(data_dict["labels"])

#         x_train, x_test, y_train, y_test = train_test_split(
#             data, labels, test_size=0.2, shuffle=True, stratify=labels
#         )
#         rf_trainer = RandomForestTrainer()
#         context = TrainingContext(rf_trainer)

#         context.fit(x_train, y_train)
#         context.evaluate(x_test, y_test)
#         context.save_model("random_forest_model.pickle")


# main()


import logging

import torch

from httmodels.config import settings
from httmodels.dataprocessors.cnn.mnist import MNISTDataProcessor
from httmodels.trainers.cnn.train import CNNTrainer
from httmodels.trainers.context import TrainingContext


def main_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug("device: %s", device)
    processor = MNISTDataProcessor()

    train_data, test_data = processor.load(
        "data/mnist/sign_mnist_train/sign_mnist_train.csv",
        "data/mnist/sign_mnist_test/sign_mnist_test.csv",
    )

    train_loader = processor.create_dataloader(*train_data, batch_size=64)
    test_loader = processor.create_dataloader(*test_data, batch_size=64, shuffle=False)

    trainer = CNNTrainer(input_shape=(1, 28, 28), device=device, num_classes=25)

    context = TrainingContext(trainer)

    for images, labels in train_loader:
        context.fit(images.numpy(), labels.numpy())

    accuracy = 0
    for images, labels in test_loader:
        accuracy = context.evaluate(images.numpy(), labels.numpy())

    logging.info(f"Final accuracy on test set: {accuracy:.2f}%")

    context.save("cnn.pth")
    context.load("cnn.pth")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.debug(settings().model_dump())
    main_cnn()
