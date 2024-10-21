import logging

import numpy as np
import torch

from httmodels.config import settings
from httmodels.dataprocessors.cnn.aslhands import ASLHandsBoxesSelectedProcessor
from httmodels.dataprocessors.cnn.mnist import MNISTDataProcessor
from httmodels.dataprocessors.randomforest.aslhands import ASLHandsProcessor
from httmodels.trainers.adabooster.train import AdaBoostTrainer
from httmodels.trainers.cnn.train import CNNTrainer
from httmodels.trainers.context import TrainingContext
from httmodels.trainers.randomforest.train import RandomForestTrainer


def main_cnnasl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug("device: %s", device)
    processor = ASLHandsBoxesSelectedProcessor()

    img_paths = processor.load("/home/piotr/Documents/htt/images")
    data = processor.preprocess(img_paths)

    x_train, x_test, y_train, y_test = processor.split(data["data"], data["labels"])

    label_to_int = {label: idx for idx, label in enumerate(sorted(set(y_train)))}
    y_train = [label_to_int[label] for label in y_train]
    y_test = [label_to_int[label] for label in y_test]

    x_train = np.array(x_train).reshape(-1, 1, 28, 28)
    x_test = np.array(x_test).reshape(-1, 1, 28, 28)

    trainer = CNNTrainer(
        input_shape=(1, 28, 28), device=device, num_classes=len(label_to_int)
    )

    context = TrainingContext(trainer)

    context.fit(x_train, np.array(y_train))

    accuracy = context.evaluate(x_test, np.array(y_test))
    logging.info(f"Final accuracy on test set: {accuracy:.2f}%")

    context.save("cnnasl.pth")
    context.load("cnnasl.pth")


def main_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug("device: %s", device)
    processor = MNISTDataProcessor()
    train_data, test_data = processor.load(
        "data/mnist/sign_mnist_train/sign_mnist_train.csv",
        "data/mnist/sign_mnist_test/sign_mnist_test.csv",
        apply_augmentation=True,  # Apply augmentation on training data
    )
    train_loader = processor.create_dataloader(*train_data, batch_size=64)
    test_loader = processor.create_dataloader(*test_data, batch_size=64, shuffle=False)
    trainer = CNNTrainer(input_shape=(1, 28, 28), device=device, num_classes=25)
    trainer.fit(train_loader, epochs=10)
    accuracy = trainer.evaluate(test_loader)
    logging.info(f"Final accuracy on test set: {accuracy:.2f}%")
    trainer.save("cnn_v2.pth")
    trainer.load("cnn_v2.pth")


def main_adaboost():
    processor = ASLHandsProcessor()
    img_paths = processor.load("/home/piotr/Documents/htt/images")
    data = processor.preprocess(img_paths)
    x_train, x_test, y_train, y_test = processor.split(data["data"], data["labels"])
    trainer = AdaBoostTrainer()
    context = TrainingContext(trainer)
    context.fit(x_train, y_train)
    accuracy = context.evaluate(x_test, y_test)
    logging.info(f"Accuracy after reloading model: {accuracy * 100:.2f}%")
    context.save("ada.pickle")
    context.load("ada.pickle")


def main_rf():
    processor = ASLHandsProcessor()
    img_paths = processor.load("/home/piotr/Documents/htt/images")
    data = processor.preprocess(img_paths)
    x_train, x_test, y_train, y_test = processor.split(data["data"], data["labels"])
    trainer = RandomForestTrainer()
    context = TrainingContext(trainer)
    context.fit(x_train, y_train)
    accuracy = context.evaluate(x_test, y_test)
    logging.info(f"Final accuracy on test set: {accuracy * 100:.2f}%")
    context.save("rf.pickle")
    context.load("rf.pickle")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.debug(settings().model_dump())
    main_cnn()
