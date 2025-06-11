import logging

from sklearn.metrics import accuracy_score

from httmodels.config import settings
from httmodels.dataloaders.loaders import MLDataLoader
from httmodels.datasets.mnist import MnistASLDataset
from httmodels.trainers.rf import RandomForestTrainer


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.debug(settings().model_dump())
    dataset = MnistASLDataset(csv_path="../data/mnist/sign_mnist_train.csv")

    train_loader = MLDataLoader(
        dataset, batch_size=1, shuffle=True, sample_ratio=0.3, augmentation=True
    )
    test_loader = MLDataLoader(
        dataset, batch_size=1, shuffle=False, sample_ratio=0.1, augmentation=False
    )

    rf_trainer = RandomForestTrainer()

    logging.info("[INFO] Training RandomForest model...")
    print("[INFO] Training RandomForest model...")
    rf_trainer.fit(train_loader, 5)

    logging.info("[INFO] Evaluating model...")
    print("[INFO] Evaluating model...")
    y_pred = rf_trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
