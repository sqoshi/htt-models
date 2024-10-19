import numpy as np

from httmodels.trainers.abstract import Trainer


class TrainingContext:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def fit(self, x_train: np.array, y_train: np.array):
        self.trainer.fit(x_train, y_train)

    def evaluate(self, x_test: np.array, y_test: np.array):
        return self.trainer.evaluate(x_test, y_test)

    def save(self, filepath: str):
        self.trainer.save(filepath)

    def load(self, filepath: str):
        self.trainer.load(filepath)
