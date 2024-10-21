import logging
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from httmodels.trainers.abstract import Trainer


class RandomForestTrainer(Trainer):
    def __init__(self):
        self.model = RandomForestClassifier(verbose=1, warm_start=True)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        logging.debug(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
