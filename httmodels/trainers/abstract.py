from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass
