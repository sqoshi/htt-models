from abc import ABC, abstractmethod


class DataProcessor(ABC):
    @abstractmethod
    def load(self, source):
        pass

    @abstractmethod
    def preprocess(self, data):
        pass

    @abstractmethod
    def save(self, data, destination):
        pass

    @abstractmethod
    def split(self, data, labels, test_size=0.2):
        pass
