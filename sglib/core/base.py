from abc import ABC, abstractmethod
import numpy as np

class Generator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def preprocessing(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, **kwargs):
        pass


