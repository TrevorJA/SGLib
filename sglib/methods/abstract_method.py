from abc import ABC, abstractmethod

class AbstractGenerator(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def preprocessing(self, **kwargs):
        """
        Preprocesses the input time series data. 
        """
        pass
    
    @abstractmethod
    def fit(self, **kwargs):
        """
        Fits the model to the data.
        """
        pass
    
    @abstractmethod
    def generate(self, **kwargs):
        """
        Generates synthetic data from the fitted model.
        """
        pass
    
    def plot(self,**kwargs):
        """
        Plots the distribution of the fitted model.
        """
        pass