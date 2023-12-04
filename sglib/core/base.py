from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class Generator(ABC):
    """
    Abstract base class for all time series models.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a TimeSeriesModel instance with optional parameters.
        """
        pass
    
    @abstractmethod
    def preprocessing(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the input time series data.
        """
        self.validate_input(data)
        return data

    @abstractmethod
    def fit(self, data: np.ndarray):
        """
        Fit the model to the given time series data.
        """
        self.validate_input(data)

    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """
        Generate a synthetic time series of length n.
        """
        pass

    @abstractmethod
    def plot(self, data: np.ndarray):
        """
        Plot the generated synthetic time series.
        """
        pass

    def validate_input(self, data: np.ndarray):
        """
        Validate the input time series data.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a NumPy array.")
        if data.ndim != 1:
            raise ValueError("Data must be a 1D array.")

    def export(self):
        """
        Export synthetic timeseries.
        """
        return
        


