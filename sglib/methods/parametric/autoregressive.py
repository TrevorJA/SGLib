"""
The AR class is a parent class for all autoregressive models.
"""
# Importing necessary libraries for testing the AR model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sglib.core.base import Generator

class AR(Generator):
    """
    Autoregressive model for generating synthetic timeseries data.

    Attributes:
        name (str): Name of the model.
        lag (int): The number of lagged observations in the model.
        model (statsmodels.tsa.ar_model.AutoReg): The fitted autoregressive model.
    """

    def __init__(self, name, lag=1):
        """
        Initializes the AR model with the specified lag.

        Args:
            name (str): Name of the model.
            lag (int): The number of lagged observations in the model.
        """
        super().__init__(name)
        self.lag = lag
        self.model = None

    def preprocessing(self, data):
        """
        Preprocesses the input time series data. Currently a placeholder, can be expanded for specific preprocessing needs.

        Args:
            data (np.ndarray): Input time series data as a NumPy array.

        Returns:
            np.ndarray: Preprocessed time series data.
        """
        return data

    def fit(self, data):
        """
        Fits the AR model to the given time series data.

        Args:
            data (np.ndarray): Input time series data as a NumPy array.
        """
        self.validate_input(data)
        self.model = AutoReg(data, lags=self.lag).fit()

    def generate(self, n):
        """
        Generates a synthetic time series of length n using the fitted model.

        Args:
            n (int): The length of the time series to be generated.

        Returns:
            np.ndarray: Generated time series data.
        """
        if self.model is None:
            raise Exception("Model not fitted. Call fit() before generate().")
        predictions = self.model.predict(start=len(self.model.data.endog), end=len(self.model.data.endog) + n - 1)
        return predictions

    def plot(self, data):
        """
        Plots the generated synthetic time series data.

        Args:
            data (np.ndarray): Time series data to be plotted.
        """
        pass




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.api import VAR

class VARModel(Generator):
    """
    Vector Autoregressive model for generating synthetic multi-correlated timeseries data.

    Attributes:
        name (str): Name of the model.
        lag_order (int): The number of lagged observations in the model.
        model (statsmodels.tsa.api.VAR): The fitted vector autoregressive model.
    """

    def __init__(self, name, lag_order=1):
        """
        Initializes the VAR model with the specified lag order.

        Args:
            name (str): Name of the model.
            lag_order (int): The number of lagged observations in the model.
        """
        super().__init__(name)
        self.lag_order = lag_order
        self.model = None

    def preprocessing(self, data):
        """
        Preprocesses the input time series data. Placeholder for specific preprocessing needs.

        Args:
            data (np.ndarray): Input time series data as a 2D NumPy array, each column representing a time series.

        Returns:
            np.ndarray: Preprocessed time series data.
        """
        return data

    def fit(self, data):
        """
        Fits the VAR model to the given multi-correlated time series data.

        Args:
            data (np.ndarray): Input time series data as a 2D NumPy array.
        """
        self.validate_input(data)
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array for VAR model.")
        self.model = VAR(data).fit(self.lag_order)

    def generate(self, n):
        """
        Generates synthetic time series data of length n using the fitted model.

        Args:
            n (int): The length of the time series to be generated.

        Returns:
            np.ndarray: Generated time series data.
        """
        if self.model is None:
            raise Exception("Model not fitted. Call fit() before generate().")
        forecast = self.model.forecast(y=self.model.endog, steps=n)
        return forecast

    def plot(self, data):
        """
        Plots the generated synthetic time series data.

        Args:
            data (np.ndarray): Time series data to be plotted, each column representing a time series.
        """
        plt.figure(figsize=(10, 6))
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=f'Series {i+1}')
        plt.title(f"Generated Time Series Data - {self.name}")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def validate_input(self, data):
        """
        Validate the input time series data.

        Args:
            data (np.ndarray): Input time series data to validate.
        """
        super().validate_input(data)
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array for VAR model.")

