import numpy as np
import matplotlib.pyplot as plt


class Preprocessing:
    @staticmethod
    def deseasonalize(data: np.ndarray):
        """
        Remove seasonal components from the data.
        """
        # Implementation

    @staticmethod
    def detrend(data: np.ndarray):
        """
        Remove trends from the data.
        """
        # Implementation





class PlottingUtilities:
    """
    Class for plotting utilities.
    """

    @staticmethod
    def basic_time_series_plot(data: np.ndarray, title: str = "Time Series"):
        """
        Basic time series plotting utility.
        """
        plt.figure()
        plt.plot(data)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()
        