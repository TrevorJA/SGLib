from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
from sglib.core.base import Generator, PlottingUtilities
import numpy as np

class ARIMA(Generator):
    """
    ARIMA class implementing the TimeSeriesModel interface.
    """

    def __init__(self, p: int, d: int, q: int):
        """
        Initialize ARIMA instance with model orders.
        """
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.model = None

    def fit(self, data: np.ndarray):
        """
        Fit the ARIMA model to the given time series data.
        """
        super().fit(data)
        self.model = ARIMA_model(data, order=(self.p, self.d, self.q))
        self.model = self.model.fit()

    def generate(self, n: int) -> np.ndarray:
        """
        Generate synthetic time series using the fitted ARIMA model.
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet.")
        
        # Here you would implement ARIMA-based synthetic time series generation.
        # For demonstration, let's assume it returns a numpy array.
        synthetic_data = np.random.rand(n)
        
        return synthetic_data

    def plot(self, data: np.ndarray):
        """
        Plot the generated synthetic time series.
        """
        PlottingUtilities.basic_time_series_plot(data, title="ARIMA Generated Time Series")
