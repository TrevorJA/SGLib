"""
Pytest configuration and fixtures for synhydro tests.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime


@pytest.fixture
def sample_daily_series():
    """Generate a sample daily time series for testing."""
    dates = pd.date_range(start="2010-01-01", end="2015-12-31", freq="D")
    np.random.seed(42)
    values = np.random.gamma(shape=2.0, scale=50.0, size=len(dates))
    return pd.Series(values, index=dates, name="site_1")


@pytest.fixture
def sample_monthly_series():
    """Generate a sample monthly time series for testing."""
    dates = pd.date_range(start="2010-01-01", end="2020-12-31", freq="MS")
    np.random.seed(42)
    values = np.random.gamma(shape=2.0, scale=100.0, size=len(dates))
    return pd.Series(values, index=dates, name="site_1")


@pytest.fixture
def sample_daily_dataframe():
    """Generate a sample daily multi-site DataFrame for testing."""
    dates = pd.date_range(start="2010-01-01", end="2015-12-31", freq="D")
    np.random.seed(42)
    n_sites = 3
    data = {}
    for i in range(n_sites):
        # Generate correlated data
        base = np.random.gamma(shape=2.0, scale=50.0, size=len(dates))
        noise = np.random.normal(0, 10, size=len(dates))
        data[f"site_{i+1}"] = np.clip(base + noise, 0.01, None)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_monthly_dataframe():
    """Generate a sample monthly multi-site DataFrame for testing."""
    dates = pd.date_range(start="2010-01-01", end="2020-12-31", freq="MS")
    np.random.seed(42)
    n_sites = 3
    data = {}
    for i in range(n_sites):
        # Generate correlated data with seasonal pattern
        base = np.random.gamma(shape=2.0, scale=100.0, size=len(dates))
        seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 20, size=len(dates))
        data[f"site_{i+1}"] = np.clip(base + seasonal + noise, 0.01, None)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_weekly_dataframe():
    """Generate a sample weekly multi-site DataFrame for testing."""
    dates = pd.date_range(start="2010-01-03", end="2020-12-27", freq="W-SUN")
    np.random.seed(42)
    n_sites = 3
    data = {}
    for i in range(n_sites):
        base = np.random.gamma(shape=2.0, scale=25.0, size=len(dates))
        seasonal = 12 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
        noise = np.random.normal(0, 5, size=len(dates))
        data[f"site_{i+1}"] = np.clip(base + seasonal + noise, 0.01, None)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_correlation_matrix():
    """Generate a sample correlation matrix for testing."""
    np.random.seed(42)
    # Create a positive definite correlation matrix
    n = 5
    A = np.random.randn(n, n)
    corr = A @ A.T
    # Normalize to correlation matrix
    D = np.diag(1.0 / np.sqrt(np.diag(corr)))
    corr = D @ corr @ D
    return corr


@pytest.fixture
def sample_non_psd_matrix():
    """Generate a non-positive semi-definite matrix for testing repair algorithms."""
    # Create a matrix with negative eigenvalues
    matrix = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.95], [0.8, 0.95, 1.0]])
    # Force it to be non-PSD by making eigenvalues negative
    matrix[0, 1] = 1.1  # This makes it invalid
    return matrix


@pytest.fixture
def temp_hdf5_file(tmp_path):
    """Create a temporary HDF5 file path for testing."""
    return tmp_path / "test_ensemble.h5"


@pytest.fixture
def sample_ensemble_data():
    """Generate sample ensemble data for testing."""
    dates = pd.date_range(start="2010-01-01", end="2012-12-31", freq="D")
    np.random.seed(42)

    # Create ensemble with 3 realizations and 2 sites
    ensemble = {}
    for realization in range(3):
        data = {}
        for site in ["site_1", "site_2"]:
            np.random.seed(42 + realization)
            data[site] = np.random.gamma(shape=2.0, scale=50.0, size=len(dates))
        ensemble[realization] = pd.DataFrame(data, index=dates)

    return ensemble


@pytest.fixture
def sample_annual_series():
    """Generate a sample annual time series for testing."""
    dates = pd.date_range(start="2000-01-01", end="2020-12-31", freq="YS")
    np.random.seed(42)
    values = np.random.gamma(shape=2.0, scale=1000.0, size=len(dates))
    return pd.Series(values, index=dates, name="site_1")


@pytest.fixture
def sample_annual_dataframe():
    """Generate a sample annual multi-site DataFrame for testing."""
    dates = pd.date_range(start="2000-01-01", end="2020-12-31", freq="YS")
    np.random.seed(42)
    n_sites = 3
    data = {}
    for i in range(n_sites):
        base = np.random.gamma(shape=2.0, scale=1000.0, size=len(dates))
        noise = np.random.normal(0, 100, size=len(dates))
        data[f"site_{i+1}"] = np.clip(base + noise, 0.01, None)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_ssi_data():
    """Generate sample data for SSI drought metrics testing."""
    dates = pd.date_range(start="2000-01-01", end="2020-12-31", freq="D")
    np.random.seed(42)

    # Create data with some drought periods (low flows)
    values = np.random.gamma(shape=2.0, scale=50.0, size=len(dates))

    # Inject some drought periods
    drought_indices = [
        slice(365, 465),  # Year 2 drought
        slice(1095, 1195),  # Year 4 drought
        slice(3650, 3800),  # Year 11 drought
    ]
    for idx in drought_indices:
        values[idx] = values[idx] * 0.3  # Reduce flows significantly

    return pd.Series(values, index=dates, name="flow")
