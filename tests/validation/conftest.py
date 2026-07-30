"""Shared fixtures for validation suite tests."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble, EnsembleMetadata


@pytest.fixture(scope="module")
def monthly_observed():
    """20-year monthly observed data, 2 sites, lognormal with seasonality."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", "2019-12-31", freq="MS")
    n = len(dates)
    seasonal = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 12)
    data = {
        "site_A": rng.lognormal(5.0, 0.4, n) * seasonal,
        "site_B": rng.lognormal(4.5, 0.5, n) * seasonal,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture(scope="module")
def monthly_ensemble():
    """Ensemble of 10 realizations x 20 years, 2 sites."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    n = len(dates)
    seasonal = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 12)

    realization_dict = {}
    for i in range(10):
        data = {
            "site_A": rng.lognormal(5.0, 0.4, n) * seasonal,
            "site_B": rng.lognormal(4.5, 0.5, n) * seasonal,
        }
        realization_dict[i] = pd.DataFrame(data, index=dates)

    metadata = EnsembleMetadata(
        generator_class="TestGenerator",
        n_realizations=10,
        n_sites=2,
        time_resolution="MS",
    )
    return Ensemble(realization_dict, metadata=metadata)
