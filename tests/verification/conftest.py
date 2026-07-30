"""Shared fixtures for verification suite tests."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble

N_YEARS = 30
SITES = ["site_a", "site_b"]


def lognormal_frame(index, rng, sites=SITES, mu=3.0, sigma=0.5):
    return pd.DataFrame(
        {site: rng.lognormal(mean=mu, sigma=sigma, size=len(index)) for site in sites},
        index=index,
    )


@pytest.fixture(scope="module")
def monthly_index():
    return pd.date_range("1980-01-01", periods=N_YEARS * 12, freq="MS")


@pytest.fixture(scope="module")
def monthly_observed(monthly_index):
    rng = np.random.default_rng(42)
    return lognormal_frame(monthly_index, rng)


@pytest.fixture(scope="module")
def monthly_ensemble(monthly_index):
    rng = np.random.default_rng(7)
    data = {rid: lognormal_frame(monthly_index, rng) for rid in range(10)}
    return Ensemble(data)


@pytest.fixture(scope="module")
def daily_index():
    return pd.date_range("2000-01-01", periods=12 * 365, freq="D")


@pytest.fixture(scope="module")
def daily_observed(daily_index):
    rng = np.random.default_rng(11)
    return lognormal_frame(daily_index, rng, sites=["site_a"])


@pytest.fixture(scope="module")
def daily_ensemble(daily_index):
    rng = np.random.default_rng(13)
    data = {
        rid: lognormal_frame(daily_index, rng, sites=["site_a"]) for rid in range(4)
    }
    return Ensemble(data)


def ar1_series(n, phi, seed, index=None):
    """AR(1) series with autocorrelation phi at lag 1."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    values = np.empty(n)
    values[0] = noise[0]
    for i in range(1, n):
        values[i] = phi * values[i - 1] + noise[i] * np.sqrt(1 - phi**2)
    if index is None:
        index = pd.date_range("1700-01-01", periods=n, freq="MS")
    return pd.Series(values, index=index)
