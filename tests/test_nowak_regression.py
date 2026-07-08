"""
Bit-identity regression tests for NowakDisaggregator monthly-to-daily output.

The golden file (tests/data/nowak_m2d_golden.npz) was captured from the
pre-generalization implementation. These tests guarantee that refactoring the
disaggregator to support other timescale pairs does not change the
monthly-to-daily output in any way for a fixed seed.

To (re)capture the golden file against the current implementation:

    venv/Scripts/python tests/test_nowak_regression.py --capture
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.methods.disaggregation.temporal.nowak import NowakDisaggregator

GOLDEN_PATH = Path(__file__).parent / "data" / "nowak_m2d_golden.npz"
SEED = 1234


def _blend_kwarg(value):
    """Return the blend kwarg under whichever name the constructor accepts."""
    params = inspect.signature(NowakDisaggregator.__init__).parameters
    if "boundary_blend_timesteps" in params:
        return {"boundary_blend_timesteps": value}
    return {"blend_days": value}


def _make_daily_obs(multisite):
    """Deterministic daily observations, 3 sites or 1 site, 2010-2015."""
    dates = pd.date_range(start="2010-01-01", end="2015-12-31", freq="D")
    rng = np.random.default_rng(20260708)
    n_sites = 3 if multisite else 1
    data = {}
    for i in range(n_sites):
        base = rng.gamma(shape=2.0, scale=50.0, size=len(dates))
        noise = rng.normal(0, 10, size=len(dates))
        data[f"site_{i + 1}"] = np.clip(base + noise, 0.01, None)
    df = pd.DataFrame(data, index=dates)
    if multisite:
        return df
    return df.iloc[:, 0]


def _make_monthly_ensemble(daily_obs):
    """Two-realization monthly ensemble spanning 2016-2017 (2016 is leap)."""
    if isinstance(daily_obs, pd.Series):
        daily_obs = daily_obs.to_frame()
    monthly = daily_obs.resample("MS").sum()
    values = monthly.iloc[:24].to_numpy()
    index = pd.date_range("2016-01-01", periods=24, freq="MS")
    r0 = pd.DataFrame(values, index=index, columns=monthly.columns)
    r1 = r0 * 1.1
    metadata = EnsembleMetadata(
        n_realizations=2,
        n_sites=r0.shape[1],
        time_resolution="MS",
        time_period=(str(index[0].date()), str(index[-1].date())),
    )
    return Ensemble({0: r0, 1: r1}, metadata=metadata)


CASES = {
    "multisite_default": {
        "multisite": True,
        "blend": 2,
        "sample_method": "distance_weighted",
    },
    "multisite_noblend": {
        "multisite": True,
        "blend": 0,
        "sample_method": "distance_weighted",
    },
    "multisite_lall_sharma": {
        "multisite": True,
        "blend": 2,
        "sample_method": "lall_and_sharma_1996",
    },
    "singlesite_default": {
        "multisite": False,
        "blend": 2,
        "sample_method": "distance_weighted",
    },
}


def _run_case(case):
    """Fit on deterministic data and disaggregate a seeded ensemble."""
    daily_obs = _make_daily_obs(case["multisite"])
    disagg = NowakDisaggregator(**_blend_kwarg(case["blend"]))
    disagg.fit(daily_obs)
    ensemble = _make_monthly_ensemble(daily_obs)
    daily_ensemble = disagg.disaggregate(
        ensemble, sample_method=case["sample_method"], seed=SEED
    )
    out = {}
    for rid, df in daily_ensemble.data_by_realization.items():
        out[f"values_{rid}"] = df.to_numpy(dtype=np.float64)
        out[f"index_{rid}"] = df.index.asi8
    return out


def capture():
    """Write the golden file from the current implementation."""
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, case in CASES.items():
        for key, arr in _run_case(case).items():
            arrays[f"{name}__{key}"] = arr
    np.savez(GOLDEN_PATH, **arrays)
    print(f"Captured golden file: {GOLDEN_PATH}")


@pytest.mark.parametrize("case_name", list(CASES))
def test_monthly_to_daily_bit_identical(case_name):
    """Disaggregated output must exactly match the pre-refactor golden file."""
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"Golden file missing: {GOLDEN_PATH}. "
            "Run: python tests/test_nowak_regression.py --capture"
        )
    golden = np.load(GOLDEN_PATH)
    result = _run_case(CASES[case_name])
    for key, arr in result.items():
        golden_key = f"{case_name}__{key}"
        assert golden_key in golden, f"Golden file lacks {golden_key}"
        assert np.array_equal(
            golden[golden_key], arr
        ), f"Case {case_name}: {key} differs from golden output"


if __name__ == "__main__":
    import sys

    if "--capture" in sys.argv:
        capture()
    else:
        print("Usage: python tests/test_nowak_regression.py --capture")
