# Getting Started

## Installation

SynHydro is not yet published on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/TrevorJA/SynHydro.git
```

For development (editable install with dev extras):

```bash
git clone https://github.com/TrevorJA/SynHydro.git
cd SynHydro
pip install -e ".[dev]"
```

## Data Format

All generators expect a `pd.DataFrame` with a `DatetimeIndex`:

| Property | Requirement |
|----------|------------|
| Index | `DatetimeIndex` |
| Monthly data | `freq='MS'` (month-start) |
| Daily data | `freq='D'` |
| Columns | Site names (one column per gauge) |
| Units | cfs, MGD, or cms (consistent) |

```python
import pandas as pd

# Minimal example: single site, monthly
dates = pd.date_range("1980-01", periods=480, freq="MS")
Q_obs = pd.DataFrame({"site_A": [...]}, index=dates)
```

Use `synhydro.load_example_data()` to get a ready-to-use daily dataset:

```python
import synhydro

Q_daily = synhydro.load_example_data()                 # USGS daily streamflow (cms)
Q_monthly = Q_daily.resample("MS").sum()            # aggregate to monthly
```

## Choosing a Generator

| Need | Generator |
|------|-----------|
| Monthly, single-site, parametric | `ThomasFieringGenerator` |
| Monthly, multi-site, parametric | `MatalasGenerator` |
| Monthly, multi-site, nonparametric | `KirschGenerator` |
| Daily, single-site | `PhaseRandomizationGenerator` |
| Annual, single-site | `WARMGenerator` |
| Annual, multi-site, drought-aware | `MultiSiteHMMGenerator` |
| Monthly-to-Daily disaggregation | `NowakDisaggregator` (or use a Pipeline) |

## Basic Workflow

Every generator follows the same two-step pattern:

```python
gen = synhydro.ThomasFieringGenerator()
gen.fit(Q_obs)                                      # validate, prepare, and estimate parameters
ensemble = gen.generate(n_realizations=10,           # synthetic flows
                        n_years=30,
                        seed=42)
```

!!! tip "Next: Tutorials"
    Work through the hands-on tutorials covering single-site generation, multi-site
    correlation, monthly-to-daily disaggregation, drought analysis, and ensemble
    verification and validation: [Tutorials](tutorials/index.md).
