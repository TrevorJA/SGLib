# Working with Ensembles

Every generator and disaggregator in SynHydro returns an `Ensemble`: a
container that holds many synthetic realizations for many sites, plus
metadata about how they were produced. This tutorial covers the data model,
how to build an `Ensemble` from your own data, and the built-in analysis and
I/O methods.

## The dual data model

An `Ensemble` maintains two complementary views of the same data:

| View | Keys | Each value |
|------|------|------------|
| `data_by_realization` | realization index (`int`) | `DataFrame` [time x sites] |
| `data_by_site` | site name (`str`) | `DataFrame` [time x realizations] |

Use whichever fits the task: per-realization DataFrames feed simulation
models one trace at a time, while per-site DataFrames make it easy to
compute distributional statistics across realizations.

```python
import synhydro

Q_daily = synhydro.load_example_data()
Q_monthly = Q_daily.resample("MS").sum()

gen = synhydro.KirschGenerator()
gen.fit(Q_monthly)
ensemble = gen.generate(n_realizations=50, n_years=30, seed=42)

ensemble.realization_ids       # [0, 1, ..., 49]
ensemble.site_names            # site names from the fitted data

trace = ensemble.data_by_realization[0]                # one trace, all sites
site_flows = ensemble.data_by_site["USGS-01434000"]    # one site, all realizations
```

The by-site view is computed lazily from the by-realization view the first
time you access it, so holding an `Ensemble` costs no more memory than the
realization dictionary until you need the second view.

## Building an Ensemble from your own data

Pass a dictionary in either orientation; the structure is detected from the
key type. Integer keys mean the dictionary is keyed by realization, string
keys mean it is keyed by site.

```python
import pandas as pd
from synhydro.core.ensemble import Ensemble, EnsembleMetadata

# Keyed by realization: each DataFrame is [time x sites]
data = {
    0: df_realization_0,
    1: df_realization_1,
}

# Or keyed by site: each DataFrame is [time x realizations]
data = {
    "Hopland": df_hopland,      # columns are realization labels
    "Healdsburg": df_healdsburg,
}

ensemble = Ensemble(
    data,
    metadata=EnsembleMetadata(
        time_resolution="YS",
        description="Annual paleo reconstruction ensemble",
    ),
)
```

!!! note "Realization keys are integers"
    Realization identifiers are always integers. When you pass site-keyed
    data whose columns are integer-like strings (for example `"1"`, `"2"`
    read from CSV headers), the labels are coerced to `int` automatically.
    Labels that cannot be coerced (for example `"ens_a"`) are kept as-is
    and a warning is logged, because string realization keys are
    indistinguishable from site names if the dictionary is later used to
    build another `Ensemble`.

Set `time_resolution` in the metadata when constructing from raw data.
Disaggregators check it to confirm the input frequency (for example `"YS"`
for annual input to an annual-to-monthly `NowakDisaggregator`).

## Summary statistics

`summary` reduces each site (or realization) to scalar statistics, and
`percentile` returns time-varying quantiles across realizations:

```python
stats = ensemble.summary(by="site")            # mean/std/min/max per site
bands = ensemble.percentile([10, 50, 90], by="site")
median_flow = bands["USGS-01434000"]["p50"]    # median across realizations
```

## Subsetting and resampling

Both return a new `Ensemble` and leave the original untouched:

```python
subset = ensemble.subset(
    sites=["USGS-01434000", "USGS-01438500"],
    realizations=[0, 1, 2],
    start_date="2000-01-01",
    end_date="2010-12-31",
)

annual = ensemble.resample("YS")   # sums to the new frequency
```

## Saving and loading

Ensembles serialize to HDF5. By default data is grouped by site
(`stored_by_node=True`), which matches the layout expected by downstream
tools such as Pywr-DRB:

```python
ensemble.to_hdf5("synthetic_flows.h5")

loaded = Ensemble.from_hdf5("synthetic_flows.h5")
first_three = Ensemble.from_hdf5(
    "synthetic_flows.h5", realization_subset=[0, 1, 2]
)
```

## Next steps

- **Quantitative validation of an ensemble** - [Tutorial 06](06_verification.md)
- **Plotting ensembles** - [Tutorial 07](07_plotting.md)
- **Full API reference** - [Core Data Structures](../api/core.md)

---

**Previous:** [Plotting Walkthrough](07_plotting.md)
