# Quickstart: Single-Site Monthly Generation

This tutorial demonstrates the core SynHydro workflow: load observed data,
fit a generator, generate synthetic realizations, and plot the results.

## Load example data

```python
import synhydro

Q_daily = synhydro.load_example_data()       # multi-site daily DataFrame
Q_monthly = Q_daily.resample("MS").sum()     # aggregate to monthly
site = Q_monthly.columns[0]                  # pick one site
Q_single = Q_monthly[[site]]                 # keep as DataFrame
```

## Fit and generate

Every generator follows two steps: **fit**, **generate**.
`ThomasFieringGenerator` fits a seasonal AR(1) model to single-site monthly data.

```python
gen = synhydro.ThomasFieringGenerator()
gen.fit(Q_single)
ensemble = gen.generate(n_realizations=50, n_years=30, seed=42)
```

The returned `Ensemble` contains 50 synthetic 30-year monthly timeseries.

```python
Q_syn_0 = ensemble.data_by_realization[0]
print(Q_syn_0.shape)  # (360, 1) -- 30 years x 12 months
```

!!! note "Seed reproducibility"
    Passing the same `seed` value always produces identical results.

## Visualize

```python
from synhydro.plotting import plot_timeseries, plot_flow_duration_curve

fig, ax = plot_timeseries(
    ensemble,
    observed=Q_monthly[site],
    show_members=3,
)
```

![Ensemble timeseries: synthetic members, median, and 10-90th percentile envelope plotted against observed monthly flows](../assets/images/tutorials/01_timeseries.png){: width="700px" }

```python
fig, ax = plot_flow_duration_curve(
    ensemble,
    observed=Q_monthly[site],
)
```

![Flow duration curve: observed FDC overlaid on ensemble uncertainty bands](../assets/images/tutorials/01_fdc.png){: width="700px" }

## Next steps

- **Multi-site generation** - [Tutorial 02](02_multisite.md)
- **Monthly-to-daily disaggregation** - [Tutorial 03](03_disaggregator.md)
- **Algorithm details** - [Thomas-Fiering AR(1)](../algorithms/thomas_fiering.md)

---

**Next:** [Multi-Site Monthly (Kirsch)](02_multisite.md)
