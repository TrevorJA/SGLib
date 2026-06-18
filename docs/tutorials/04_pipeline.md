# Monthly-to-Daily Pipeline

Many applications need daily synthetic flows, but the most robust generation
methods operate at a monthly scale. SynHydro's **pipelines** chain a monthly
generator with a temporal disaggregator in a single interface.

`KirschNowakPipeline` combines the Kirsch monthly bootstrap with the Nowak
KNN disaggregator. You provide daily observed data; it handles the internal
monthly aggregation and disaggregation automatically.

## Generate daily synthetic flows

```python
import synhydro

Q_daily = synhydro.load_example_data()

pipeline = synhydro.KirschNowakPipeline()
pipeline.fit(Q_daily)
daily_ensemble = pipeline.generate(n_realizations=10, n_years=30, seed=42)
```

```python
Q_syn_daily = daily_ensemble.data_by_realization[0]
print(Q_syn_daily.shape)  # (~10957 days x n_sites)
```

!!! tip "Single-site alternative"
    ```python
    pipeline = synhydro.ThomasFieringNowakPipeline()
    pipeline.fit(Q_daily.iloc[:, [0]])
    ```

## Visualize

Plot a one-year window to inspect daily variability:

```python
from synhydro.plotting import plot_timeseries

site = Q_daily.columns[0]

fig, ax = plot_timeseries(
    daily_ensemble,
    observed=Q_daily[site],
    start_date="2000-01-01",
    end_date="2000-12-31",
    show_members=3,
)
```

![Daily synthetic hydrograph for one year with ensemble uncertainty and observed flows](../assets/images/tutorials/04_daily_timeseries.png){: width="700px" }

## Next steps

- **Drought analysis** on synthetic flows - [Tutorial 05](05_drought_analysis.md)
- **Ensemble validation** - [Tutorial 06](06_validation.md)
- **Algorithm details** - [Kirsch Bootstrap](../algorithms/kirsch.md),
  [Nowak Disaggregation](../algorithms/nowak_disaggregation.md)

---

**Previous:** [Disaggregator (Nowak)](03_disaggregator.md) | **Next:** [Drought Analysis (SSI)](05_drought_analysis.md)
