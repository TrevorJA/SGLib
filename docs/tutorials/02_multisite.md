# Multi-Site Monthly Generation

Single-site generators ignore spatial dependencies between gages.
`KirschGenerator` is a nonparametric bootstrap that preserves cross-site
correlations using Cholesky decomposition.

## Fit and generate

```python
import synhydro

Q_daily = synhydro.load_example_data()
Q_monthly = Q_daily.resample("MS").sum()    # all sites, monthly

gen = synhydro.KirschGenerator()
gen.fit(Q_monthly)
ensemble = gen.generate(n_realizations=50, n_years=30, seed=42)
```

## Verify spatial correlations

The key diagnostic for a multi-site generator is whether pairwise correlations
match the historical record. Side-by-side heatmaps make this easy to assess:

```python
from synhydro.plotting import plot_spatial_correlation

fig, axes = plot_spatial_correlation(
    ensemble,
    observed=Q_monthly,
    timestep="monthly",
)
```

![Side-by-side heatmaps of observed and synthetic pairwise Pearson correlations across sites](../assets/images/tutorials/02_spatial_correlation.png){: width="700px" }

## Next steps

- **Monthly-to-daily disaggregation** - [Tutorial 03](03_disaggregator.md)
- **Monthly-to-daily pipeline** - [Tutorial 04](04_pipeline.md)
- **Quantitative validation** - [Tutorial 06](06_validation.md)
- **Algorithm details** - [Kirsch Bootstrap](../algorithms/kirsch.md)

---

**Previous:** [Quickstart (Thomas-Fiering)](01_quickstart.md) | **Next:** [Disaggregator (Nowak)](03_disaggregator.md)
