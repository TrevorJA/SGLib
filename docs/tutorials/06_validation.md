# Ensemble Validation

After generating synthetic flows you need to verify that they reproduce the
statistical properties of the historical record. `validate_ensemble` computes
a comprehensive suite of metrics comparing the ensemble against observed data.

## Setup

```python
import synhydro

Q_daily = synhydro.load_example_data()
Q_monthly = Q_daily.resample("MS").sum()

gen = synhydro.KirschGenerator()
gen.fit(Q_monthly)
ensemble = gen.generate(n_realizations=100, n_years=50, seed=42)
```

## Run validation

`validate_ensemble` compares the ensemble to observed data across eight
metric categories:

| Category | What it checks |
|----------|----------------|
| **Marginal** | Mean, std, skewness, kurtosis, CV, percentiles, KS test |
| **Temporal** | Lag-1/2 autocorrelation, Hurst exponent, ACF RMSE |
| **Spatial** | Cross-site correlation RMSE and bias |
| **Drought** | Mean/max duration, mean/max severity, frequency |
| **Spectral** | Power spectrum RMSE, correlation, low-frequency ratio |
| **Seasonal** | Per-month mean/std/skewness bias, Wilcoxon p-values |
| **Annual** | Annual mean, variance, skewness, lag-1 ACF, variance ratio |
| **FDC** | Flow duration curve RMSE, bias at Q10/Q50/Q90, envelope coverage |

```python
result = synhydro.validate_ensemble(ensemble, Q_monthly)
```

You can also request a subset of categories:

```python
result = synhydro.validate_ensemble(
    ensemble, Q_monthly,
    metrics=["marginal", "seasonal", "fdc"],
)
```

## Explore results

Each metric entry contains the observed value, synthetic median and spread,
and relative error:

```python
site = Q_monthly.columns[0]
for metric, values in result.marginal[site].items():
    print(
        f"  {metric:12s}  obs={values['observed']:.3f}  "
        f"syn={values['synthetic_median']:.3f}  "
        f"rel_err={values['relative_error']:.3f}"
    )
```

Summary scores give a single-number snapshot of generation quality:

```python
for score, value in result.summary.items():
    print(f"  {score}: {value:.4f}")
```

For further analysis, flatten results into a tidy DataFrame:

```python
df = result.to_dataframe()
df[df["relative_error"].abs() > 0.1]   # flag large errors
```

## Visual validation

`plot_validation_panel` produces a 5-panel figure comparing observed and
synthetic distributions by month, with Wilcoxon rank-sum and Levene test
p-values. Values above the dashed line (p = 0.05) indicate the distributions
are statistically indistinguishable.

```python
from synhydro.plotting import plot_validation_panel

fig, axes = plot_validation_panel(
    ensemble,
    observed=Q_monthly[site],
    site=site,
)
```

![Five-panel validation: boxplots by month, monthly mean and std comparison, Wilcoxon and Levene p-values](../assets/images/tutorials/06_validation_panel.png){: width="700px" }

!!! tip "Log-space validation"
    Streamflow is often right-skewed. Validating in log space can reveal
    differences in the lower tail:
    ```python
    fig, axes = plot_validation_panel(
        ensemble, observed=Q_monthly[site], log_space=True
    )
    ```

## Next steps

- **Diagnostic plots** for specific properties (ACF, FDC, spatial correlation)
  are available in `synhydro.plotting` -- see the [API reference](../api/generators.md)
- **Algorithm details** can help diagnose which statistical properties a
  generator is designed to preserve -- [Algorithms](../algorithms/index.md)

---

**Previous:** [Drought Analysis (SSI)](05_drought_analysis.md)
