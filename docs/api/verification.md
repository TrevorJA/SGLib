# Verification

The `synhydro.verification` module evaluates statistical property
preservation: whether a synthetic ensemble reproduces the moments,
correlations, and distributional shape of the observed record.

The terminology follows Stedinger and Taylor (1982). *Verification*
demonstrates that generated flows reproduce the statistics the
generator was designed to reproduce. *Validation* demonstrates that
characteristics not explicitly fit, such as drought behavior, are also
consistent with the historical record; see
[Validation](validation.md).

## How results are reported

Every metric is computed once on the observed record and once on each
realization. The result compares the observed statistic against the
distribution of the statistic across realizations, following the
standard reporting convention of the synthetic streamflow literature:

- `syn_median`, `syn_q05`, ..., `syn_q95`: quantiles of the statistic
  across realizations.
- `obs_percentile`: the position of the observed value within the
  synthetic sample, computed as `(n_below + 0.5 * n_equal + 0.5) / (n + 1)`
  where `n` is the number of realizations. Values near 0 or 1 mean the
  observed statistic lies in the tail of the ensemble distribution;
  values near 0.5 mean it is central.
- `in_90_band`: whether the observed value falls between the ensemble's
  5th and 95th percentiles.
- `relative_diff`: `(syn_median - observed) / |observed|`, omitted when
  the observed value is too close to zero for a ratio to be meaningful.

There is no single cross-category score: metrics with different units
are never averaged together. `category_summary()` provides a unit-free
rollup per category.

## Metric kinds

Each metric has one of four kinds, which determines its signature and
how it appears in the tidy frame:

| Kind | Signature | Reported as |
|------|-----------|-------------|
| `scalar` | `f(x: pd.Series) -> float` | one value per site per realization |
| `curve` | `f(x: pd.Series) -> pd.Series` | one value per component (month, lag, exceedance probability, period band) |
| `matrix` | `f(frame: pd.DataFrame) -> pd.Series` | one value per site pair |
| `comparison` | `f(x, reference) -> float or pd.Series` | divergence of one realization from observed; no observed value of its own |

## Metric selection

`verify()` requires an explicit metric selection: `metrics="all"`, a
list of metric names, category names, or callables. There is no
default subset, so a report never silently omits categories.

```python
import synhydro

result = synhydro.verify(ensemble, Q_obs, metrics="all")
result = synhydro.verify(ensemble, Q_obs, metrics=["marginal", "acf"])
result.summary()
```

## Metric reference

| Metric | Category | Kind | Units | Frequencies | Citation |
|--------|----------|------|-------|-------------|----------|
| `mean` | marginal | scalar | flow | any | Matalas (1967); Stedinger and Taylor (1982) |
| `std` | marginal | scalar | flow | any | Matalas (1967); Stedinger and Taylor (1982) |
| `cv` | marginal | scalar | dimensionless | any | Matalas (1967); Stedinger and Taylor (1982) |
| `skewness` | marginal | scalar | dimensionless | any | Matalas (1967); Stedinger and Taylor (1982) |
| `kurtosis` | marginal | scalar | dimensionless | any | Matalas (1967); Stedinger and Taylor (1982) |
| `minimum` | marginal | scalar | flow | any | Matalas (1967); Stedinger and Taylor (1982) |
| `maximum` | marginal | scalar | flow | any | Matalas (1967); Stedinger and Taylor (1982) |
| `flow_q10` | marginal | scalar | flow | any | Matalas (1967); Stedinger and Taylor (1982) |
| `flow_q50` | marginal | scalar | flow | any | Matalas (1967); Stedinger and Taylor (1982) |
| `flow_q90` | marginal | scalar | flow | any | Matalas (1967); Stedinger and Taylor (1982) |
| `ks_statistic` | marginal | comparison | dimensionless | any | Two-sample Kolmogorov-Smirnov distance |
| `lag1_autocorrelation` | temporal | scalar | dimensionless | any | Matalas (1967) |
| `lag2_autocorrelation` | temporal | scalar | dimensionless | any | Matalas (1967) |
| `acf` | temporal | curve | dimensionless | any | Salas et al. (1980); Kirsch et al. (2013) |
| `hurst` | temporal | scalar | dimensionless | any, 20+ years | Hurst (1951); Koutsoyiannis (2002) |
| `monthly_mean` | seasonal | curve | flow | sub-annual | Lall and Sharma (1996); Nowak et al. (2010) |
| `monthly_std` | seasonal | curve | flow | sub-annual | Lall and Sharma (1996); Nowak et al. (2010) |
| `monthly_skewness` | seasonal | curve | dimensionless | sub-annual | Lall and Sharma (1996); Nowak et al. (2010) |
| `monthly_maximum` | seasonal | curve | flow | sub-annual | Lall and Sharma (1996); Nowak et al. (2010) |
| `monthly_minimum` | seasonal | curve | flow | sub-annual | Lall and Sharma (1996); Nowak et al. (2010) |
| `monthly_lag1_correlation` | seasonal | curve | dimensionless | sub-annual | Kirsch et al. (2013) |
| `monthly_ranksum_pvalue` | seasonal | comparison | pvalue | sub-annual | Herman et al. (2016) |
| `monthly_levene_pvalue` | seasonal | comparison | pvalue | sub-annual | Herman et al. (2016) |
| `annual_mean` | annual | scalar | flow (annual total) | any, 10+ years | Stedinger and Taylor (1982) |
| `annual_sd` | annual | scalar | flow (annual total) | any, 10+ years | Stedinger and Taylor (1982); Srinivas and Srinivasan (2005) |
| `annual_cv` | annual | scalar | dimensionless | any, 10+ years | Stedinger and Taylor (1982) |
| `annual_skewness` | annual | scalar | dimensionless | any, 10+ years | Stedinger and Taylor (1982) |
| `annual_lag1_autocorrelation` | annual | scalar | dimensionless | any, 10+ years | Stedinger and Taylor (1982) |
| `annual_minimum` | annual | scalar | flow (annual total) | any, 10+ years | Stedinger and Taylor (1982) |
| `annual_maximum` | annual | scalar | flow (annual total) | any, 10+ years | Stedinger and Taylor (1982) |
| `cross_correlation` | spatial | matrix | dimensionless | any | Matalas (1967); Tsoukalas et al. (2018) |
| `cross_correlation_lag1` | spatial | matrix | dimensionless | any | Matalas (1967); Tsoukalas et al. (2018) |
| `fdc` | fdc | curve | flow | any | Vogel and Fennessey (1995) |
| `fdc_log_rmse` | fdc | comparison | dimensionless | any | Vogel and Fennessey (1995) |
| `l_cv` | lmoments | scalar | dimensionless | any | Hosking (1990) |
| `l_skewness` | lmoments | scalar | dimensionless | any | Hosking (1990) |
| `l_kurtosis` | lmoments | scalar | dimensionless | any | Hosking (1990) |
| `annual_max_mean` | extremes | scalar | flow | any, 10+ years | Stedinger et al. (1993); Zaerpour et al. (2021) |
| `annual_max_cv` | extremes | scalar | dimensionless | any, 10+ years | Stedinger et al. (1993); Zaerpour et al. (2021) |
| `gev_rp10` | extremes | scalar | flow | any, 10+ years | Stedinger et al. (1993); Zaerpour et al. (2021) |
| `gev_rp50` | extremes | scalar | flow | any, 10+ years | Stedinger et al. (1993); Zaerpour et al. (2021) |
| `gev_rp100` | extremes | scalar | flow | any, 10+ years | Stedinger et al. (1993); Zaerpour et al. (2021) |
| `annual_min_mean` | extremes | scalar | flow | any, 10+ years | Stedinger et al. (1993); Zaerpour et al. (2021) |
| `annual_min_cv` | extremes | scalar | dimensionless | any, 10+ years | Stedinger et al. (1993); Zaerpour et al. (2021) |
| `seven_day_min_mean` | extremes | scalar | flow | daily, 10+ years | Stedinger et al. (1993); Zaerpour et al. (2021) |
| `seven_day_min_cv` | extremes | scalar | dimensionless | daily, 10+ years | Stedinger et al. (1993); Zaerpour et al. (2021) |
| `spectral_density` | spectral | curve | dimensionless | any | Nowak et al. (2011) |
| `low_frequency_variance_fraction` | spectral | scalar | dimensionless | any | Nowak et al. (2011) |

Notes:

- P-value metrics (`monthly_ranksum_pvalue`, `monthly_levene_pvalue`)
  are computed per realization and summarized as a rejection rate at
  alpha = 0.05, never as a pseudo relative error. Under a perfect
  generator the rejection rate is near alpha.
- `hurst` is estimated on annually aggregated flows because sub-annual
  estimation conflates seasonal persistence with long-range dependence
  (Stedinger and Taylor, 1982).
- Frequency gates are enforced automatically; skipped metrics are
  recorded in `result.skipped` with a reason.

---

## Orchestrator and result

::: synhydro.verification.verify

::: synhydro.verification.VerificationResult

---

## Bootstrap and comparison tools

::: synhydro.verification.bootstrap_metric_ci

::: synhydro.verification.compare_methods

---

## Extending the suite

::: synhydro.verification.register_metric

::: synhydro.verification.list_metrics

---

## Metric functions

All metric functions are directly callable, e.g.
`synhydro.verification.lag1_autocorrelation(series)`.

::: synhydro.verification.metrics.marginal

::: synhydro.verification.metrics.temporal

::: synhydro.verification.metrics.seasonal

::: synhydro.verification.metrics.annual

::: synhydro.verification.metrics.spatial

::: synhydro.verification.metrics.fdc

::: synhydro.verification.metrics.lmoments

::: synhydro.verification.metrics.extremes

::: synhydro.verification.metrics.spectral
