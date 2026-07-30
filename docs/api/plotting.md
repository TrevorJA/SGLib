# Plotting

The `synhydro.plotting` module provides standardized plotting utilities for
ensemble hydrologic data. All functions accept an `Ensemble` object directly
and return `(fig, ax)` (or `(fig, axes)` for multi-panel layouts), so they
compose easily inside larger figures.

The module follows a few consistent conventions:

- **Ensemble-first inputs.** Every function takes an `Ensemble` as the first
  positional argument; `observed` is optional and rendered as an overlay.
- **Percentile bands by default.** Ensemble-vs-time plots show a median line
  with 10th-90th percentile shading, optionally with a few member traces.
- **Single source of truth.** Colors, line widths, and layout defaults live in
  `COLORS`, `STYLE`, `LAYOUT`, and `LABELS`. Override these dicts and call
  `apply_plotting_style()` to re-theme every plot in a script.
- **Optional auto-save.** Pass `filename=...` to write the figure to disk at
  `dpi` (default 300) without an explicit `fig.savefig` call.

## Quick reference

| Function | Description |
|----------|-------------|
| `plot_timeseries` | Ensemble timeseries with median, percentile band, and optional members |
| `plot_flow_ranges` | Min/max/median ranges aggregated by daily, weekly, or monthly periods |
| `plot_seasonal_cycle` | Per-period mean or std with inter-realization band |
| `plot_flow_duration_curve` | FDC with ensemble uncertainty and optional annual range |
| `plot_cdf` | Empirical CDF with ensemble uncertainty |
| `plot_histogram` | Flow histogram with optional KDE overlay |
| `plot_monthly_distributions` | Monthly boxplot or violin plot of ensemble vs observed |
| `plot_autocorrelation` | Lagged autocorrelation with ensemble range |
| `plot_spatial_correlation` | Multi-site correlation heatmap (ensemble, observed, or difference) |
| `plot_drought_characteristics` | Scatter of drought duration, magnitude, and severity |
| `plot_ssi_timeseries` | SSI timeseries with drought-severity shading |
| `plot_verification_panel` | 5-panel marginal and seasonal verification figure |
| `plot_metric_distributions` | Metric boxplots across realizations with observed overlay |
| `plot_metric_curve` | Ensemble band for curve metrics (FDC, ACF, monthly statistics) |
| `apply_plotting_style` | Apply current `COLORS`/`STYLE` to matplotlib rcParams |
| `warn_if_many_realizations` | Emit a warning if an ensemble is too large for a given plot |
| `warn_if_few_realizations` | Emit a warning if an ensemble is too small for stable percentiles |

---

## Timeseries

::: synhydro.plotting.plot_timeseries

::: synhydro.plotting.plot_flow_ranges

::: synhydro.plotting.plot_seasonal_cycle

---

## Distributions

::: synhydro.plotting.plot_flow_duration_curve

::: synhydro.plotting.plot_cdf

::: synhydro.plotting.plot_histogram

::: synhydro.plotting.plot_monthly_distributions

---

## Correlation

::: synhydro.plotting.plot_autocorrelation

::: synhydro.plotting.plot_spatial_correlation

---

## Drought

::: synhydro.plotting.plot_drought_characteristics

::: synhydro.plotting.plot_ssi_timeseries

---

## Verification and Validation

::: synhydro.plotting.plot_verification_panel

::: synhydro.plotting.plot_metric_distributions

::: synhydro.plotting.plot_metric_curve

---

## Configuration

The configuration dictionaries are imported directly from
`synhydro.plotting`. Mutate them in place and then call
`apply_plotting_style()` to push the changes into matplotlib's rcParams.

::: synhydro.plotting.apply_plotting_style

---

## Helpers

::: synhydro.plotting.warn_if_many_realizations

::: synhydro.plotting.warn_if_few_realizations
