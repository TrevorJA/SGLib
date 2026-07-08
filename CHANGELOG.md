# Changelog

All notable changes to SynHydro are documented in this file.

## [Unreleased]

### Added
- `NowakDisaggregator` generalized to arbitrary discrete timescale pairs via
  `input_timestep` / `output_timestep` constructor arguments: any input in
  {annual, monthly, weekly} disaggregates to any finer output in
  {monthly, weekly, daily} (six pairs). Weekly timesteps follow the ISO
  52-week convention (Sunday anchors, week 53 folded into week 52 pools),
  consistent with `KirschGenerator`. For monthly-to-weekly, weeks are
  assigned to the calendar month containing their Sunday anchor. Monthly to
  daily remains the default and its output is bit-identical to the previous
  implementation for a given seed (guarded by a golden-file regression test
  in `tests/test_nowak_regression.py`).
- `KirschGenerator` now supports weekly resolution in addition to monthly.
  `preprocessing(..., timestep="weekly")` aggregates daily input to ISO weekly
  (W-SUN) buckets, drops ISO week 53, and runs the Kirsch (2013) algorithm with
  52 periods per year and a 26-period half-shift. Monthly remains the default
  and is auto-detected when no `timestep` is supplied. Pre-aggregated weekly
  input (W-SUN family aliases) is also accepted.
- ARFIMA MA(q) component with CSS estimation and BIC-based order selection
- KNN Bootstrap generator for nonparametric multi-site generation
- Valencia-Schaake temporal disaggregation method
- Validation framework with 12 metric categories:
  - L-moment ratios, GEV extreme-value metrics
  - CRPS and SSI drought metrics
  - Bootstrap hypothesis testing module
- Month blending for Nowak disaggregation
- Input validation with warnings for poor data quality
- Pre-commit lint hook
- CONTRIBUTING.md
- MkDocs documentation site with algorithm reference pages

### Changed
- Breaking: `NowakDisaggregator` constructor arguments renamed for
  timescale-neutral clarity: `max_month_shift` is now
  `max_knn_pool_shift_timesteps` and `blend_days` is now
  `boundary_blend_timesteps`, both in units of output timesteps.
  Per-pair defaults apply when `max_knn_pool_shift_timesteps` is omitted
  (7 for monthly-to-daily, unchanged from before). The `KirschNowakPipeline`
  and `ThomasFieringNowakPipeline` public `max_month_shift` argument is
  unchanged. The legacy `disaggregate_monthly_flows()` method has been
  removed; use `disaggregate(ensemble, seed=...)`.
- `NowakDisaggregator` boundary smoothing now defaults to off
  (`boundary_blend_timesteps=0`, previously 2) so the default reproduces the
  published Nowak et al. (2010) method, which applies no boundary correction.
  Boundary smoothing remains available as an opt-in SynHydro extension.
- `KNNBootstrapGenerator` no longer accepts sub-monthly input. Restricted to
  monthly (Lall & Sharma, 1996; Prairie et al., 2006) and annual (Prairie et
  al., 2008) per the primary streamflow literature. Sub-monthly input now
  raises `ValueError`; for daily output, generate an annual realization and
  disaggregate with `NowakDisaggregator`. Daily KNN bootstrap in the
  literature is established only for weather variables (Rajagopalan and Lall,
  1999) or as a disaggregation step (Nowak et al., 2010), not as a standalone
  daily streamflow generator.
- Updated pandas frequency strings to current preferred names: `'AS'` ->
  `'YS'`, `'A'` -> `'YE'`, `'M'` -> `'ME'`. Input-alias sets in ARFIMA,
  WARM, and SMARTA still accept both old and new aliases for compatibility
  with `pd.infer_freq()` on older pandas versions.
- Reorganized generators into three classification bins based on the
  mathematical character of their generative mechanism (Studnicka and Panu,
  2025): `synhydro.methods.generation.parametric` (Thomas-Fiering, Matalas,
  ARFIMA, SPARTA, SMARTA, MS-HMM), `synhydro.methods.generation.hybrid`
  (Kirsch, WARM, Phase Randomization, MS Phase Randomization, HMM-KNN), and
  `synhydro.methods.generation.nonparametric` (KNN-Bootstrap). Top-level
  `synhydro.<Generator>` imports are unchanged.
- **Breaking:** `KirschGenerator` internal attribute renames to be
  resolution-agnostic: `n_months` -> `n_periods_per_year`,
  `mean_month` -> `mean_period`, `std_month` -> `std_period`, and the public
  property `Q_obs_monthly` -> `Q_obs_aggregated`. Downstream callers
  referencing these attributes (e.g., custom plotting code) must update.
  `MatalasGenerator` and `ThomasFieringGenerator` retain their original
  `Q_obs_monthly` naming.
- Migrated all generators to `np.random.Generator` (replaces legacy `np.random`)
- Replaced all `print()` with `logging.getLogger(__name__)`
- Major API refactor: standardized preprocessing/fit/generate interface
- Improved ensemble HDF5 loading performance
- Stedinger transform lower bound now correctly falls back to tau=0
  when the formula produces values outside the valid range
- Pinned minimum dependency versions in pyproject.toml
- Test fixtures produce physically realistic (non-negative) streamflow data
- SSI clip floor is now configurable via `cdf_epsilon` (keyword-only); the
  default changed from an implicit +/- 6.36 bound to +/- 4.5 to suppress
  gamma-tail extrapolation artifacts. Pre-existing values previously below
  -4.5 or above +4.5 will now be clipped to those bounds.
- Renamed package from SGLib to SynHydro

### Fixed
- Matalas correlation matrix repair via shared `repair_correlation_matrix`
- KNN Bootstrap `block_size` and `index_site` parameters
- Multisite `ValueError` for univariate generators
- Nowak non-leap-year February bug
- Kirsch correlation matrix bug
- Kirsch `generate()` and `generate_from_indices()` now build X' as the
  deterministic 6-month shift of X per Kirsch et al. (2013), p. 6, rather
  than an independent bootstrap (`generate()`) or a shared-index lookup
  (`generate_from_indices()`). All three entry points
  (`generate`/`generate_single_series`, `generate_from_indices`,
  `generate_from_residuals`) are now statistically equivalent and route
  through a shared `_pipeline_from_X` helper. Breaking for seed-level
  reproducibility: `.generate(seed=S)` produces different numerical output
  than prior releases (the corrected distribution).
- Kirsch weekly output DatetimeIndex is now built per synthetic year using
  `pd.Timestamp.fromisocalendar(y, w, 7)` instead of a plain
  `pd.date_range(freq="W-SUN")`. The previous range marched in 7-day steps
  across year boundaries, accumulating a ~1.25 day/year backward drift (since
  52 x 7 = 364 days), which manifested as a multi-week leftward shift in
  seasonal-cycle plots over multi-decade horizons. Each synthetic position
  `k` of year `y` now lands on the Sunday of ISO week `(k+1)` of `y` so the
  per-period mean/std applied at fit time aligns with the calendar week of
  the labeled date.
- SSI Python version compatibility issue
- SSI deprecated pandas `"M"` frequency string (now `"ME"`)
- Valencia-Schaake divide-by-zero in correlation matrix computation
- RuntimeWarnings from `np.log` on edge-case data in Stedinger transform
  and HMM preprocessing

### Removed
- Deprecated Kirsch-Nowak combined generator
- Outdated `core/validation.py` monolith (replaced by `core/validation/` package)

## [0.0.2] - 2025-06-09

### Added
- Thomas-Fiering monthly generator
- Kirsch nonparametric generator
- Nowak temporal disaggregation
- Multisite HMM generator
- WARM wavelet-based generator
- Phase randomization generator
- Matalas multivariate autoregressive generator
- SSI drought index calculation
- Ensemble management with HDF5 storage
- Pipeline for chaining generation and disaggregation
- Basic package structure with `pip install` support

## [0.0.1] - 2023-06-15

- Initial commit with project scaffolding
