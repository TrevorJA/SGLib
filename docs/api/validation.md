# Validation

The `synhydro.validation` module evaluates fit for purpose: whether a
synthetic ensemble reproduces characteristics that the generator was
not explicitly fit to, currently drought duration, severity, and
frequency.

The terminology follows Stedinger and Taylor (1982). *Validation*
demonstrates that characteristics not explicitly reproduced by
parameter estimation, such as drought and storage behavior, are
consistent with the historical record. For statistical property
preservation, see [Verification](verification.md).

Results are reported in the same form as the verification suite: each
statistic is computed once on the observed record and once per
realization, and the observed value is compared against the
distribution across realizations. See
[Verification](verification.md#how-results-are-reported) for the
summary columns.

Storage-yield and further run-theory statistics (sequent-peak storage,
reliability) are planned extensions of this suite.

## Metric selection

```python
import synhydro

result = synhydro.validate(ensemble, Q_obs, metrics="all")
result = synhydro.validate(ensemble, Q_obs, metrics=["threshold_drought"])
result.summary()
```

## Metric reference

| Metric | Category | Units | Description | Citation |
|--------|----------|-------|-------------|----------|
| `mean_drought_duration` | threshold_drought | timesteps | Mean below-threshold run length | Yevjevich (1967); Salas et al. (1980) |
| `mean_drought_severity` | threshold_drought | flow (cumulative deficit) | Mean cumulative deficit per event | Yevjevich (1967); Salas et al. (1980) |
| `max_drought_duration` | threshold_drought | timesteps | Longest below-threshold run | Yevjevich (1967); Salas et al. (1980) |
| `max_drought_severity` | threshold_drought | flow (cumulative deficit) | Largest cumulative deficit | Yevjevich (1967); Salas et al. (1980) |
| `drought_frequency` | threshold_drought | events per timestep | Drought events per timestep | Yevjevich (1967); Salas et al. (1980) |
| `ssi_mean_drought_duration` | ssi_drought | timesteps | Mean SSI drought event length | McKee et al. (1993) |
| `ssi_max_drought_duration` | ssi_drought | timesteps | Longest SSI drought event | McKee et al. (1993) |
| `ssi_mean_drought_severity` | ssi_drought | dimensionless | Mean cumulative absolute SSI per event | McKee et al. (1993) |
| `ssi_max_drought_severity` | ssi_drought | dimensionless | Largest cumulative absolute SSI | McKee et al. (1993) |
| `ssi_drought_frequency` | ssi_drought | events per year | SSI drought events per year | McKee et al. (1993) |

Notes:

- Threshold drought events are below-threshold runs: an event is a
  maximal sequence of consecutive timesteps with flow below the
  truncation level (by default the 20th percentile of observed flows
  at each site). Severity is the cumulative deficit, the sum of
  (threshold minus flow) over the event.
- SSI drought events start when the Standardized Streamflow Index
  drops to -1 or below and end when it returns to zero or above. The
  SSI is fit on the observed record and the same fitted distributions
  transform every realization, so synthetic droughts are measured on
  the observed climatology.

---

## Orchestrator and result

::: synhydro.validation.validate

::: synhydro.validation.ValidationResult

::: synhydro.validation.list_metrics
