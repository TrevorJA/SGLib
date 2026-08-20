# Nowak KNN Temporal Disaggregation (Nowak et al., 2010)

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | {Annual, Monthly, Weekly} to {Monthly, Weekly, Daily} |
| **Sites** | Univariate / Multisite |

## Overview

The Nowak disaggregator converts synthetic coarse-timestep flows (annual, monthly, or weekly) to a finer timestep (monthly, weekly, or daily) by borrowing within-period flow patterns from the closest historical analogs. For each synthetic coarse period, the $K$ nearest historical periods (by total flow magnitude at an index gauge) are identified, one is selected stochastically using either inverse-distance weighting (default) or Lall-Sharma kernel weights, and its fine-timestep flow proportions are applied to the synthetic coarse total. The method preserves coarse-period totals by construction and maintains realistic fine-timestep flow dynamics drawn directly from the observed record.

Any pairing of input timestep in {annual, monthly, weekly} with a finer output timestep in {monthly, weekly, daily} is supported: annual-to-monthly, annual-to-weekly, annual-to-daily, monthly-to-weekly, monthly-to-daily (the default), and weekly-to-daily.

### Relation to Nowak et al. (2010)

The original Nowak et al. (2010) paper presents the method at annual-to-daily resolution: KNN selects one donor year, and its 365-day proportions are applied to the synthetic annual total at every site. The paper notes the algorithm "can be readily applied to any space and time scales" and demonstrates annual-to-monthly disaggregation as well. This implementation adds one extension for sub-annual input timesteps: candidate pools are conditioned on the calendar period (month of year, or ISO week of year), and each pool is enlarged by including time-shifted copies of the historical windows (see Analog Pool Construction). Operating at the monthly level yields a substantially larger pool of historical analogs (12 x N months vs. N years) and therefore better representation of within-pool sampling uncertainty. For annual input the pool is unconditioned, exactly as in the original paper.

Two further differences from the paper apply to every timescale pair and are easy to miss because they are defaults:

- **Neighbor selection defaults.** The paper uses the Lall and Sharma (1996) kernel (rank-based weights $1/i$) with the heuristic $K = \sqrt{N_p}$. SynHydro defaults to `sample_method="distance_weighted"` (inverse-distance weights) and `n_neighbors=5`, independent of pool size. To reproduce the paper's selection rule pass `sample_method="lall_and_sharma_1996"` to `disaggregate()` and set `n_neighbors=int(round(sqrt(N_p)))`, where $N_p = N_{\text{years}} (2s + 1)$ is the pool size for the label (for annual input with `max_knn_pool_shift_timesteps=0`, $N_p$ is simply the number of historical years).
- **Per-site temporal disaggregation instead of joint space-time disaggregation.** In the paper a single synthetic value at an index gauge is disaggregated both spatially (to the sites) and temporally (to days) in one step: the donor year's daily flows at every site are scaled by the ratio of the synthetic to the donor index total, so the spatial pattern comes entirely from the donor. SynHydro instead expects a synthetic coarse flow *per site* (e.g. from a multisite monthly generator) and applies the donor's *per-site* proportion vector, normalized to sum to one at each site, to that site's own coarse flow. The donor period is still selected once, on the index-gauge total (sum over sites), and shared by all sites, so the within-period timing is spatially coherent, but the coarse-period spatial distribution is taken from the synthetic input rather than from the donor. Spatial disaggregation from a single index series is not provided.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_p^{\text{syn}}$ | Synthetic total flow for coarse period $p$ |
| $q_t^{\text{syn}}$ | Synthetic fine-timestep flow at step $t$ |
| $q_t^*$ | Observed fine-timestep flow at step $t$ of the selected analog period |
| $K$ | Number of nearest neighbors |
| $w_i$ | Selection weight for the $i$-th closest neighbor |
| $d_i$ | Euclidean distance from the synthetic coarse flow to the $i$-th neighbor |
| $b$ | Number of blending timesteps at period boundaries |
| $s$ | Maximum pool window shift, in output timesteps |
| $N_p$ | Number of historical candidates in the pool for period label $p$ |

## Formulation

### Analog Pool Construction

For monthly or weekly input, a pool of candidate historical periods is assembled for each calendar period label (month of year 1-12, or ISO week of year 1-52). The pool is enlarged by shifting each historical window by up to $\pm s$ output timesteps (`max_knn_pool_shift_timesteps`), giving $N_p = N_{\text{years}} \cdot (2s + 1)$ candidates per label. Larger shifts enlarge the pool but rotate the sampled profiles relative to the calendar, so the default shift scales with the timescale pair (7 days for monthly-to-daily, 2 days for weekly-to-daily, 1 week for monthly-to-weekly, 0 for annual-to-monthly, 2 weeks for annual-to-weekly, 7 days for annual-to-daily).

For annual input there is a single pool containing every historical year (plus shifted copies when $s > 0$), matching the original paper.

Each pool entry stores the total coarse flow at the index gauge (the sum across sites for multisite data) and the corresponding fine-timestep flow proportion vector for each site.

### Neighbor Selection

For each synthetic coarse flow $Q_p^{\text{syn}}$, the $K$ nearest historical candidates are found by Euclidean distance on total coarse flow at the index gauge. One neighbor is then drawn stochastically from the $K$ candidates using one of two weighting schemes.

**Inverse-distance weighting** (default). The selection probability for the $i$-th neighbor is proportional to the inverse of its distance:

$$
w_i = \frac{1/d_i}{\displaystyle\sum_{j=1}^{K} 1/d_j}
$$

where $d_i$ is the Euclidean distance between $Q_p^{\text{syn}}$ and the $i$-th neighbor's coarse total (a small epsilon is added to $d_i$ to guard against an exact match). This gives stronger preference to closer analogs when the distance differences are large, but approaches uniform selection when all neighbors are similarly distant. This scheme and the default $K = 5$ are SynHydro choices, not the paper's (see Relation to Nowak et al. (2010)).

**Lall-Sharma kernel** (Lall and Sharma, 1996). The selection probability depends only on rank, not distance:

$$
w_i = \frac{1/i}{\displaystyle\sum_{j=1}^{K} 1/j}, \qquad i = 1, \ldots, K
$$

This harmonic weighting gives the closest neighbor approximately twice the probability of the second-closest, regardless of the actual distance magnitudes. This is the kernel used by Nowak et al. (2010), who pair it with $K = \sqrt{N_p}$; SynHydro does not set $K$ automatically, so pass `n_neighbors` explicitly to match.

### Proportional Disaggregation

The fine-timestep flows of the selected analog period are used as a template. Let $\{q_t^*\}_{t=1}^{T}$ denote the observed fine flows in the selected analog period (with $T$ steps). The synthetic fine flows are computed by proportional scaling:

$$
q_t^{\text{syn}} = Q_p^{\text{syn}} \cdot \frac{q_t^*}{\displaystyle\sum_{t'=1}^{T} q_{t'}^*}
$$

This ensures that the synthetic fine flows sum to the synthetic coarse total. For multisite data, each site is disaggregated with its own proportion vector (normalized per site) using the same selected analog period, preserving inter-site consistency of the within-period timing.

If a pool entry has zero (or all-missing) total flow at a site, its proportion vector is set to the uniform profile $1/T$ for that site so that disaggregation remains defined.

**Period-length mismatch.** When the analog and target periods differ in length (leap-year February or leap years at daily output, four- versus five-Sunday months at weekly output) the sampled proportion vector is adjusted so it still sums to one:

- *Longer analog, shorter target* (e.g. a 29-day February profile applied to a 28-day February, or a 366-day year applied to a 365-day year): the profile is truncated to the target length and renormalized to sum to one. This applies to every timescale pair, including monthly-to-daily, so coarse totals are conserved exactly.
- *Shorter analog, longer target* (28-day profile applied to a 29-day February, 365-day profile applied to a 366-day year, four-Sunday profile applied to a five-Sunday month): the trailing (missing) step is filled with the average proportion $1/T$ of the target period and the remaining proportions are scaled by $(1 - 1/T)$. The final step of the target period therefore receives the period-mean flow rather than an observed value, which is visible as a small discontinuity on 29 February or 31 December of leap years.

*Zero-flow caveat.* The length fix is triggered by zero entries in the proportion vector, so a genuine zero-flow day inside an analog profile is indistinguishable from the missing trailing step. When a leap-length target draws an analog that contains observed zeros, the code falls back to spreading the missing mass uniformly over the non-zero steps and leaves the zero steps (including the trailing one) at zero. Totals are still conserved, but intermittent (ephemeral) records will see the last step of leap periods set to zero rather than $1/T$.

### Timescale Conventions

- Annual input must be calendar-year, i.e. January-anchored (`YS`). Water-year or otherwise anchored annual frequencies (e.g. `YS-OCT`) are rejected with a `ValueError` at disaggregation time.
- Weekly timesteps use ISO weeks anchored on Sundays (`W-SUN`). Years are treated as exactly 52 ISO weeks: ISO week 53 is folded into the week 52 pool on input and never generated on output, consistent with `KirschGenerator`.
- Weeks do not nest inside calendar months. For monthly-to-weekly disaggregation, each week is assigned to the calendar month containing its Sunday anchor; disaggregated weekly flows sum to the synthetic monthly flow over the weeks assigned to that month.
- Weekly output indices are anchored per year via the ISO calendar (`Timestamp.fromisocalendar`), avoiding calendar drift across long simulations.

### Period Boundary Smoothing

Boundary smoothing is a SynHydro extension beyond the published Nowak et al. (2010) method and is **disabled by default** (`boundary_blend_timesteps=0`), so the default behavior reproduces the published algorithm. When enabled with a positive value, a blending step applies a centered rolling mean across $b$ output timesteps on each side of each coarse-period boundary to reduce discontinuities from independent per-period sampling. After smoothing, each period is rescaled to restore the original period total:

$$
q_t^{\text{smoothed}} \leftarrow q_t^{\text{smoothed}} \cdot \frac{Q_p^{\text{syn}}}{\displaystyle\sum_{t'} q_{t'}^{\text{smoothed}}}
$$

### Synthesis Procedure

1. Fit a KNN model on the historical coarse-flow totals for each period label. Shifted windows that extend past the ends of the observed record are filled by wrapping around to the opposite end of the record.
2. For each synthetic coarse flow $Q_p^{\text{syn}}$:
   - Query the $K$ nearest neighbors by total flow at the index gauge.
   - Select one analog period using inverse-distance or Lall-Sharma kernel weights.
   - Disaggregate by applying the analog's fine-timestep proportions to $Q_p^{\text{syn}}$.
3. Optionally smooth period boundaries and rescale to preserve period totals.
4. Enforce non-negativity.

## Statistical Properties

Coarse-period totals are preserved by construction for every timescale pair, including truncated leap-February and leap-year profiles. Fine-timestep flow patterns within each period are drawn from the historical record, maintaining realistic intra-period dynamics including storm hydrographs and recession curves. Multisite consistency within each period is preserved through joint analog selection.

Period-to-period fine-timestep transitions are not explicitly modeled, though the optional boundary blending partially addresses this. The method cannot produce fine-timestep patterns not observed in the historical record, limiting its ability to represent unprecedented extremes. Fine-timestep autocorrelation across period boundaries depends on the quality of analog matching.

## Limitations

- Cannot produce fine-timestep flow patterns outside the historical range.
- Period-length mismatches (leap years, four- versus five-Sunday months) require proportional adjustment when the analog and target periods differ in length; a shorter analog is extended by filling the trailing step with the period-mean proportion $1/T$ (or, if the analog contains zero-flow steps, by leaving it at zero), see Proportional Disaggregation.
- The defaults (`sample_method="distance_weighted"`, `n_neighbors=5`) differ from the paper's Lall-Sharma kernel with $K = \sqrt{N_p}$; pass those options explicitly to reproduce the published selection rule.
- Spatial disaggregation from a single index-gauge series is not implemented; each site must be supplied with its own synthetic coarse flow and only the within-period timing is borrowed from the donor.
- Quality depends on having a sufficiently long historical record to find good analogs across the range of synthetic coarse totals. This is most acute for annual input, where the pool contains one candidate per historical year.
- Nonzero pool window shifts rotate sampled profiles relative to the calendar; large shifts trade seasonal fidelity for pool size.
- Period-to-period transitions may exhibit discontinuities despite blending.

## References

**Primary:**
Nowak, K., Prairie, J., Rajagopalan, B., and Lall, U. (2010). A nonparametric stochastic approach for multisite disaggregation of annual to daily streamflow. *Water Resources Research*, 46(8). https://doi.org/10.1029/2009WR008530

**See also:**
- Lall, U., and Sharma, A. (1996). A nearest neighbor bootstrap for resampling hydrologic time series. *Water Resources Research*, 32(3), 679-693.

---

**Implementation:** `src/synhydro/methods/disaggregation/temporal/nowak.py`
