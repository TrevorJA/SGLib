# K-Nearest Neighbor Bootstrap (Lall and Sharma, 1996)

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | Monthly / Annual |
| **Sites** | Univariate / Multisite |

## Overview

The K-Nearest Neighbor (KNN) bootstrap generates synthetic streamflow by conditionally resampling from the historical record. At each time step, the most recent generated value defines a query point in feature space; the $K$ closest historical states are identified, and one is selected with probability inversely proportional to its rank (Lall-Sharma kernel). The generated value is the historical successor of the selected neighbor. This nonparametric approach preserves the empirical marginal distribution exactly and can capture nonlinear dependence structures that parametric models may miss.

This implementation supports **monthly** (Lall and Sharma, 1996) and **annual** streamflow only. The annual mode is the Lall-Sharma bootstrap applied to annual flows; Prairie et al. (2008) establish annual KNN for streamflow but their paleo-state conditioning is not implemented (see Deviations below). Daily KNN bootstrap is not established for streamflow synthesis in the primary literature: Rajagopalan and Lall (1999) apply daily KNN to weather variables (precipitation, temperature, wind), not streamflow, and Nowak et al. (2010) use KNN at the daily timescale only as a disaggregation step from annual flows. For daily streamflow output, generate an annual realization and disaggregate with `NowakDisaggregator`.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_t \in \mathbb{R}^S$ | Observed flow vector at time $t$ across $S$ sites |
| $\hat{Q}_t$ | Synthetic flow vector at time $t$ |
| $K$ | Number of nearest neighbors |
| $N$ | Number of historical time steps |
| $w_i$ | Lall-Sharma kernel weight for the $i$-th closest neighbor |
| $\mathbf{x}_t$ | Feature vector at time $t$ (flow values at selected sites) |
| $d(\cdot, \cdot)$ | Euclidean distance in feature space |

## Formulation

### Neighbor Selection and Kernel Weights

The number of neighbors defaults to $K = \lceil \sqrt{n} \rceil$, where $n$ is the size of the sample that is searched (Lall and Sharma, 1996, give $K = n^{1/2}$; rounding up with the ceiling is an implementation choice). For annual data this is the full record, $n = N - 1$ feature-successor pairs. For monthly data the pools are partitioned by calendar month, so each month $m$ gets its own $K_m = \lceil \sqrt{n_m} \rceil$ with $n_m \approx N/12$ the number of pairs in that month's pool. A user-supplied `n_neighbors` overrides the heuristic and is applied to every pool (clipped to $n_m - 1$). The Lall-Sharma kernel assigns probability to the $i$-th closest neighbor ($i = 1, 2, \ldots, K$) as:

$$
w_i = \frac{1/i}{\displaystyle\sum_{j=1}^{K} 1/j}
$$

This harmonic weighting gives the closest neighbor exactly twice the selection probability of the second-closest ($w_1 / w_2 = 2$ by construction), encouraging fidelity to the local neighborhood while maintaining stochastic diversity.

### Feature-Successor Structure

For each historical time step $t$, a feature-successor pair is stored:

$$
(\mathbf{x}_t,\; Q_{t+1})
$$

where $\mathbf{x}_t$ is the feature vector (flow values at time $t$ for the selected sites) and $Q_{t+1}$ is the observed flow at the next time step. For monthly data, the pairs are partitioned by calendar month to respect seasonality: at generation time, only neighbors from the same calendar month are considered.

### Multisite Extension

In multisite mode, the neighbor search is performed on a single index site or on the full multivariate feature vector using Euclidean distance:

$$
d(\mathbf{x}, \mathbf{x}') = \left\|\mathbf{x} - \mathbf{x}'\right\|_2
$$

Once a neighbor is selected, the successor vector across all $S$ sites is taken jointly, preserving spatial correlation by construction.

### Synthesis Procedure

1. Select a random historical time step as the initial state and set $\hat{Q}_1$ equal to the successor of that state. For monthly data the initial state is drawn from the pool whose successors fall in the calendar month of $\hat{Q}_1$ (the month preceding the first synthetic month), so the first value is an observed flow of the correct month.
2. For each subsequent time step $t = 2, 3, \ldots, T$:
   - Form the query feature vector $\hat{\mathbf{x}}_{t-1}$ from the most recently generated flow values.
   - Find the $K$ nearest neighbors of $\hat{\mathbf{x}}_{t-1}$ among the historical feature vectors (within the same calendar month if monthly).
   - Select one neighbor $j^*$ with probability $w_i$ based on its rank $i$.
   - Set the generated value to the historical successor:

$$
\hat{Q}_t = Q_{j^* + 1}
$$

3. For multisite data, the entire successor vector is assigned jointly.

## Statistical Properties

The empirical marginal distribution is preserved exactly, since every generated value is drawn directly from the historical record. Nonlinear dependence is captured implicitly through the conditional neighborhood structure. Lag-1 autocorrelation is approximately preserved because the successor of a similar state will tend to exhibit similar temporal dynamics. Spatial cross-correlations are maintained through joint resampling.

However, generated values cannot exceed the historical range (a fundamental bootstrap limitation). Long-range persistence beyond the conditioning lag is not explicitly modeled, and the method does not capture trends or nonstationarity. The curse of dimensionality can degrade neighbor selection quality when many sites are used simultaneously as features.

## Limitations

- Cannot generate values outside the range of the historical record.
- Sensitive to $K$: too small leads to repetitive cycling through a few neighbors; too large destroys local temporal structure.
- Curse of dimensionality for high-dimensional feature spaces (many sites).
- Requires sufficient record length (roughly 20+ years for monthly data) to avoid excessive repetition of analogs.

## Deviations from the Reference Papers

The implementation follows Lall and Sharma (1996) in its core mechanics (month-partitioned pools, successor resampling, harmonic kernel weights, exclusion of the final observation from the feature set). The following points differ from, or simplify, the published methods:

- **Unscaled Euclidean distance.** Lall and Sharma (1996) define the distance with optional feature weights $w_j$ and suggest $w_j = 1/s_j$ (inverse standard deviation) when the feature components are on different scales. This implementation uses plain Euclidean distance with unit weights. With `index_site` (a single feature) this is immaterial; with several `feature_cols` on different scales the largest-magnitude site dominates the neighbor search. Standardize the inputs beforehand if that matters.
- **Lag-1 features only.** The feature vector is the flow at the previous time step. The papers allow a general lag-$p$ (or other) state vector; higher-order conditioning is not supported.
- **No tie permutation.** Lall and Sharma (1996) randomly permute tied distances before ranking. Ties are resolved deterministically by the scikit-learn neighbor search order. Ties are rare for continuous flow data but occur for repeated values (e.g. zero flows).
- **No GCV selection of $K$.** The paper's generalized cross-validation criterion for choosing $K$ is not implemented. $K$ is either user-supplied or the $\lceil \sqrt{n} \rceil$ heuristic described above.
- **`block_size` extension.** Copying `block_size` consecutive successors per neighbor draw is a SynHydro extension not described in the papers. With `block_size > 1`, the monthly pool is only re-queried every `block_size` steps, and the month key for the query is the month of the last generated value.
- **Prairie et al. (2006) modified KNN not implemented.** Prairie et al. (2006) modified KNN (local-polynomial conditional mean + residual resampling) is not implemented; generated values are always historical values, so values outside the observed range cannot be produced.
- **Annual mode is plain Lall-Sharma.** For annual input a single global pool is used. Prairie et al. (2008) condition the annual KNN on a paleo-reconstructed hydrologic state (wet/dry classification from tree-ring records) and combine observational and paleo data; neither the state conditioning nor the paleo blending is implemented here. The annual path is Lall and Sharma's (1996) bootstrap applied to annual totals.

## References

**Primary (streamflow generator):**
- Lall, U., and Sharma, A. (1996). A nearest neighbor bootstrap for resampling hydrologic time series. *Water Resources Research*, 32(3), 679-693. https://doi.org/10.1029/95WR02966 -- introduces the method, applied to monthly streamflow on the Weber River, Utah.
- Prairie, J., Nowak, K., Rajagopalan, B., Lall, U., and Fulp, T. (2008). A stochastic nonparametric approach for streamflow generation combining observational and paleoreconstructed data. *Water Resources Research*, 44, W06423. https://doi.org/10.1029/2007WR006684 -- annual streamflow at Lees Ferry; motivates the annual timescale here, but its paleo-state conditioning is not implemented (see Deviations).

**Related (not the basis for this generator):**
- Prairie, J., Rajagopalan, B., Fulp, T., and Zagona, E. (2006). Modified K-NN model for stochastic streamflow simulation. *Journal of Hydrologic Engineering*, 11(4), 371-378. https://doi.org/10.1061/(ASCE)1084-0699(2006)11:4(371) -- local-polynomial conditional mean plus residual resampling for monthly streamflow at Lees Ferry; the modification is not implemented (see Deviations).
- Rajagopalan, B., and Lall, U. (1999). A k-nearest-neighbor simulator for daily precipitation and other weather variables. *Water Resources Research*, 35(10), 3089-3101. https://doi.org/10.1029/1999WR900028 -- daily KNN bootstrap applied to weather variables, not streamflow.
- Nowak, K., Prairie, J., Rajagopalan, B., and Lall, U. (2010). A nonparametric stochastic approach for multisite disaggregation of annual to daily streamflow. *Water Resources Research*, 46, W08529. https://doi.org/10.1029/2009WR008530 -- daily KNN used as a disaggregation step on annual realizations; see `NowakDisaggregator`.
- Lall, U. (1995). Recent advances in nonparametric function estimation: Hydrologic applications. *Reviews of Geophysics*, 33(S2), 1093-1102.

---

**Implementation:** `src/synhydro/methods/generation/nonparametric/knn_bootstrap.py`
