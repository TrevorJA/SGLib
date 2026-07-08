# Valencia-Schaake Temporal Disaggregation (Valencia and Schaake, 1973)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual to any equal-month-stride split: 12 months, 6 two-month periods, 4 quarters, 3 four-month seasons, or 2 half-years (`n_subperiods` in {12, 6, 4, 3, 2}) |
| **Sites** | Multisite (joint formulation per Valencia and Schaake, 1973) |

## Overview

The Valencia-Schaake method is the foundational parametric temporal
disaggregation approach in stochastic hydrology. Given an aggregate flow
vector (e.g., per-site annual totals), it generates a vector of sub-period
flows (e.g., monthly flows at every site) by fitting a joint multivariate
linear regression model. The full cross-site, cross-sub-period covariance
structure is preserved, and per-site aggregate totals are reproduced
exactly by construction in the untransformed model.

The method is the classical baseline against which subsequent disaggregation
methods are compared.

## Notation

The symbols below match the manuscript convention (Valencia and Schaake,
1973).

| Symbol | Description |
|--------|-------------|
| $m$ | Number of sites |
| $s$ | Number of sub-periods per aggregate (e.g., 12 months, 4 quarters) |
| $N$ | Number of complete years in the historical record |
| $\mathbf{Y} \in \mathbb{R}^{s m}$ | Sub-period vector for one year, ordered site-major (paper Eq. 3): all $s$ sub-period values for site 1, then site 2, and so on |
| $\mathbf{X} \in \mathbb{R}^{m}$ | Aggregate vector for one year (per-site annual totals; paper Eq. 4) |
| $\boldsymbol{\mu}_Y,\,\boldsymbol{\mu}_X$ | Sample mean vectors of $\mathbf{Y},\,\mathbf{X}$ |
| $\mathbf{S}_{yy},\,\mathbf{S}_{xx},\,\mathbf{S}_{yx}$ | Sample covariance and cross-covariance matrices (paper Eq. 13) |
| $\mathbf{A} \in \mathbb{R}^{s m \times m}$ | Regression matrix (paper Eq. 15) |
| $\mathbf{B}$ | Noise factor with $\mathbf{B}\mathbf{B}^\top = \mathbf{S}_{yy} - \mathbf{S}_{yx}\mathbf{S}_{xx}^{-1}\mathbf{S}_{xy}$ (paper Eq. 19) |
| $\mathbf{V} \sim \mathcal{N}(\mathbf{0},\,\mathbf{I})$ | Standard-normal innovations (paper Eq. 7) |
| $\mathbf{C} \in \mathbb{R}^{m \times s m}$ | Aggregation operator: $\mathbf{X} = \mathbf{C}\mathbf{Y}$ by definition (paper Eqs. 5-6) -- one row per site, ones across that site's sub-period block |

## Formulation

### Model Structure

The sub-period vector $\mathbf{Y}$ is modeled as a linear function of the
aggregate $\mathbf{X}$ plus Gaussian noise (paper Eqs. 1 and 7,
mean-incorporated form):

$$
\mathbf{Y} \;=\; \boldsymbol{\mu}_Y \,+\, \mathbf{A}(\mathbf{X} - \boldsymbol{\mu}_X) \,+\, \mathbf{B}\mathbf{V}.
$$

Equivalently, the conditional distribution of $\mathbf{Y}$ given $\mathbf{X}$
is multivariate normal,

$$
\mathbf{Y} \mid \mathbf{X} \;\sim\; \mathcal{N}\!\left(\boldsymbol{\mu}_Y + \mathbf{A}(\mathbf{X} - \boldsymbol{\mu}_X),\; \mathbf{B}\mathbf{B}^\top\right).
$$

The paper writes the centered form $\mathbf{Y} = \mathbf{A}\mathbf{X} +
\mathbf{B}\mathbf{V}$ and notes that "all random variables have been
transformed to have zero mean" and that rendering the data Gaussian is
"convenient but not absolutely necessary." The implementation works in
mean-incorporated coordinates (subtracting and adding back sample means),
which is algebraically equivalent.

### Parameter Estimation

Sample moments are estimated from $N$ historical years. Following paper
Eqs. 14, 15, 19:

$$
\mathbf{A} \;=\; \mathbf{S}_{yx}\mathbf{S}_{xx}^{-1},
\qquad
\mathbf{B}\mathbf{B}^\top \;=\; \mathbf{S}_{yy} - \mathbf{S}_{yx}\mathbf{S}_{xx}^{-1}\mathbf{S}_{xy}.
$$

In practice we use the Moore-Penrose pseudo-inverse for $\mathbf{S}_{xx}^{-1}$
to remain numerically stable when the aggregate covariance is poorly
conditioned, and we factor $\mathbf{B}\mathbf{B}^\top$ via a rank-aware
spectral decomposition (see Numerical considerations below). The paper uses
the biased estimator $\hat{\mathbf{S}} = (1/r)\,\mathbf{R}\mathbf{R}^\top$
(paper Eq. 25). The implementation uses the unbiased estimator
$\hat{\mathbf{S}} = (1/(r-1))\,\mathbf{R}\mathbf{R}^\top$; the
$1/(r-1)$ factor cancels in both
$\mathbf{A} = \mathbf{S}_{yx}\mathbf{S}_{xx}^{-1}$ and
$\mathbf{B}\mathbf{B}^\top$, so the parameter estimates are identical.

An optional transformation may be applied to sub-period flows before
fitting (`transform` in {'log', 'boxcox', 'none'}; the implementation
default is 'log'). The paper does not prescribe a specific transformation
-- these are practical extensions for highly skewed hydrologic flows. Note
that with the default log transform, per-site additivity is maintained by
the proportional rescale described below rather than by the exact linear
identity; pass `transform='none'` to run the pure linear model of the
paper.

### Exact additivity by construction

A key result of Valencia and Schaake (1973) is that the linear model
preserves the per-site aggregate identity $\mathbf{X} = \mathbf{C}\mathbf{Y}$
*exactly* for every draw, with no post-hoc rescaling. The proof (paper
Eqs. 31-42) proceeds in three steps:

1. **Historical identity (Eq. 32):** the training data satisfy
   $\mathbf{X} = \mathbf{C}\mathbf{Y}$ by construction (annual totals are
   sums of sub-periods).

2. **Inherited covariance identities (Eqs. 33-35):**

   $$
   \mathbf{S}_{xx} = \mathbf{C}\mathbf{S}_{yy}\mathbf{C}^\top, \quad
   \mathbf{S}_{yx} = \mathbf{S}_{yy}\mathbf{C}^\top, \quad
   \mathbf{S}_{xy} = \mathbf{C}\mathbf{S}_{yy}.
   $$

3. **Parameter identities (Eqs. 36-40):** substituting into the formulas
   for $\mathbf{A}$ and $\mathbf{B}\mathbf{B}^\top$ gives

   $$
   \mathbf{A} = \mathbf{S}_{yy}\mathbf{C}^\top\left(\mathbf{C}\mathbf{S}_{yy}\mathbf{C}^\top\right)^{-1},
   $$

   $$
   \mathbf{B}\mathbf{B}^\top = \mathbf{S}_{yy} - \mathbf{S}_{yy}\mathbf{C}^\top\left(\mathbf{C}\mathbf{S}_{yy}\mathbf{C}^\top\right)^{-1}\mathbf{C}\mathbf{S}_{yy}.
   $$

   Premultiplying by $\mathbf{C}$ yields

   $$
   \mathbf{C}\mathbf{A} = \mathbf{I}, \qquad
   \mathbf{C}\mathbf{B}\mathbf{B}^\top = \mathbf{0}.
   $$

   The second identity implies $\mathbf{C}\mathbf{B} = \mathbf{0}$
   because $(\mathbf{C}\mathbf{B})(\mathbf{C}\mathbf{B})^\top = \mathbf{0}$.

Combining (paper Eq. 42),

$$
\mathbf{C}\mathbf{Y} \;=\; \mathbf{C}\boldsymbol{\mu}_Y + \mathbf{C}\mathbf{A}(\mathbf{X} - \boldsymbol{\mu}_X) + \mathbf{C}\mathbf{B}\mathbf{V} \;=\; \boldsymbol{\mu}_X + (\mathbf{X} - \boldsymbol{\mu}_X) + \mathbf{0} \;=\; \mathbf{X}.
$$

**Scope of the identity.** This proof assumes $\mathbf{Y}$ in the fit is on
the *same scale* as $\mathbf{X}$, so the historical identity
$\mathbf{X} = \mathbf{C}\mathbf{Y}$ holds. That is true with
`transform='none'`. When a log or Box-Cox transform is applied,
$\mathbf{Y}$ is fit in transformed space while $\mathbf{X}$ remains on the
original scale, so Eqs. 33-35 no longer hold and the synthesized
sub-period flows do not automatically sum to the input annual totals.
Per-site additivity is then restored numerically by the proportional
rescale in step 4 of the synthesis procedure.

### Synthesis Procedure

For each year $t$ in the input ensemble with aggregate vector
$\mathbf{X}_t$:

1. Compute the conditional mean
   $\boldsymbol{\mu}_{Y \mid X} = \boldsymbol{\mu}_Y + \mathbf{A}(\mathbf{X}_t - \boldsymbol{\mu}_X)$.

2. Draw $\mathbf{V} \sim \mathcal{N}(\mathbf{0},\,\mathbf{I}_r)$, where
   $r$ is the retained rank of $\mathbf{B}$, and set
   $\mathbf{Y}_t = \boldsymbol{\mu}_{Y \mid X} + \mathbf{B}\mathbf{V}$.

3. If a transformation was applied during fitting, invert it. Non-finite
   values from inverse Box-Cox (out-of-domain draws) are mapped to zero
   before any further processing.

4. If `conservation_method='proportional'` (the default), clip any negative entries to
   zero and rescale each site's sub-period block to its target annual sum
   $X_{t,s}$. When the post-clip sum is zero (rare, requires extreme
   clipping), the site's $X_{t,s}$ is split uniformly across its
   sub-periods. With `transform='none'` and a full-rank $\mathbf{B}$, this
   step is a no-op because the linear model already satisfies
   $\mathbf{C}\mathbf{Y} = \mathbf{X}$ exactly; it becomes meaningful only
   when a nonlinear transform or aggressive clipping breaks linear
   additivity. The constructor rejects the combination
   `transform != 'none'` with `conservation_method='none'` because the
   resulting output would be silently non-conservative.

## Statistical Properties

The method preserves, to within sampling error:

- Sub-period means, standard deviations, and the full joint covariance of
  $\mathbf{Y}$.
- Cross-site, cross-sub-period correlations.
- Per-site aggregate totals -- exactly, by construction, in the
  untransformed full-rank case (paper Eq. 31); exactly after proportional
  adjustment in the transformed case.

Inter-annual serial correlation between sub-periods of consecutive years
is not modeled; see Stedinger and Vogel (1984) for an extension that
addresses this.

## Numerical considerations

- $\mathbf{B}\mathbf{B}^\top$ is rank-deficient by construction with rank
  at most $\min(N - 1,\,(s - 1)\,m)$. The paper's bound (page 5) is
  $N - 1$; the tighter $(s - 1)\,m$ ceiling comes from the null-space
  constraint $\mathbf{C}\mathbf{B} = \mathbf{0}$.
- $\mathbf{B}$ is constructed by eigendecomposing
  $\mathbf{B}\mathbf{B}^\top$ in an orthonormal basis of the null space of
  $\mathbf{C}$ and retaining only eigenvalues above a relative tolerance.
  This matches the paper's "principal component technique" (page 5) of
  retaining the leading $N - 1$ eigenvectors, with the additional null-
  space restriction that guarantees $\mathbf{C}\mathbf{B} = \mathbf{0}$ to
  floating-point precision -- preserving exact additivity for every draw.
- When the residual covariance is numerically zero (e.g., a perfectly
  cyclic input), the noise factor takes rank 0 and the disaggregation
  becomes deterministic, equal to the conditional mean.
- Box-Cox uses a single shared $\lambda$ fit on flattened data across all
  sites and sub-periods. This is a practical simplification; per-site
  lambdas would be more appropriate for basins with heterogeneous flow
  magnitudes.

## Limitations

- The multivariate normality assumption may be violated for strongly skewed
  sub-period distributions. A log or Box-Cox transform mitigates this but
  breaks the linear-additivity property; proportional rescaling restores
  per-site conservation at the cost of small distortion of the conditional
  covariance.
- Full joint covariance estimation requires reasonably long records
  relative to $s \cdot m$. Short records produce a heavily rank-deficient
  noise factor.
- Inter-annual serial correlation between consecutive sub-periods is not
  modeled.
- The model is best suited for annual-to-monthly or annual-to-seasonal
  disaggregation. Monthly-to-daily is generally not recommended with this
  method.

## References

**Primary:**
Valencia, R.D., and Schaake, J.C. (1973). Disaggregation processes in stochastic hydrology. *Water Resources Research*, 9(3), 580-585. <https://doi.org/10.1029/WR009i003p00580>

**See also:**

- Grygier, J.C., and Stedinger, J.R. (1988). Condensed disaggregation procedures and conservation corrections for stochastic hydrology. *Water Resources Research*, 24(10), 1574-1584. <https://doi.org/10.1029/WR024i010p01574>
- Stedinger, J.R., and Vogel, R.M. (1984). Disaggregation procedures for generating serially correlated flow vectors. *Water Resources Research*, 20(1), 47-56. <https://doi.org/10.1029/WR020i001p00047>
- Salas, J.D., Delleur, J.W., Yevjevich, V., and Lane, W.L. (1980). *Applied Modeling of Hydrologic Time Series*. Water Resources Publications.

---

**Implementation:** `src/synhydro/methods/disaggregation/temporal/valencia_schaake.py`
