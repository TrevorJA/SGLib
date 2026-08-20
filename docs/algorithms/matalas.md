# Matalas Multi-Site MAR(1) (Matalas, 1967)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly |
| **Sites** | Multisite |

## Overview

The Matalas model extends the univariate Thomas-Fiering seasonal AR(1) to multiple sites by fitting a matrix autoregressive process to standardized monthly flows. A separate pair of transition matrices (autoregressive coefficients and innovation structure) is estimated for each of the 12 calendar-month transitions, capturing both temporal persistence and contemporaneous spatial dependence across a network of gauges.

## Notation

| Symbol | Description |
|--------|-------------|
| $\mathbf{Q}_t \in \mathbb{R}^S$ | Observed monthly flow vector at time $t$ across $S$ sites |
| $\hat{\mathbf{Q}}_t$ | Synthetic monthly flow vector at time $t$ |
| $\mathbf{Z}_t \in \mathbb{R}^S$ | Standardized flow vector at time $t$ |
| $m(t)$ | Calendar month corresponding to time $t$, $m \in \{1, \ldots, 12\}$ |
| $\boldsymbol{\mu}_m \in \mathbb{R}^S$ | Vector of site means for month $m$ |
| $\boldsymbol{\sigma}_m \in \mathbb{R}^S$ | Vector of site standard deviations for month $m$ |
| $\mathbf{S}_0^{(m)} \in \mathbb{R}^{S \times S}$ | Lag-0 cross-correlation matrix for month $m$ |
| $\mathbf{S}_1^{(m)} \in \mathbb{R}^{S \times S}$ | Lag-1 cross-correlation matrix (month $m+1$ on month $m$) |
| $\mathbf{A}^{(m)} \in \mathbb{R}^{S \times S}$ | Autoregressive coefficient matrix for the transition from month $m$ to $m+1$ |
| $\mathbf{B}^{(m)} \in \mathbb{R}^{S \times S}$ | Lower Cholesky factor of the innovation covariance |
| $\boldsymbol{\varepsilon}_t \in \mathbb{R}^S$ | Independent standard normal innovation vector |
| $N$ | Number of complete years in the historical record |

## Formulation

### Standardization

An optional log transformation $Q \mapsto \ln(Q + 1)$ may be applied first to reduce skewness. Flows are then standardized by monthly statistics:

$$
Z_{t,s} = \frac{Q_{t,s} - \mu_{m(t),s}}{\sigma_{m(t),s}}, \qquad s = 1, \ldots, S
$$

where $\mu_{m,s}$ and $\sigma_{m,s}$ are the sample mean and standard deviation of site $s$ in month $m$.

### Model Structure

The standardized flow vectors follow a periodic MAR(1) process:

$$
\mathbf{Z}_{t+1} = \mathbf{A}^{(m)} \mathbf{Z}_t + \mathbf{B}^{(m)} \boldsymbol{\varepsilon}_{t+1}, \qquad \boldsymbol{\varepsilon}_{t+1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

where $m = m(t)$ is the calendar month of time $t$ and the month indices wrap cyclically ($m = 12$ transitions to $m = 1$).

### Parameter Estimation

For each month $m$, let $\mathbf{Z}^{(m)}$ and $\mathbf{Z}^{(m+1)}$ denote the $n \times S$ matrices of standardized observations for the $n$ aligned (month $m$, month $m+1$) pairs, where $n = N$ for transitions within a year and $n = N - 1$ for December-to-January. The lag-0 and lag-1 cross-correlation matrices are all computed from this same aligned set:

$$
\mathbf{S}_0^{(m)} = \frac{1}{n - 1} \left(\mathbf{Z}^{(m)}\right)^\top \mathbf{Z}^{(m)}, \qquad \mathbf{S}_1^{(m)} = \frac{1}{n - 1} \left(\mathbf{Z}^{(m+1)}\right)^\top \mathbf{Z}^{(m)}, \qquad \mathbf{S}_0^{(m+1)} = \frac{1}{n - 1} \left(\mathbf{Z}^{(m+1)}\right)^\top \mathbf{Z}^{(m+1)}
$$

The autoregressive coefficient matrix is obtained by:

$$
\mathbf{A}^{(m)} = \mathbf{S}_1^{(m)} \left(\mathbf{S}_0^{(m)}\right)^{-1}
$$

If $\mathbf{S}_0^{(m)}$ is singular (for example, two perfectly collinear sites), the Moore-Penrose pseudo-inverse is used in place of the inverse and a warning is logged.

The innovation covariance is the residual after accounting for the autoregressive component:

$$
\mathbf{M}^{(m)} = \mathbf{S}_0^{(m+1)} - \mathbf{A}^{(m)} \mathbf{S}_0^{(m)} \left(\mathbf{A}^{(m)}\right)^\top
$$

Because $\mathbf{S}_0^{(m+1)}$ is computed from the same aligned pairs as $\mathbf{S}_0^{(m)}$ and $\mathbf{S}_1^{(m)}$, $\mathbf{M}^{(m)}$ is exactly the sample covariance of the least-squares regression residuals. $\mathbf{M}^{(m)}$ is symmetrized and, only if its smallest eigenvalue falls below $10^{-8}$, repaired by clipping the eigenvalues at $10^{-8}$ and reconstructing the matrix. The repaired matrix is *not* rescaled to unit diagonal: $\mathbf{M}^{(m)}$ is a covariance whose diagonal sets the innovation variance ($1 - \rho^2$ in the univariate case, the scalar form of Matalas Eqs. 1 and 10), and rescaling it would inflate the synthetic variance by roughly $1/(1-\rho^2)$. A warning is logged whenever the repair is applied. The Cholesky factorization then yields:

$$
\mathbf{B}^{(m)} = \text{chol}(\mathbf{M}^{(m)}), \qquad \mathbf{M}^{(m)} = \mathbf{B}^{(m)} \left(\mathbf{B}^{(m)}\right)^\top
$$

This is the matrix relation of Matalas Eq. 17, $\mathbf{B}\mathbf{B}^\top = \mathbf{S}_0 - \mathbf{A}\mathbf{S}_1^\top$, in its periodic form. Matalas obtains $\mathbf{B}$ by principal components; SynHydro uses the Cholesky factor, which is equivalent because $\mathbf{B}^* = \mathbf{B}\mathbf{O}$ for any orthogonal $\mathbf{O}$ satisfies the same equation (Matalas Eqs. 19-20). If the Cholesky factorization still fails after the eigenvalue repair, a diagonal $\mathbf{B}^{(m)} = \text{diag}\left(\sqrt{\max(\text{diag}(\mathbf{M}^{(m)}), 10^{-8})}\right)$ is used, which preserves each site's innovation variance but drops the cross-site innovation correlation for that transition; a warning is logged.

### Synthesis Procedure

1. Initialize $\mathbf{Z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ at a January $T_b$ months ahead of the first output month, where $T_b$ is the burn-in length (`burn_in`, default 120 months, rounded up to a whole number of years).
2. For each time step $t = 0, 1, \ldots, T_b + T - 1$, with $m = m(t)$:

$$
\mathbf{Z}_{t+1} = \mathbf{A}^{(m)} \mathbf{Z}_t + \mathbf{B}^{(m)} \boldsymbol{\varepsilon}_{t+1}
$$

3. Discard the first $T_b$ months and retain $\mathbf{Z}_{T_b}, \ldots, \mathbf{Z}_{T_b + T - 1}$; the first retained month is January.
4. Back-transform to flow space:

$$
\hat{Q}_{t,s} = \sigma_{m(t),s} \cdot Z_{t,s} + \mu_{m(t),s}
$$

5. If a log transformation was applied, invert: $\hat{Q}_{t,s} \leftarrow \exp(\hat{Q}_{t,s}) - 1$, then enforce non-negativity.

### Deviations from Matalas (1967)

- **Periodic extension.** Matalas (1967) describes a stationary model with a single pair of matrices $(\mathbf{A}, \mathbf{B})$. SynHydro fits 12 month-specific pairs (a periodic MAR(1), in the spirit of Thomas-Fiering), estimating $\mathbf{S}_0^{(m)}$ and $\mathbf{S}_1^{(m)}$ from the $N$ observations of each month. December-to-January pairs December of year $y$ with January of year $y+1$.
- **Log-space estimation.** With `log_transform=True` (default), all moments and matrices are estimated from $\ln(Q+1)$ and synthetic flows are back-transformed with $\exp(\cdot)-1$. Matalas (p. 939) notes that parameters estimated in log space do not in general reproduce the real-space moments; no moment matching or bias correction is applied, so real-space monthly means are reproduced only approximately (medians and log-space moments are preserved).
- **Calendar-month alignment.** Parameters are estimated by calendar month (`index.month`), so any start month is accepted; generated output is dated from January of the first observed year.
- **Burn-in.** The recursion is started from $\mathbf{Z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, which has no cross-site correlation. Without a burn-in the first month of every realization would have near-zero spatial correlation, followed by a short transient. SynHydro therefore simulates `burn_in` extra months (default 120, rounded up to whole years so the output still begins in January) and discards them before the first output month. A fixed seed remains fully reproducible. `burn_in=0` recovers the un-burned start.
- **Numerical fallbacks.** A singular $\mathbf{S}_0^{(m)}$ is inverted with the pseudo-inverse, and a failed Cholesky factorization of $\mathbf{M}^{(m)}$ falls back to a diagonal $\mathbf{B}^{(m)}$ (see Parameter Estimation). Both paths log a warning and are not part of Matalas (1967).

## Statistical Properties

The MAR(1) model preserves the first two moments (mean and variance) and the lag-1 autocorrelation at each site, as well as the contemporaneous cross-site correlation structure, all at the monthly scale. The seasonal cycle of these statistics is captured through the 12 sets of month-specific matrices.

Higher-order temporal autocorrelations (lag $> 1$) emerge only indirectly through the chain of first-order transitions and are generally underestimated. The model assumes that the standardized residuals are multivariate Gaussian, which may inadequately represent heavy-tailed or skewed marginal distributions. Long-range persistence (Hurst phenomenon) is not captured.

## Limitations

- First-order memory only; multi-month drought persistence is underrepresented.
- Multivariate Gaussian assumption may not hold for strongly skewed flows.
- Innovation covariance matrices may require positive-definiteness repair (eigenvalue clipping) when the record is short relative to the number of sites; the repair perturbs the innovation structure slightly.
- Stationarity is assumed; the model does not accommodate trends or regime shifts.

## References

**Primary:**
Matalas, N.C. (1967). Mathematical assessment of synthetic hydrology. *Water Resources Research*, 3(4), 937-945. https://doi.org/10.1029/WR003i004p00937

**See also:**
- Salas, J.D., Delleur, J.W., Yevjevich, V., and Lane, W.L. (1980). *Applied Modeling of Hydrologic Time Series*. Water Resources Publications.

---

**Implementation:** `src/synhydro/methods/generation/parametric/matalas.py`
