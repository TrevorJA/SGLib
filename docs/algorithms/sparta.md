# SPARTA -- Stochastic Periodic AutoRegressive To Anything (Tsoukalas et al., 2018)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly |
| **Sites** | Multisite |

## Overview

SPARTA generates cyclostationary (seasonally varying) synthetic timeseries with per-month parametric marginal distributions and month-to-month temporal persistence. The core idea parallels SMARTA: simulate an auxiliary Gaussian process, then map to the target domain via inverse CDFs. However, SPARTA uses a Periodic AutoRegressive PAR(1) model as the auxiliary Gaussian process instead of SMA, which naturally handles seasonal nonstationarity by allowing different autoregressive coefficients, marginal distributions, and cross-correlation structures at each month. Equivalent correlations are identified through the Nataf joint distribution model.

The Tsoukalas et al. (2018) framework supports any continuous marginal distribution. The current SynHydro implementation restricts marginal selection to **gamma** and **lognormal** families, chosen per (month, site) by BIC. Extending to additional families (e.g., kappa, Weibull, GEV) is a planned enhancement.

## Notation

| Symbol | Description |
|--------|-------------|
| $x_{s,t}^i$ | Observed process $i$ at season (month) $s$, year $t$ |
| $z_{s,t}^i$ | Auxiliary standard Gaussian process for site $i$ at season $s$, year $t$ |
| $F_{x_s^i}$ | Target marginal CDF of process $i$ at season $s$ |
| $F_{x_s^i}^{-1}$ | Target marginal ICDF at season $s$ |
| $\Phi(\cdot)$ | Standard normal CDF |
| $\rho_s^i$ | Target lag-1 season-to-season autocorrelation of process $i$ at season $s$ |
| $\tilde{\rho}_s^i$ | Equivalent autocorrelation in the Gaussian domain |
| $\rho_s^{i,j}$ | Target lag-0 cross-correlation between processes $i$ and $j$ at season $s$ |
| $\tilde{\rho}_s^{i,j}$ | Equivalent lag-0 cross-correlation |
| $\tilde{A}_s$ | Diagonal autoregressive coefficient matrix at season $s$ |
| $\tilde{B}_s$ | Lower Cholesky factor of innovation covariance at season $s$ |
| $\tilde{C}_s$ | Equivalent lag-0 cross-correlation matrix at season $s$ |
| $\tilde{G}_s$ | Innovation covariance matrix at season $s$ |
| $m$ | Number of sites |
| $S$ | Number of seasons (12 for monthly) |
| $w_{s,t}$ | Standard normal i.i.d. innovation vector |

## Formulation

### Nataf Mapping

Identical to SMARTA. Each auxiliary Gaussian variate is mapped to the target domain by:

$$
x_{s,t}^i = F_{x_s^i}^{-1}\!\left(\Phi(z_{s,t}^i)\right)
$$

The key difference from SMARTA is that the ICDF varies by season $s$ and site $i$.

### PAR(1)-N Auxiliary Model

The multivariate contemporaneous PAR(1) model in the Gaussian domain is:

$$
\mathbf{z}_s = \tilde{A}_s \mathbf{z}_{s-1} + \tilde{B}_s \mathbf{w}_s, \qquad \mathbf{w}_s \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

where:
- $\tilde{A}_s = \text{diag}(\tilde{\rho}_s^1, \ldots, \tilde{\rho}_s^m)$ is the diagonal matrix of equivalent autoregressive coefficients
- $\tilde{G}_s = \tilde{C}_s - \tilde{A}_s \tilde{C}_{s-1} \tilde{A}_s^\top$ is the innovation covariance
- $\tilde{B}_s \tilde{B}_s^\top = \tilde{G}_s$ (Cholesky decomposition)

For the univariate case ($m = 1$), this simplifies to:

$$
z_s = \tilde{\rho}_s \, z_{s-1} + \sqrt{1 - \tilde{\rho}_s^2} \, w_s
$$

### Equivalent Correlation Identification

The Nataf procedure is applied pairwise. For autocorrelations, the pair involves different marginals (month $s$ and month $s-1$):

$$
\tilde{\rho}_s^i = \mathcal{F}^{-1}\!\left(\rho_s^i \mid F_{x_s^i}, F_{x_{s-1}^i}\right)
$$

This requires $S \times m$ Nataf inversions for autocorrelations, plus $S \times m(m-1)/2$ for cross-correlations.

### Parameter Estimation

1. **Marginal fitting:** Fit a distribution $F_{x_s^i}$ for each (month, site) pair -- $S \times m$ distributions.
2. **Target autocorrelations:** Compute empirical lag-1 season-to-season correlations for each site.
3. **Target cross-correlations:** Compute empirical lag-0 cross-correlations per month for each site pair.
4. **Equivalent autocorrelations:** Nataf inversion for each (site, month) pair.
5. **Equivalent cross-correlations:** Nataf inversion for each (site pair, month).
6. **PAR(1) matrices:** Build $\tilde{A}_s$, compute $\tilde{G}_s$, Cholesky decompose to get $\tilde{B}_s$.

**Input handling.** Daily or weekly input is summed to monthly totals before fitting. Season-to-season correlations (step 2) are computed on an $(N \times 12)$ calendar matrix whose column 0 is January; partial calendar years at either end of the record are trimmed for this step with a logged warning (at least two complete January-to-December years are required), while marginals and cross-correlations (steps 1 and 3) use every available month. Records may therefore start in any month.

**Repair of $\tilde{G}_s$.** When the Cholesky factorization of $\tilde{G}_s$ fails (for instance when the equivalent cross-correlations of adjacent months are inconsistent), $\tilde{G}_s$ is made positive definite by clipping its eigenvalues at $10^{-8}$ (`matrix_repair_method="spectral"`, default). Because $\tilde{G}_s$ is a covariance whose diagonal $1 - (\tilde{\rho}_s^i)^2$ fixes the innovation variance, it is *not* rescaled to unit diagonal; other `matrix_repair_method` values repair the correlation form of $\tilde{G}_s$ and restore its diagonal. The repair perturbs the off-diagonal structure slightly, so the cross-correlation of that month is reproduced only approximately.

### Synthesis Procedure

1. For each year $t = 1, \ldots, T$ and month $s = 1, \ldots, 12$, draw $\mathbf{w}_{s,t} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_m)$.

2. Compute the auxiliary Gaussian series via PAR(1) recursion:

$$
\mathbf{z}_{s,t} = \tilde{A}_s \, \mathbf{z}_{s-1,t} + \tilde{B}_s \, \mathbf{w}_{s,t}
$$

   with wrapping: $\mathbf{z}_{0,t} = \mathbf{z}_{12,t-1}$.

3. Map to the target domain using per-(month, site) ICDFs:

$$
x_{s,t}^i = F_{x_s^i}^{-1}\!\left(\Phi(z_{s,t}^i)\right)
$$

Generated output is dated from January of the first observed year, so position 0 of each realization is always January regardless of the observed start month.

### Deviations from Tsoukalas et al. (2018)

- **Nataf inversion.** Equivalent correlations are obtained with Gauss-Hermite quadrature on a grid of support points followed by polynomial/interpolation inversion (`nataf_method="GH"`, default), consistent with the anySim R package rather than the Monte-Carlo polynomial fit described in the paper. `"MC"` and `"Int"` are available as alternatives. Gauss-Hermite quadrature is less accurate for discrete or zero-inflated marginals (a point mass at zero is not well resolved by the quadrature nodes); use `nataf_method="MC"` or `"Int"` in that case.
- **Initial state and burn-in.** Following SimSPARTA.R, the univariate recursion starts from $z_{1,1} = w_{1,1}$ and the multivariate recursion from $\mathbf{z}_{1,1} = \tilde{B}_1 \mathbf{w}_{1,1}$ (covariance $\tilde{G}_1$ rather than $\tilde{C}_1$). No burn-in is discarded; the effect is confined to the first few months of each realization.
- **Marginal families.** Only gamma and lognormal marginals (BIC-selected by maximum likelihood, location fixed at zero) are implemented, whereas the paper's case studies fit Pearson type III marginals by the method of moments. The sample size used in BIC counts all values; this does not affect the selection because both candidates have two parameters.
- **Zero flows.** Observed zeros are excluded from the marginal fit for their (month, site) pair: preprocessing clips the stored record to a floor of $10^{-6}$, and any value at or below that floor is dropped before the gamma/lognormal MLE, so the fitted parameters describe the positive flows only. If fewer than five positive values remain, an exponential marginal with the month's mean is used instead. Because the fitted marginals are strictly positive, the generator never produces zeros; months with intermittent (zero-inflated) flow are therefore not explicitly modeled, and their dry-month frequency will not be reproduced. Season-to-season and cross-correlations (steps 2 and 3) are computed on the clipped record, so zeros do contribute to the correlation targets.

## Statistical Properties

SPARTA exactly preserves the target marginal distribution at each (month, site) by construction. Lag-1 season-to-season autocorrelations are preserved through the PAR(1) structure and Nataf inversion. Lag-0 cross-correlations across sites are preserved per month. Higher-order autocorrelations (lag > 1) are not explicitly modeled but emerge from the PAR(1) chain.

## Limitations

- Only lag-1 autocorrelation is explicitly modeled. Long-range dependence requires SMARTA or higher-order PAR.
- Per-season Nataf inversion is computationally intensive: $S \times (m + m(m-1)/2)$ inversions.
- Innovation covariance $\tilde{G}_s$ may not be positive-definite when sites have very different distributions or autocorrelation strengths across seasons; it is then repaired by eigenvalue clipping (see above), which slightly perturbs the cross-correlation of that month.
- The model assumes cyclostationarity (same seasonal pattern every year). Trends or regime shifts are not modeled.
- Only lag-0 cross-correlations are explicitly preserved; lagged cross-correlations emerge implicitly.

## References

**Primary:**
Tsoukalas, I., Efstratiadis, A., & Makropoulos, C. (2018). Stochastic periodic autoregressive to anything (SPARTA): Modeling and simulation of cyclostationary processes with arbitrary marginal distributions. Water Resources Research, 54(1), 161-185. https://doi.org/10.1002/2017WR021394

**See also:**
- Tsoukalas, I., Makropoulos, C., & Koutsoyiannis, D. (2018). Simulation of stochastic processes exhibiting any-range dependence and arbitrary marginal distributions. Water Resources Research, 54(11), 9484-9513. https://doi.org/10.1029/2017WR022462
- Cario, M. C., & Nelson, B. L. (1996). Autoregressive to anything: Time-series input processes for simulation. Operations Research Letters, 19(2), 51-58.
- Tsoukalas, I., Kossieris, P., & Makropoulos, C. (2020). Simulation of non-Gaussian correlated random variables, stochastic processes and random fields: Introducing the anySim R-package for environmental applications and beyond. Water, 12(6), 1645. https://doi.org/10.3390/w12061645

---

**Implementation:** `src/synhydro/methods/generation/parametric/sparta.py`
