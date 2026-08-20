# SMARTA -- Symmetric Moving Average (neaRly) To Anything (Tsoukalas et al., 2018)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual |
| **Sites** | Multisite |

## Overview

SMARTA generates stationary synthetic timeseries with arbitrary marginal distributions and any-range autocorrelation structure (short-range or long-range / Hurst-Kolmogorov). The core idea is to simulate an auxiliary standard Gaussian process using the Symmetric Moving Average (SMA) model, then map the Gaussian variates to the target domain via their inverse CDF. Because Pearson correlation is not invariant under this nonlinear transformation, the Gaussian process must use "equivalent" (inflated) correlation coefficients, identified through the Nataf joint distribution model. In multivariate mode the model additionally preserves the lag-0 cross-correlation matrix across sites.

## Notation

| Symbol | Description |
|--------|-------------|
| $x_t^i$ | Observed process $i$ at time $t$ |
| $z_t^i$ | Auxiliary standard Gaussian process for site $i$ |
| $F_{x^i}$ | Target marginal CDF of process $i$ |
| $F_{x^i}^{-1}$ | Target marginal ICDF (quantile function) of process $i$ |
| $\Phi(\cdot)$ | Standard normal CDF |
| $\phi_2(\cdot;\tilde{\rho})$ | Bivariate standard normal PDF with correlation $\tilde{\rho}$ |
| $\rho_\tau^i$ | Target autocorrelation of process $i$ at lag $\tau$ |
| $\tilde{\rho}_\tau^i$ | Equivalent (Gaussian-domain) autocorrelation of process $i$ at lag $\tau$ |
| $\rho_0^{i,j}$ | Target lag-0 cross-correlation between processes $i$ and $j$ |
| $\tilde{\rho}_0^{i,j}$ | Equivalent lag-0 cross-correlation between processes $i$ and $j$ |
| $\tilde{a}_\zeta^i$ | SMA weight coefficients for process $i$, $\zeta = -q, \ldots, q$ |
| $q$ | SMA truncation order (typically a power of 2, e.g. 512) |
| $v_t^i$ | Standard normal i.i.d. innovation for process $i$ |
| $\tilde{G}$ | Equivalent lag-0 cross-correlation matrix of innovations |
| $\tilde{B}$ | Lower Cholesky factor of $\tilde{G}$ |
| $m$ | Number of sites (processes) |
| $\kappa, \beta$ | Parameters of the Cauchy-type autocorrelation structure (CAS) |
| $H$ | Hurst coefficient |

## Formulation

### Nataf Mapping (Gaussian to Target Domain)

Each auxiliary Gaussian variate is mapped to the target domain by:

$$
x_t^i = F_{x^i}^{-1}\!\left(\Phi(z_t^i)\right)
$$

This guarantees exact preservation of the target marginal distribution.

### Equivalent Correlation Identification

Because Pearson's correlation is not invariant under the nonlinear ICDF mapping, the target correlation $\rho$ and equivalent Gaussian-domain correlation $\tilde{\rho}$ are related by:

$$
\rho = \mathcal{F}(\tilde{\rho} \mid F_{x^i}, F_{x^j})
= \frac{
  \int_{-\infty}^{\infty} \int_{-\infty}^{\infty}
  F_{x^i}^{-1}(\Phi(z^i))\;
  F_{x^j}^{-1}(\Phi(z^j))\;
  \phi_2(z^i, z^j;\, \tilde{\rho})\; dz^i\, dz^j
  \;-\; E[x^i]\, E[x^j]
}{
  \sqrt{Var[x^i]\; Var[x^j]}
}
$$

Key properties (Lemmas from the paper):
- $\rho$ is a strictly increasing function of $\tilde{\rho}$
- $|\rho| \leq |\tilde{\rho}|$ (equality only when both marginals are Gaussian)
- $\tilde{\rho} = 0$ iff $\rho = 0$

The relationship $\mathcal{F}(\cdot)$ generally has no closed-form solution (exception: log-normal case, Eq. 18-19 in paper). It is approximated numerically following the procedure of Appendix A in the paper (see also Tsoukalas et al., 2018a):

1. Select $\Omega$ support points $\tilde{\rho}_k$ in $[-1, 1]$ (default $\Omega = 9$).
2. For each support point, evaluate $\rho_k = \mathcal{F}(\tilde{\rho}_k)$. The paper describes Monte Carlo simulation (draw $N$ bivariate normal pairs with correlation $\tilde{\rho}_k$, map through the ICDFs, compute the sample Pearson correlation; recommended $N = 150{,}000$). The implementation's default is two-dimensional Gauss-Hermite quadrature (`nataf_method="GH"`, 21 nodes per axis), which is deterministic and much faster; `nataf_method="MC"` and `nataf_method="Int"` (direct numerical integration with `scipy.integrate.dblquad`) are also available.
3. Fit a polynomial of degree $d$ (default $d = 8$) through the $(\tilde{\rho}_k, \rho_k)$ pairs.
4. Invert the polynomial: evaluate it densely over $[-1, 1]$ and interpolate to find $\tilde{\rho}$ for each target $\rho$. (The anySim R package uses `polyval` on a fine grid followed by linear interpolation.)

Gauss-Hermite quadrature assumes smooth integrands and may be less robust than Monte Carlo for discrete or mixed-type marginals; for the continuous gamma/log-normal marginals used here it is accurate. Zero-inflated or discrete marginals should use `nataf_method="MC"` or `nataf_method="Int"` rather than GH.

This procedure is applied a total of $m(m+1)/2$ times:
- $m$ times for autocorrelation (one per site, each producing the full equivalent ACF vector $\tilde{\rho}_1^i, \ldots, \tilde{\rho}_q^i$)
- $m(m-1)/2$ times for lag-0 cross-correlations (one per site pair, each producing a single $\tilde{\rho}_0^{i,j}$)

### Cauchy-type Autocorrelation Structure (CAS)

CAS is a two-parameter theoretical ACF that captures both SRD and LRD:

$$
\rho_\tau^{CAS} = (1 + \kappa \beta \tau)^{-1/\beta}, \qquad \tau \geq 0
$$

Special cases:
- $\beta > 1$: long-range dependence (approximates HK/fGn behavior). When $\kappa = \kappa_0$ (see Eq. 8 in paper), CAS closely matches the fGn ACF with $H$ related by $\beta = 1/(2 - 2H)$.
- $\beta = 0$ (via L'Hopital): $\rho_\tau = \exp(-\kappa \tau)$, which is SRD (Markovian/AR(1) type).

The paper estimates $\kappa$ and $\beta$ by minimizing the MSE between the sample and theoretical ACF or climacogram, and recommends the climacogram for LRD processes because it is less biased than the sample ACF (Dimitriadis & Koutsoyiannis, 2015). **Implementation deviation:** this implementation fits CAS by ACF-MSE only, using the sample ACF up to lag $\min(q, n/3)$; no climacogram-based estimator is provided.

With `acf_model="hurst"` the implementation instead uses the fGn/HK ACF with $H$ derived from the CAS fit as $H = 1 - 1/(2\beta)$. If the CAS fit returns $\beta \leq 1$ (no LRD signal in the sample ACF) this mapping is undefined and the implementation falls back to $H = 0.6$, emitting a logger warning. Use `acf_model="cas"` (the default) when this happens.

### SMA Model (Auxiliary Gaussian Process)

The univariate SMA generating mechanism is:

$$
z_t^i = \sum_{\zeta=-q}^{q} \tilde{a}_{|\zeta|}^i \; v_{t+\zeta}^i
$$

where $v_t^i$ are i.i.d. standard normal, and the symmetric weights $\tilde{a}_\zeta^i$ satisfy:

$$
\tilde{\rho}_\tau^i = \sum_{\zeta=-q}^{q-\tau} \tilde{a}_{|\zeta|}^i \; \tilde{a}_{|\tau+\zeta|}^i, \qquad \tau = 0, 1, \ldots, q
$$

The weights are computed via FFT (Koutsoyiannis, 2000). Form the full symmetric equivalent ACF vector of length $2q+1$: $[\tilde{\rho}_q, \ldots, \tilde{\rho}_1, 1, \tilde{\rho}_1, \ldots, \tilde{\rho}_q]$. Compute its DFT to get the power spectrum $S_{\tilde{\rho}}(\omega)$. The DFT of the weight sequence is $S_{\tilde{a}}(\omega) = \sqrt{|S_{\tilde{\rho}}(\omega)|}$ (absolute value guards against small negative values from numerical imprecision). Apply the inverse FFT to obtain the weights $\tilde{a}_\zeta$. The weights must satisfy $\sum_\zeta \tilde{a}_\zeta^2 = 1$ (i.e., $\tilde{\rho}_0 = 1$).

Note: The paper (referencing Koutsoyiannis, 2000) writes $S_{\tilde{a}}(\omega) = \sqrt{2 S_{\tilde{\rho}}(\omega)}$ using a one-sided spectral convention. When using the full two-sided symmetric ACF as input to the DFT (as in the anySim R implementation), the factor of 2 is not needed.

### Multivariate Extension

For $m > 1$ sites, innovations $v_t^i$ must be contemporaneously cross-correlated. Define:

$$
\tilde{g}^{i,j} = \frac{\tilde{\rho}_0^{i,j}}{\sum_{\zeta=-q}^{q} \tilde{a}_{|\zeta|}^i \; \tilde{a}_{|\zeta|}^j}
$$

Form matrix $\tilde{G}$ with $\tilde{G}[i,j] = \tilde{g}^{i,j}$; its diagonal is $1 / \sum_\zeta (\tilde{a}^i_{|\zeta|})^2$, which equals one whenever the equivalent ACF is positive definite (Parseval). $\tilde{G}$ is the covariance of the innovations, not a correlation matrix. Decompose $\tilde{G} = \tilde{B}\tilde{B}^\top$ via Cholesky (if positive definite) or the diagonal-preserving repair described under Implementation Notes. Generate correlated innovations: $\mathbf{v}_t = \tilde{B}\,\mathbf{w}_t$ where $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

### Parameter Estimation

1. **Marginal fitting:** The paper allows any distribution with finite variance. The implementation fits a two-parameter gamma and a two-parameter log-normal (both with location fixed at zero) by maximum likelihood and selects between them by BIC, per site.
2. **Autocorrelation fitting:** Fit CAS parameters $(\kappa^i, \beta^i)$ to each site's sample ACF (ACF-MSE only; see above).
3. **Equivalent ACF:** For each site, compute $\tilde{\rho}_\tau^i$ for $\tau = 1, \ldots, q$ by inverting $\mathcal{F}$.
4. **SMA weights:** Compute $\tilde{a}_\zeta^i$ via FFT of the equivalent ACF.
5. **Equivalent cross-correlations:** For each pair $(i,j)$, compute $\tilde{\rho}_0^{i,j}$ by inverting $\mathcal{F}$.
6. **Cross-correlation decomposition:** Build $\tilde{G}$, decompose to get $\tilde{B}$.

### Synthesis Procedure

1. For each time step $t = 1, \ldots, T + 2q$, draw $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_m)$.

2. Compute correlated innovations: $\mathbf{v}_t = \tilde{B}\,\mathbf{w}_t$ (univariate case: $\tilde{B} = 1$).

3. For each site $i = 1, \ldots, m$, compute the auxiliary Gaussian series via SMA convolution:

$$
z_t^i = \sum_{\zeta=-q}^{q} \tilde{a}_{|\zeta|}^i \; v_{t+\zeta}^i, \qquad t = 1, \ldots, T
$$

4. Map to the target domain:

$$
x_t^i = F_{x^i}^{-1}\!\left(\Phi(z_t^i)\right)
$$

## Statistical Properties

SMARTA exactly preserves the target marginal distribution at each site (by construction via the ICDF mapping). The theoretical autocorrelation structure is preserved up to lag $q$ in the Gaussian domain; in the actual domain, correlation preservation depends on the accuracy of the Nataf polynomial approximation. The lag-0 cross-correlation structure is preserved across sites to the extent that the matrix $\tilde{G}$ is positive definite and the Nataf inversion is accurate. Lagged cross-correlations ($\tau > 0$) are not explicitly modeled but emerge from the SMA convolution with shared temporal structure.

The model is capable of reproducing long-range dependence (Hurst-Kolmogorov behavior) through the CAS autocorrelation structure, which is a key advantage over ARMA-type models. The climacogram (variance vs. aggregation scale) provides a diagnostic that is less biased than the ACF for verifying LRD preservation.

## Limitations

- Requires target marginal distributions with finite variance (excludes heavy-tailed distributions with infinite second moment).
- The maximum attainable correlation in the target domain may be less than 1 when marginals are highly skewed or differ significantly between sites (Frechet-Hoeffding bounds).
- The Nataf polynomial inversion is an approximation; accuracy depends on the number and placement of evaluation points and polynomial degree.
- Only lag-0 cross-correlations are explicitly preserved; lagged cross-correlations are implicitly determined by the SMA structure and may not match observations.
- The model assumes stationarity. Seasonal nonstationarity must be handled externally (e.g., treating each month as a separate process as in the daily rainfall example).
- SMA order $q$ must be large enough for the target ACF to decay sufficiently; for strong LRD, this requires large $q$ (e.g., $2^{12}$), increasing memory and computation.
- The Nataf evaluation (Gauss-Hermite quadrature or Monte Carlo) can be slow when many sites and lags are involved ($m + m(m-1)/2$ inversions).

## Implementation Notes

- **FFT normalization:** NumPy's `np.fft.fft` uses the same convention as R's `fft()`. When computing weights from the full symmetric ACF of length $2q+1$, no normalization adjustment is needed. The absolute value of the power spectrum is taken before the square root, $S_{\tilde{a}} = \sqrt{|S_{\tilde{\rho}}|}$, exactly as in anySim's `SimSMARTA`; small negative spectral values arising from numerical imprecision or a slightly non-PD equivalent ACF are therefore reflected rather than clamped to zero.
- **Input resolution:** The generator operates on annual totals. If the observed data are at a finer resolution (e.g. monthly, `"MS"`), `preprocessing` sums them to calendar-year (`"YS"`) totals before fitting. Output is always annual.
- **Long simulations:** The paper's LRD use case requires simulations of thousands of years. pandas' default nanosecond `DatetimeIndex` cannot represent dates beyond year 2262, so for runs that would pass that year the output index is built as a second-resolution (`datetime64[s]`) year-start `DatetimeIndex` with the same calendar values. Runs that stay within the nanosecond range use the standard `pd.date_range(freq="YS")` index, so default behaviour is unchanged. `Ensemble` construction and `Ensemble.to_hdf5()` accept the long index; note that `Ensemble.from_hdf5()` currently re-parses dates with nanosecond resolution and cannot round-trip files extending beyond 2262. The `Ensemble` date index is further limited to year 9999 (Python `datetime` maximum), so a single run longer than that must be split into multiple realizations.
- **Positive-definiteness:** After Nataf inversion, the matrix $\tilde{G}$ may not be positive definite. When Cholesky fails, a warning is logged and the matrix is repaired in a diagonal-preserving way: $\tilde{G}$ is split as $D^{1/2} R D^{1/2}$ with $D = \mathrm{diag}(\tilde{G})$, the correlation part $R$ is repaired with `repair_correlation_matrix` using `matrix_repair_method` (`"spectral"` eigenvalue clipping by default, or `"nearest"` Higham / `"hypersphere"`), and the original diagonal is restored. This keeps $\mathrm{diag}(\tilde{B}\tilde{B}^\top) = \mathrm{diag}(\tilde{G})$ exactly, so the variance of the SMA output $Z$ (and therefore the Nataf mapping) is unaffected; only the off-diagonal innovation covariances are perturbed, and the realized lag-0 cross-correlations then deviate from the targets by the size of that perturbation. An earlier version passed $\tilde{G}$ straight to `repair_correlation_matrix`, which rescales the diagonal to one; that is harmless when $\sum_\zeta (\tilde{a}_{|\zeta|})^2 = 1$ but would inflate or deflate the output variance otherwise.
- **Polynomial accuracy for skewed marginals:** For highly skewed or zero-inflated distributions, the $\mathcal{F}(\cdot)$ relationship is strongly nonlinear. The 8th-degree polynomial may oscillate near the bounds. Validate that interpolated $\tilde{\rho}$ values lie in $[-1, 1]$ and clip if necessary.
- **Reference R implementation:** The `anySim` R package (https://github.com/itsoukal/anySim) by the paper's authors provides a validated implementation with functions `EstSMARTA()` and `SimSMARTA()`.

## References

**Primary:**
Tsoukalas, I., Makropoulos, C., & Koutsoyiannis, D. (2018). Simulation of stochastic processes exhibiting any-range dependence and arbitrary marginal distributions. Water Resources Research, 54(11), 9484-9513. https://doi.org/10.1029/2017WR022462

**See also:**
- Koutsoyiannis, D. (2000). A generalized mathematical framework for stochastic simulation and forecast of hydrologic time series. Water Resources Research, 36(6), 1519-1533. https://doi.org/10.1029/2000WR900044
- Nataf, A. (1962). Determination des distributions de probabilites dont les marges sont donnees. Comptes Rendus de l'Academie des Sciences, 225, 42-43.
- Cario, M. C., & Nelson, B. L. (1996). Autoregressive to anything: Time-series input processes for simulation. Operations Research Letters, 19(2), 51-58. https://doi.org/10.1016/0167-6377(96)00017-X
- Tsoukalas, I., Kossieris, P., & Makropoulos, C. (2020). Simulation of non-Gaussian correlated random variables, stochastic processes and random fields: Introducing the anySim R-package for environmental applications and beyond. Water, 12(6), 1645. https://doi.org/10.3390/w12061645

---

**Implementation:** `src/synhydro/methods/generation/parametric/smarta.py`
