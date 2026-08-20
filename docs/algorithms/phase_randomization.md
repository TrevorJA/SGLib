# Phase Randomization (Brunner et al., 2019)

| | |
|---|---|
| **Type** | Hybrid (nonparametric FFT phase randomization with parametric kappa marginals) |
| **Resolution** | Daily |
| **Sites** | Univariate |

## Overview

Phase randomization generates synthetic daily streamflow by replacing the Fourier phase spectrum with uniform random phases while preserving the amplitude (power) spectrum. Because the power spectrum encodes all second-order temporal structure, this approach maintains both short-range autocorrelation and long-range dependence (the Hurst phenomenon). A four-parameter kappa distribution, fitted per day of year via L-moments, enables the back-transformation to reproduce or extrapolate beyond the observed marginal distributions.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_t$ | Observed daily streamflow at time $t$ |
| $\hat{Q}_t$ | Synthetic daily streamflow at time $t$ |
| $X_t$ | Normal score-transformed series |
| $\mathcal{F}\{X\}_k$ | Discrete Fourier Transform of $X$ at frequency index $k$ |
| $A_k$ | Amplitude (modulus) at frequency $k$, $A_k = |\mathcal{F}\{X\}_k|$ |
| $\varphi_k$ | Original phase at frequency $k$ |
| $\varphi_k'$ | Randomized phase, $\varphi_k' \sim \text{Uniform}(-\pi, \pi)$ |
| $N$ | Length of the observed series (in days) |
| $d$ | Day-of-year index, $d \in \{1, \ldots, 365\}$ |
| $\xi, \alpha, \kappa, h$ | Kappa distribution parameters (location, scale, shape 1, shape 2) |
| $\lambda_1, \lambda_2$ | First and second L-moments |
| $\tau_3, \tau_4$ | L-skewness and L-kurtosis ratios |

## Formulation

### Normal Score Transform

For each day of year $d$, the observed values across all years are ranked and mapped to standard normal quantiles using the Van der Waerden plotting position:

$$
X_t = \Phi^{-1}\!\left(\frac{r_d(Q_t)}{N_d + 1}\right)
$$

where $r_d(\cdot)$ is the rank among the $N_d$ observations for day $d$ and $\Phi^{-1}$ is the standard normal inverse CDF. This produces a series $\{X_t\}$ with approximately $\mathcal{N}(0,1)$ marginals at each calendar day.

**Deviation from PRSim.** The reference R implementation (`PRSim`) obtains the normal scores by drawing a *random* standard normal sample of size $N_d$ and assigning its sorted values to the observed ranks, so the normalized series (and hence the amplitude spectrum) differs slightly from run to run. SynHydro uses the deterministic Van der Waerden scores above, so the normalized series and its spectrum are fixed by the data; all randomness enters through the phases and the kappa back-transform.

### Fourier Decomposition

The Discrete Fourier Transform of the normalized series yields:

$$
\mathcal{F}\{X\}_k = A_k\,e^{i\varphi_k}, \qquad k = 0, 1, \ldots, N-1
$$

The power spectrum $|A_k|^2$ encodes all second-order temporal dependence, including the spectral slope that characterizes long-range persistence.

### Phase Randomization

New phases are drawn independently from a uniform distribution:

$$
\varphi_k' \sim \text{Uniform}(-\pi, \pi), \qquad k = 1, \ldots, \lfloor N/2 \rfloor
$$

The surrogate spectrum is constructed by combining the original amplitudes with the random phases:

$$
\mathcal{F}\{\hat{X}\}_k = A_k\,e^{i\varphi_k'}
$$

The DC component ($k = 0$) is preserved unchanged. Conjugate symmetry is imposed for the negative frequencies to ensure a real-valued inverse:

$$
\mathcal{F}\{\hat{X}\}_{N-k} = \overline{\mathcal{F}\{\hat{X}\}_k}
$$

For even-length signals, the Nyquist component ($k = N/2$) is set to its real modulus. The phase-randomized series in normal space is recovered via the inverse FFT:

$$
\hat{X}_t = \text{Re}\left(\mathcal{F}^{-1}\{\mathcal{F}\{\hat{X}\}\}_t\right)
$$

### Kappa Distribution and Back-Transformation

For each day of year $d$, a four-parameter kappa distribution (Hosking, 1994) is fitted via L-moment matching to the pooled observations in a circular moving window of $\pm$`win_h_length` days around $d$ (default 15, i.e. a 31-day window $d-15, \ldots, d+15$ across all years). Brunner et al. (2019) and PRSim describe a 30-day window; the one-day difference is immaterial in practice but means fitted parameters will not match the R package to the last digit. The quantile function is:

$$
F^{-1}(u) = \xi + \frac{\alpha}{\kappa}\left[1 - \left(\frac{1 - u^h}{h}\right)^{\!\kappa}\right]
$$

with special cases: when $h = 0$ it reduces to the generalized extreme value (GEV) distribution; when $\kappa = 0$ it further reduces to the Gumbel distribution.

The L-moments are computed from probability weighted moments $b_0, b_1, b_2, b_3$:

$$
\lambda_1 = b_0, \quad \lambda_2 = 2b_1 - b_0, \quad \tau_3 = \frac{2(3b_2 - b_0)}{2b_1 - b_0} - 3, \quad \tau_4 = \frac{5(2(2b_3 - 3b_2) + b_0)}{2b_1 - b_0} + 6
$$

The shape parameters $(\kappa, h)$ are determined by minimizing the squared difference between the sample $(\tau_3, \tau_4)$ and the theoretical L-moment ratios of the kappa distribution (Nelder-Mead from $(1, 1)$, as in PRSim). The location and scale parameters $(\xi, \alpha)$ are then derived analytically from $\lambda_1$ and $\lambda_2$.

**Fit-rejection rule (SynHydro addition).** A fit is rejected when the minimized objective $(\tau_3 - \tau_3^{\text{th}})^2 + (\tau_4 - \tau_4^{\text{th}})^2$ exceeds 0.1 or when the derived scale $\alpha \le 0$. PRSim accepts whatever the optimizer returns. A rejected day copies the parameters of the previous day ($d - 1$); days that still have no parameters after the forward pass copy from the following day. If no neighbour has parameters the day falls back to the empirical marginal. Rejections are logged at debug level.

**Upper-tail caveat.** With $\kappa < 0$ (heavy upper tail), which the L-moment fit returns for most days on typical streamflow records, the kappa quantile function is unbounded above and the rank-matched back-transform can produce daily values many times larger than the observed maximum (an order of magnitude or more is not unusual for long simulations). This is inherent to the published method and identical to PRSim's behaviour, but users who need to bound extremes should use `marginal="empirical"` or post-process the output.

The back-transformation maps the phase-randomized normal scores to the fitted kappa distribution by rank matching: for each day $d$, the ranks of $\hat{X}_t$ among all values for that day are computed, a kappa sample of the same size is generated and sorted, and the ranks are used to select the corresponding kappa quantiles. If the empirical marginal option is used instead, observed values replace the kappa sample, preventing extrapolation beyond the historical range. Negative back-transformed values (possible with a kappa lower tail extending below zero) are replaced by independent uniform draws on $(0, \min_d Q)$ for that day, or by zero if the observed minimum is zero. PRSim draws one uniform replacement value per day per realization and reuses it for every negative value on that day.

**PRSim options not exposed.** The PRSim arguments `n_par`, `GoFtest`, `marginalpar`, `verbose` and `suppWarn` have no SynHydro equivalent; the kappa marginal is always four-parameter and no goodness-of-fit test is run.

### Synthesis Procedure

1. Remove leap days and construct the day-of-year index.
2. Fit the kappa distribution for each day $d$ (using a moving window of $\pm 15$ days).
3. Apply the normal score transform to obtain $\{X_t\}$.
4. Compute the FFT and extract amplitudes $\{A_k\}$.
5. Draw random phases $\{\varphi_k'\}$ and construct the surrogate spectrum.
6. Apply the inverse FFT to obtain $\{\hat{X}_t\}$.
7. Back-transform via rank matching against kappa (or empirical) samples.
8. Enforce non-negativity.
9. If `n_years` requests more days than the observed record, repeat steps 5-8 and concatenate.

### Output Length and `n_years` Chunking

One phase-randomized surrogate has exactly the length $N$ of the observed (leap-day-free) record, because the amplitude spectrum is defined on $N$ frequencies. This is the only output length the paper and PRSim produce. SynHydro's `generate(n_years=...)` extends this as follows: `ceil(365 * n_years / N)` independent surrogates are generated (each with its own phases and its own kappa sample), back-transformed, concatenated end to end, and the result is truncated to `365 * n_years` days. By default (`n_years=None`) a single surrogate of length $N$ is returned.

Consequences of chunking:

- Within each chunk the full amplitude spectrum is preserved; across a chunk boundary the two segments are independent, so autocorrelation, spectral continuity and persistence are broken at every multiple of $N$ days (preprocessing trims the record to a whole multiple of 365 days, so boundaries fall between whole years).
- The spectrum of a concatenated series is therefore not the observed spectrum at periods comparable to or longer than $N$, and low-frequency (multi-decadal) persistence cannot be extrapolated beyond the observed record length.
- The output index always starts on 1 January of a nominal year; each chunk follows the day-of-year sequence of the observed record, so the seasonal cycle is aligned with the output dates only when the observed record starts on 1 January.

For simulations that require continuous long-range dependence over horizons longer than the record, prefer a single surrogate per realization and a larger ensemble.

## Statistical Properties

The method preserves the full power spectrum of the normalized series, which in expectation reproduces all temporal autocorrelations including long-range dependence characterized by the Hurst exponent. The seasonal marginal distributions are preserved (or extrapolated via the kappa fit) through the day-of-year back-transformation.

Phase coherence is destroyed by design, meaning the temporal sequencing of events (e.g., the specific ordering of flood peaks within a year) is randomized. Individual autocorrelation values are preserved only in expectation across the ensemble, not in each realization. The method is univariate and does not model spatial dependence.

## Limitations

- Univariate only; no spatial correlation between sites.
- A single surrogate has the observed series length; longer series requested via `n_years` are concatenations of independent surrogates and lose temporal dependence at the chunk boundaries (see Output Length and `n_years` Chunking).
- February 29 dates are excluded.
- Kappa distribution fitting may fail or be rejected for some days of year (objective > 0.1 or $\alpha \le 0$), in which case the adjacent day's parameters or the empirical distribution are used.
- The kappa upper tail is unbounded for $\kappa < 0$ and can extrapolate far beyond the observed maximum.
- The kappa fitting window is 31 days ($\pm 15$) rather than the 30 days described in the paper, and normal scores are deterministic (Van der Waerden) rather than PRSim's random normal sample; both are minor and do not affect the algorithm's properties.
- Minimum of 2 complete years required; 10+ recommended for stable kappa fits.

## References

**Primary:**
Brunner, M.I., Bardossy, A., and Furrer, R. (2019). Technical note: Stochastic simulation of streamflow time series using phase randomization. *Hydrology and Earth System Sciences*, 23, 3175-3187. https://doi.org/10.5194/hess-23-3175-2019

**See also:**
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and Farmer, J.D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D*, 58, 77-94.
- Hosking, J.R.M. (1990). L-moments: Analysis and estimation of distributions using linear combinations of order statistics. *Journal of the Royal Statistical Society Series B*, 52, 105-124.
- Hosking, J.R.M. (1994). The four-parameter kappa distribution. *IBM Journal of Research and Development*, 38, 251-258.

---

**Implementation:** `src/synhydro/methods/generation/hybrid/phase_randomization.py`
