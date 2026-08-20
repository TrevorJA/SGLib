# Multisite Wavelet Phase Randomization (Brunner and Gilleland, 2020)

| | |
|---|---|
| **Type** | Hybrid (nonparametric wavelet phase randomization with parametric kappa marginals) |
| **Resolution** | Daily |
| **Sites** | Multisite |

## Overview

The Multisite Wavelet Phase Randomization generator extends the univariate Fourier phase randomization method of Brunner et al. (2019) to multiple sites using the continuous wavelet transform (CWT). The key innovation is drawing a single set of random phases from a white-noise CWT and applying those shared phases to all sites simultaneously. Because the phase perturbation is identical across sites, the spatial dependence structure of the original record is approximately preserved while the per-site power spectra (wavelet amplitudes) remain intact. The method is well suited for generating long daily streamflow ensembles where both temporal autocorrelation and inter-site synchrony -- such as basin-wide wet or dry spells -- must be reproduced.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_{t,s}$ | Observed daily streamflow at time $t$, site $s$ |
| $\hat{Q}_{t,s}$ | Synthetic daily streamflow at time $t$, site $s$ |
| $N$ | Length of the historical record (days, multiple of 365, no leap days) |
| $S$ | Number of sites |
| $d$ | Day of year (1 to 365) |
| $w$ | Half-window length for marginal fitting (default 15 days) |
| $\bar{Q}_s$ | Global mean of observed flow at site $s$ |
| $y_{t,s}$ | Pre-CWT transformed value of $Q_{t,s}$ (mean-centered or normal-score) |
| $\hat{y}_{t,s}$ | Synthetic transformed value for site $s$ |
| $a$ | CWT scale index |
| $W_s(a, t)$ | CWT coefficient at scale $a$, time $t$, site $s$ |
| $\phi_\varepsilon(a, t)$ | Phase of the white-noise CWT at scale $a$, time $t$ |
| $(\xi_d^s, \alpha_d^s, k_d^s, h_d^s)$ | Four-parameter kappa distribution fitted to day-of-year $d$, site $s$ |

## Formulation

### Model Structure

The generator operates in a transformed domain. For each site $s$, the observed flow series is pre-processed before the wavelet transform using one of two options controlled by the `transform` parameter:

- **`mean_center`** (default): subtract the global site mean, $\tilde{Q}_{t,s} = Q_{t,s} - \bar{Q}_s$. This matches the PRSim reference implementation of Brunner and Gilleland (2020).
- **`normal_score`**: apply the per-day-of-year Van der Waerden normal-score transform (see Parameter Estimation below), producing a more Gaussian-distributed CWT input.

In both cases the transformed series is denoted $y_{t,s}$ and the CWT is computed:

$$
W_s(a, t) = \int_{-\infty}^{\infty} y_{u,s} \, \frac{1}{\sqrt{a}} \psi^*\!\left(\frac{u - t}{a}\right) du
$$

where $\psi$ is the complex Morlet wavelet (pywt identifier `cmor1.5-1.0`, bandwidth 1.5, center frequency 1.0), evaluated with pywt's FFT-based CWT (`method="fft"`). In the discrete implementation, scales are spaced log-uniformly from 2 to $N/8$ over $n_{\text{scales}} = 100$ values, `np.geomspace(2, N / 8, 100)`.

**Scale grid (deliberate deviation from PRSim.wave).** PRSim.wave (`prsim.wave()`, `n_wave=100`) builds 100 log2-spaced scales spanning $1, \ldots, N$, rounds them to integers and removes duplicates, leaving roughly 90 unique integer scales; the paper likewise states 100 scales. SynHydro keeps 100 scales but differs in the end points and in not rounding: the grid omits scale 1 (the wavelet is barely resolved at one sample and contributes mostly aliasing), truncates at $N/8$ (coefficients at longer scales are dominated by the cone of influence), and uses exact geometric spacing so that $\Delta_j$ is constant and the inverse-CWT sum is well conditioned. The computational cost is comparable to PRSim.wave, so the choice is about scale coverage rather than speed. The choice is not neutral for short-lag autocorrelation: on the example record the lag-1 autocorrelation of the synthetic series was 0.71 with a minimum scale of 1, 0.81 with the default minimum scale of 2, and 0.92 with a minimum scale of 8 (observed 0.95), because removing the smallest scales removes the high-frequency noise that decorrelates adjacent days. Users who need closer short-lag fidelity should expect to trade it against the attenuation of the highest-frequency variance; the number of scales is configurable (`n_scales`), but the end points 2 and $N/8$ are fixed at fit time and are not constructor options.

**Wavelet (deviation from PRSim.wave).** The same `cmor1.5-1.0` Morlet is used both for the observed series and for the white-noise phase field. PRSim.wave uses the Morlet with non-dimensional frequency parameter `wparam` 5 for the data and 6 for the noise. For comparison, pywt's `cmorB-C` family with $B = 1.5$ corresponds to an effective Torrence and Compo (1998) frequency parameter $\omega_0 = 2\pi\sqrt{B/2} pprox 5.44$, between PRSim's 5 and the paper's 6. The bandwidth/centre-frequency pair determines how sharply scale maps to Fourier period; using one wavelet for both fields makes the shared-phase substitution scale-consistent, whereas PRSim's mismatch slightly blurs it. Any pywt complex wavelet name can be passed via the `wavelet` argument.

Phase randomization replaces the observed phase at each scale and time with a shared random phase:

$$
\hat{W}_s(a, t) = |W_s(a, t)| \cdot e^{i \phi_\varepsilon(a, t)}
$$

where $\phi_\varepsilon(a, t) = \arg\!\left[W_\varepsilon(a, t)\right]$ is derived from the CWT of a single white-noise realization $\varepsilon \sim \mathcal{N}(0, 1)$ of length $N$. Sharing $\phi_\varepsilon$ across all $S$ sites is the mechanism that preserves spatial correlation.

The synthetic normal-score series is recovered by an inverse CWT approximation:

$$
\hat{y}_{t,s} \propto \Delta_j \sum_{a} \frac{\operatorname{Re}\!\left[\hat{W}_s(a, t)\right]}{\sqrt{a}}
$$

where $\Delta_j = \ln(a_{j+1}/a_j)$ is the log-scale spacing (constant for geometrically spaced scales). This is the inverse-transform (reconstruction) formula of Torrence and Compo (1998); the wavelet-dependent reconstruction constant $C_\delta$ (and the other fixed factors) are absorbed into the per-site normalization $C$ described below, so only the relative weighting $\Delta_j / \sqrt{a}$ across scales matters here.

### Parameter Estimation

**Marginal distributions.** For each site $s$ and day of year $d$, the four-parameter kappa distribution is fitted by the method of L-moments. All observations within a $\pm w$-day circular window around $d$ (pooled across all years) are used, giving $2w + 1 = 31$ pooling days (default `win_h_length=15`) and $N_{\text{yr}} \times 31$ samples. Brunner and Gilleland (2020) and PRSim.wave use a 30-day window; the extra day is immaterial but means parameters will not match the R package exactly. A fit is rejected when the minimized L-moment objective exceeds 0.1 or the derived scale $\alpha \le 0$ (PRSim.wave has no such check); a rejected day inherits the previous day's parameters, or the next day's if none exist, and falls back to the empirical marginal otherwise. The kappa quantile function is:

$$
F^{-1}(p;\, \xi, \alpha, k, h) =
\xi + \frac{\alpha}{k} \left[ 1 - \left(\frac{1 - p^h}{h}\right)^k \right]
$$

with the GEV limit when $h = 0$. Parameters $(k, h)$ are found by minimizing the squared difference between sample and theoretical L-skewness ($\tau_3$) and L-kurtosis ($\tau_4$), and $(\xi, \alpha)$ are then determined analytically from the first two L-moments.

**Normal score transform** (`transform='normal_score'`). For each day $d$ at each site $s$, the $N_{\text{yr}}$ observations are ranked, and the Van der Waerden scores are assigned:

$$
y_{t,s} = \Phi^{-1}\!\left(\frac{r_{t,s}}{N_{\text{yr}} + 1}\right)
$$

where $r_{t,s}$ is the rank of observation $t$ among all observations on day $d$ at site $s$, and $\Phi^{-1}$ is the standard normal quantile function.

**Mean-center transform** (`transform='mean_center'`, default). The global site mean is subtracted:

$$
y_{t,s} = Q_{t,s} - \bar{Q}_s
$$

matching the PRSim reference implementation. The kappa marginal fitting always operates on the raw observed flows $Q_{t,s}$ regardless of which transform is selected.

### Synthesis Procedure

1. Fit kappa distribution parameters $(\xi_d^s, \alpha_d^s, k_d^s, h_d^s)$ for each day $d$ and site $s$ using L-moments over the $\pm w$-day window.

2. Transform $Q_{t,s}$ to $y_{t,s}$ for each site: subtract the global site mean (`transform='mean_center'`, the default, as in PRSim.wave), or apply the per-day-of-year normal score transform (`transform='normal_score'`).

3. Compute the CWT of each transformed series: $W_s(a, t) = \text{CWT}(y_{\cdot,s},\, a,\, \psi)$ on the 100-scale geometric grid.

4. For each realization:

   a. Draw white noise $\varepsilon \sim \mathcal{N}(0, 1)$ of length $N$.

   b. Compute the shared phase field:
   $$\phi_\varepsilon(a, t) = \arg\!\left[\text{CWT}(\varepsilon,\, a,\, \psi)\right]$$

   c. For each site $s$, form synthetic CWT coefficients:
   $$\hat{W}_s(a, t) = |W_s(a, t)| \cdot e^{i\phi_\varepsilon(a, t)}$$

   d. Recover the synthetic transformed series via the approximate inverse CWT:
   $$\hat{y}_{t,s} = C \, \Delta_j \sum_{a} \frac{\operatorname{Re}\!\left[\hat{W}_s(a, t)\right]}{\sqrt{a}}$$
   where $C$ is a per-site normalization constant chosen so that $\hat{y}_{t,s}$ has unit variance. Because the back-transform in step (e) is rank-based, neither the pre-CWT transform nor $C$ affects the output marginals; they only affect which series the wavelet amplitudes describe.

   e. Back-transform to original units. For each day $d$ and site $s$, rank $\{\hat{y}_{t,s} : \text{doy}(t) = d\}$, draw $N_{\text{yr}}$ kappa quantiles, and map via rank-order to produce $\hat{Q}_{t,s}$. Negative values (possible from the kappa lower tail) are each replaced by an independent uniform draw on $(0, \min_d Q_s)$, or by zero if the observed minimum for that day is zero; PRSim.wave draws a single replacement value per day instead; note that Brunner and Gilleland (2020) state that negative values are set to zero, so the paper and the PRSim code differ on this point and SynHydro follows the code.

5. If `n_years` requests more than $N$ days, repeat step 4 `ceil(365 * n_years / N)` times with independent noise and kappa samples, concatenate the chunks, and truncate to `365 * n_years` days. Spectral and spatial structure is preserved within each chunk, but consecutive chunks are independent, so temporal dependence is broken at each chunk boundary (see Limitations). With `n_years=None` (default) one chunk of length $N$ is returned.

## Statistical Properties

The method preserves, per site, the wavelet amplitude spectrum of the transformed series over the retained scale range (2 to $N/8$ days), which encodes seasonality and intermediate-range persistence. The back-transformation via the kappa distribution reproduces the daily marginal distributions accurately, including heavy tails, because L-moments are resistant to outliers and the kappa family subsumes GEV, logistic, exponential, and Pareto distributions as special cases.

Spatial correlation is preserved because all sites receive identical phase perturbations. In the wavelet domain, phase at each scale and time jointly determines whether a cross-site event (e.g., a flood pulse) occurs; sharing $\phi_\varepsilon$ keeps these co-occurrence patterns intact. The approach does not impose a parametric spatial model and is therefore robust to non-Gaussian and nonlinear dependence structures.

Higher-order cross-site statistics (e.g., cross-site lag correlations) are not explicitly targeted and will depend on how well the single shared phase field represents the observed joint phase structure across sites.

## Limitations

- The approximate inverse CWT does not guarantee exact reconstruction of the original signal; the round-trip error is acceptable for generating synthetic variability but not for signal decomposition.
- Short-lag autocorrelation is attenuated. Replacing the observed phases by those of a white-noise field (and the rank-based back-transform) lowers the lag-1 autocorrelation of the synthetic series relative to the observed record: on the example data 0.95 observed versus roughly 0.74 to 0.80 synthetic, with the exact value depending on the scale grid (see Model Structure). This is intrinsic to the method and also seen with PRSim.wave. Recession limbs and multi-day low-flow sequences are therefore rougher than observed.
- Tail dependence between sites is weaker than observed. Sharing the phase field reproduces the linear cross-site correlation well, but the co-occurrence of extremes (upper-tail dependence) is only partially preserved because each site's extremes are regenerated independently from its own kappa sample; basin-wide extreme events are under-represented relative to the record.
- Longer series requested via `n_years` are concatenations of independent chunks of length $N$; autocorrelation and persistence are lost at chunk boundaries, and multi-decadal persistence cannot be extrapolated beyond the observed record length.
- Spatial correlation is preserved on average over realizations, but individual realizations may deviate from the observed correlation matrix, particularly for short records.
- The CWT of a white-noise series does not have a flat spectrum, so the shared phase field is not strictly uniform over scale; this introduces some residual scale-dependence in the inter-site phase coherence.
- The method assumes stationarity; non-stationary trends or shifts in the observed record are embedded in the fitted amplitudes and marginals but not explicitly modeled.
- Computational cost of the FFT-based CWT scales as $O(S \cdot n_{\text{scales}} \cdot N \log N)$ and is independent of the wavelet support at each scale. With the default 100 scales the cost is comparable to PRSim.wave's roughly 90 unique integer scales.

## Implementation Notes

The default `transform='mean_center'` matches the PRSim reference implementation (Brunner and Gilleland, 2020). The `transform='normal_score'` option applies the Van der Waerden transform per day of year before the CWT, producing a more Gaussian-distributed spectral input; this approach is the basis of the single-site PRSim formulation (Brunner et al., 2019). In both cases the kappa back-transformation uses rank-order mapping, so the absolute scale of the pre-CWT series does not affect the fitted marginals.

## References

**Primary:**
Brunner, M.I., and Gilleland, E. (2020). Stochastic simulation of streamflow and spatial extremes: a continuous, wavelet-based approach. *Hydrology and Earth System Sciences*, 24, 3967-3982. https://doi.org/10.5194/hess-24-3967-2020

**See also:**
- Brunner, M.I., Bardossy, A., and Furrer, R. (2019). Technical note: Stochastic simulation of streamflow time series using phase randomization. *Hydrology and Earth System Sciences*, 23, 3175-3187. https://doi.org/10.5194/hess-23-3175-2019
- Hosking, J.R.M. (1994). The four-parameter kappa distribution. *IBM Journal of Research and Development*, 38(3), 251-258. https://doi.org/10.1147/rd.383.0251
- Hosking, J.R.M. (1990). L-moments: Analysis and estimation of distributions using linear combinations of order statistics. *Journal of the Royal Statistical Society, Series B*, 52(1), 105-124.
- Torrence, C., and Compo, G.P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79(1), 61-78. https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and Farmer, J.D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D*, 58, 77-94. https://doi.org/10.1016/0167-2789(92)90043-8

---

**Implementation:** `src/synhydro/methods/generation/hybrid/multisite_phase_randomization.py`
