# WARM -- Wavelet Auto-Regressive Method (Nowak et al., 2011)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual |
| **Sites** | Univariate (multi-site via Nowak 2010 disaggregation) |

## Overview

WARM generates synthetic annual streamflow that preserves both the marginal distribution and the non-stationary spectral structure of an observed record. The observed series is decomposed by the continuous wavelet transform (CWT) into time-frequency space; significant spectral bands are identified by chi-squared significance testing of the global wavelet spectrum against a red- or white-noise background (Torrence and Compo 1998). For each significant band the time-domain reconstruction is obtained from the band-restricted inverse CWT, divided by the square root of its Scale-Averaged Wavelet Power (SAWP) to remove the time-varying envelope, and modelled with an autoregressive process. A residual "noise" component captures everything outside the significant bands and is also fit with an AR model. Synthesis simulates the AR processes, multiplies each band's stationary signal by the square root of a bootstrapped SAWP series to restore non-stationarity, and sums the resulting bands and noise in the time domain.

This implementation follows Nowak et al. (2011) Sections 2.1-2.3 exactly: per-band SAWP, time-domain AR fitting on band-reconstructed series after envelope removal, and variance-preserving inverse CWT in the form of Eq. 4.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_t$ | Observed annual streamflow at year $t$, $t = 1, \ldots, N$ |
| $\hat{Q}_t$ | Synthetic annual streamflow at year $t$ |
| $\bar{Q}$ | Sample mean of the observed record |
| $W(a_j, t)$ | CWT coefficient at scale $a_j$ and time $t$ |
| $a_j$ | Wavelet scale, $j = 0, \ldots, J$ |
| $\delta_j$ | Voice spacing in $\log_2$-scale (= $1 / n_{\text{voices}}$) |
| $\delta_t$ | Sampling period (one year for annual input) |
| $C_\delta$ | Wavelet-specific reconstruction factor (T&C 1998 Table 2) |
| $\psi_0(0)$ | Mother wavelet evaluated at zero (T&C 1998 Table 2) |
| $\bar P^{(b)}_t$ | Band-restricted SAWP for band $b$ at time $t$ |
| $S^{(b)}_t$ | Band-restricted time-domain reconstruction |
| $\tilde S^{(b)}_t$ | Stationary signal $S^{(b)}_t / \sqrt{\bar P^{(b)}_t}$ |
| $\eta_t$ | Noise residual (everything outside significant bands) |
| $\phi_k^{(b)}$ | AR coefficient at lag $k$ for band $b$ |
| $\sigma^{(b)}$ | Innovation standard deviation for band $b$ |
| $p^{(b)}$ | AR order for band $b$ |
| $\alpha$ | Lag-1 autocorrelation of $Q_t$ used for the red-noise background |
| $\nu$ | Equivalent degrees of freedom of the global spectrum estimator |
| $P_k$ | Theoretical background spectrum at Fourier wavenumber $k$ |

## Formulation

### Continuous Wavelet Transform

The observed annual flow series $\{Q_t\}_{t=1}^{N}$ is mean-centered and decomposed using a mother wavelet $\psi$ (default: Morlet) at $J + 1$ scales constructed geometrically following Torrence and Compo (1998) Eq. 9-10:

$$
a_j = s_0 \, 2^{j \delta_j}, \qquad j = 0, 1, \ldots, J
$$

with default smallest scale $s_0 = 2 \delta_t$, voice spacing $\delta_j = 1/8$, and largest scale capped at $N \delta_t / 2$. The CWT yields a coefficient matrix $W(a_j, t)$ of dimension $(J + 1) \times N$. Each scale $a_j$ corresponds to a Fourier period $\lambda_j$ via the wavelet-specific scale-to-period relation (Torrence and Compo 1998 Table 1).

### Significance Testing and Band Identification

The global wavelet spectrum is the time average of the local power:

$$
\bar W(a_j) = \frac{1}{N} \sum_{t=1}^{N} |W(a_j, t)|^2
$$

Following Torrence and Compo (1998) Eqs. 16-23, the global-spectrum estimator at scale $a_j$ is chi-squared distributed with equivalent degrees of freedom

$$
\nu_j = 2 \sqrt{1 + \left(\frac{N \delta_t}{\gamma a_j}\right)^2}
$$

where $\gamma$ is the wavelet decorrelation factor ($\gamma = 2.32$ for Morlet, $\gamma = 1.43$ for Mexican Hat). The significance threshold at confidence level $p$ is

$$
\bar W_{\text{thr}}(a_j) = P_{k_j} \cdot \frac{\chi^2_{\nu_j}(p)}{\nu_j}
$$

with the theoretical background spectrum

$$
P_k = \frac{1 - \alpha^2}{1 + \alpha^2 - 2 \alpha \cos(2 \pi k)}, \qquad k = 1 / \lambda
$$

and lag-1 coefficient $\alpha$ either set to zero (white-noise background) or estimated from the data (red-noise background). Bands are identified as contiguous runs of scales for which $\bar W(a_j) > \bar W_{\text{thr}}(a_j)$. Runs shorter than `min_band_scales` are discarded. Users may instead supply explicit period bands $[\lambda_{\min}, \lambda_{\max}]$, in which case all scales whose Fourier period lies within the interval form one band.

### Per-Band SAWP

For each band $b$ defined by the scale-index set $J_b = \{j_1, \ldots, j_2\}$, the Scale-Averaged Wavelet Power (Nowak et al. 2011 Eq. 5; Torrence and Compo 1998 Eq. 24) is computed with summation limits restricted to $J_b$:

$$
\bar P^{(b)}_t = \frac{\delta_j \, \delta_t}{C_\delta} \sum_{j \in J_b} \frac{|W(a_j, t)|^2}{a_j}
$$

Each band has its own SAWP time series, capturing how the strength of that band evolves over time.

### Per-Band Inverse CWT

The band-restricted time-domain reconstruction is obtained from the inverse CWT of Nowak et al. (2011) Eq. 4 (Torrence and Compo 1998 Eq. 11) with summation limits restricted to $J_b$:

$$
S^{(b)}_t = \frac{\delta_j \, \delta_t^{1/2}}{C_\delta \, \psi_0(0)} \sum_{j \in J_b} \frac{\Re\!\left[W(a_j, t)\right]}{a_j^{1/2}}
$$

The wavelet-specific constants are tabulated in Torrence and Compo (1998) Table 2: for the Morlet wavelet $C_\delta = 0.776$ and $\psi_0(0) = \pi^{-1/4}$; for the Mexican Hat $C_\delta = 3.541$ and $\psi_0(0) = (2/\sqrt{3}) \pi^{-1/4}$. With these constants, $S^{(b)}_t$ is a variance-preserving reconstruction in the units of the original (mean-centered) flow series; no post-hoc moment matching is required.

### Stationary Component and AR Fitting

Each band-reconstructed signal $S^{(b)}_t$ is non-stationary because its envelope tracks $\sqrt{\bar P^{(b)}_t}$. Dividing by the square root of SAWP yields an approximately stationary series:

$$
\tilde S^{(b)}_t = \frac{S^{(b)}_t}{\sqrt{\bar P^{(b)}_t + \epsilon}}
$$

where $\epsilon$ is a small floor preventing divide-by-zero where the band power vanishes. An AR$(p^{(b)})$ model is fit to $\tilde S^{(b)}_t$ via the Yule-Walker equations:

$$
R^{(b)} \, \boldsymbol{\phi}^{(b)} = \mathbf{r}^{(b)}, \qquad
\sigma^{(b) 2} = \gamma_0^{(b)} \left( 1 - \boldsymbol{\phi}^{(b) \top} \mathbf{r}^{(b)} \right)
$$

where $R^{(b)}$ is the Toeplitz autocorrelation matrix and $\mathbf{r}^{(b)}$ is the lag-$1, \ldots, p^{(b)}$ autocorrelation vector. The order $p^{(b)}$ is either fixed (`ar_select='fixed'`) or chosen by Akaike's information criterion over $[1, n_{\text{ar,max}}]$ (`ar_select='aic'`).

### Noise Residual

The noise component captures variability outside any significant band. It is obtained by subtracting the sum of all band reconstructions from the mean-centered observed series:

$$
\eta_t = (Q_t - \bar Q) - \sum_b S^{(b)}_t
$$

An AR$(p^{(\eta)})$ model is fit to $\eta_t$ using the same Yule-Walker procedure. Following the Nowak et al. (2011) Section 4 discussion, this residual contains the stochastic high-frequency content that should not be smoothed by a bandwise wavelet treatment.

### Synthesis Procedure

For each realization of length $T$:

1. **Per-band AR simulation.** For each band $b$, simulate a synthetic stationary series:

$$
\hat{\tilde S}^{(b)}_t = \mu^{(b)} + \sum_{k=1}^{p^{(b)}} \phi_k^{(b)} \left( \hat{\tilde S}^{(b)}_{t-k} - \mu^{(b)} \right) + \sigma^{(b)} \, \varepsilon_t
$$

   with a burn-in to reach stationarity.

2. **SAWP bootstrap.** Sample SAWP indices uniformly with replacement from the historical record to obtain $\hat P^{(b)}_t$ for $t = 1, \ldots, T$. This preserves the marginal distribution of historical band power.

3. **Re-introduce non-stationarity.** Multiply by the square root of the bootstrapped SAWP to restore the time-varying envelope:

$$
\hat S^{(b)}_t = \hat{\tilde S}^{(b)}_t \cdot \sqrt{\hat P^{(b)}_t}
$$

4. **Noise simulation.** Simulate $\hat \eta_t$ from the noise AR model.

5. **Aggregate in the time domain and add the historical mean back:**

$$
\hat Q_t = \bar Q + \sum_b \hat S^{(b)}_t + \hat \eta_t
$$

6. Enforce a lower bound on flow: $\hat Q_t \leftarrow \max(\hat Q_t, L)$, where $L$ is the `lower_bound` parameter (default $L = 0$ following Nowak et al. 2011; set $L$ to the observed minimum via `lower_bound='obs_min'` to suppress unphysical near-zero years on perennial rivers, see Tail behavior below).

## Multi-site Simulation via Composition

WARM as published is a univariate generator. Nowak et al. (2011) Section 2.4 achieves multi-site simulation through a two-stage composition:

1. Apply WARM to an aggregate gauge time series (often the most-downstream gauge in the network, or a synthetic basin total constructed by summing contributing gauges).
2. Disaggregate the resulting WARM realizations spatially across upstream gauges using the proportional KNN method of Nowak et al. (2010).

In SynHydro, this composition is implemented by chaining `WARMGenerator` with `synhydro.methods.disaggregation.spatial.NowakDisaggregator`. The `WARMGenerator` class itself remains univariate. Cross-site spectral consistency depends on the spatial homogeneity of the basin: tributaries with substantially different spectral signatures (Nowak et al. 2011 Section 3.3, the San Juan example) may not inherit the aggregate band structure, in which case those gauges should be modeled independently and recombined.

## Statistical Properties

- **Mean and variance.** Preserved approximately through the variance-preserving inverse CWT and explicit re-addition of the historical mean.
- **Marginal distribution.** Reproduced well in practice for near-Gaussian flows; departures appear when the observed record has substantial skewness, in which case a log transform of the input series is recommended (the implementation does not transform automatically).
- **Lag-1 autocorrelation.** Captured at the band scale through per-band AR fits.
- **Spectral structure.** The non-stationary spectral envelope of significant bands is reproduced through the SAWP bootstrap and per-band reconstruction. The global spectrum is reproduced through the variance budget across bands and noise.
- **Higher moments.** Skewness is not explicitly modeled and is generally underrepresented when the residual deviates from Gaussianity (Nowak et al. 2011 Fig. 14 caveat).

## Tail behavior

The Gaussian AR innovations on the standardized band components and on the noise residual can sample several standard deviations into the lower tail. Combined with bootstrap-resampled SAWP and independent simulation of bands and noise (Nowak et al. 2011 Eq. 7 variance correction is not applied here), a small fraction of synthetic annual sums can fall below the observed historical minimum. Empirically, on the Delaware-basin example record (80 years), about 2% of synthetic years fall below the observed minimum and roughly 0.3% would be negative before clamping. For a perennial river where zero-flow years are not physically plausible, the synthesis step
$$
\hat Q_t \leftarrow \max(\hat Q_t, L)
$$
exposes a tunable floor $L$ via the `lower_bound` parameter. Setting `lower_bound='obs_min'` floors at the observed annual minimum and removes the unphysical near-zero tail at the cost of a one-sided distortion of the lower percentiles. The default $L = 0$ reproduces the published Nowak et al. (2011) behavior.

## Limitations

- Annual frequency only; monthly or daily output requires a downstream temporal disaggregator.
- Univariate; multi-site requires composition with `NowakDisaggregator` as described above.
- Edge effects in the CWT (the cone of influence) degrade band identification near the start and end of the record. Records shorter than 30 years are discouraged.
- The chi-squared significance test assumes a smoothly varying background spectrum; multi-modal spectra may produce noisy band identification near the threshold.
- Gaussian AR innovations may underrepresent tails; a bootstrap or non-Gaussian AR for the noise component is suggested by Nowak et al. (2011) Section 4 for records with strongly non-Gaussian residuals.
- The variance correction factor of Nowak et al. (2011) Eq. 7 is *not* applied here; the implementation relies on the variance-preserving form of the inverse CWT and explicit mean re-addition. If the band/noise components exhibit non-trivial cross-correlation in a particular dataset, the user may observe a small variance underestimation.
- Lower-tail clamping at `lower_bound` introduces a hard discontinuity at the floor; users plotting flow-duration curves should be aware that the lowest one or two percent of synthetic values represent clamped extreme draws rather than smooth tail behavior.

## References

**Primary:**
Nowak, K., Rajagopalan, B., and Zagona, E. (2011). A Wavelet Auto-Regressive Method (WARM) for multi-site streamflow simulation of data with non-stationary spectra. *Journal of Hydrology*, 410(1-2), 1-12. https://doi.org/10.1016/j.jhydrol.2011.08.051

**Significance testing methodology and inverse-CWT constants:**
Torrence, C., and Compo, G.P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79(1), 61-78. https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2

**Spatial disaggregation (multi-site composition):**
Nowak, K., Prairie, J., Rajagopalan, B., and Lall, U. (2010). A nonparametric stochastic approach for multisite disaggregation of annual to daily streamflow. *Water Resources Research*, 46(8). https://doi.org/10.1029/2009WR008530

**See also:**
- Erkyihun, S.T., Rajagopalan, B., Zagona, E., Lall, U., and Nowak, K. (2016). Wavelet-based time series bootstrap model for multidecadal streamflow simulation using climate indicators. *Water Resources Research*, 52(5), 4061-4077. https://doi.org/10.1002/2016WR018696
- Kwon, H.-H., Lall, U., and Khalil, A.F. (2007). Stochastic simulation model for nonstationary time series using an autoregressive wavelet decomposition. *Water Resources Research*, 43(5).

---

**Implementation:** `src/synhydro/methods/generation/parametric/warm.py`
