# WARM -- Wavelet Auto-Regressive Method (Nowak et al., 2011)

| | |
|---|---|
| **Type** | Hybrid (wavelet decomposition + parametric AR + nonparametric residual bootstrap) |
| **Resolution** | Annual |
| **Sites** | Univariate (multi-site via Nowak 2010 disaggregation) |

## Overview

WARM generates synthetic annual streamflow that preserves both the marginal distribution and the non-stationary spectral structure of an observed record. The observed series is decomposed by the continuous wavelet transform (CWT) into time-frequency space; significant spectral bands are identified by chi-squared significance testing of the global wavelet spectrum against a red- or white-noise background (Torrence and Compo 1998). For each significant band the time-domain reconstruction is obtained from the band-restricted inverse CWT, divided by the square root of its Scale-Averaged Wavelet Power (SAWP) to remove the time-varying envelope, and modelled with an autoregressive process. A residual "noise" component captures everything outside the significant bands and is also fit with an AR model. Synthesis simulates the AR processes, restores each band's non-stationary envelope by multiplying the stationary signal by the square root of the historical SAWP (read cyclically with a uniformly random starting offset), sums bands and noise in the time domain, and applies the Nowak et al. (2011) Eq. 7 variance correction.

This implementation follows Nowak et al. (2011) Sections 2.1-2.3 exactly: per-band SAWP, time-domain AR fitting on band-reconstructed series after envelope removal, variance-preserving inverse CWT in the form of Eq. 4, the Eq. 7 variance correction, and the recommended bootstrap of empirical noise innovations.

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
| $C_\delta$ | Wavelet-specific reconstruction factor (T&C 1998 Eq. 12; calibrated for the chosen wavelet) |
| $\psi_0(0)$ | Real part of the mother wavelet at zero |
| $\kappa$ | Integrated squared modulus $\int \lvert \psi \rvert^2 \, dt$ of the mother wavelet; equals 1 for T&C's unit-energy convention and $\approx 0.326$ for PyWavelets `cmor1.5-1.0` |
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
| $vf$ | Variance correction factor (Nowak et al. 2011 Eq. 7) |

## Formulation

### Continuous Wavelet Transform

The observed annual flow series $\{Q_t\}_{t=1}^{N}$ is internally mean-centered for numerical stability and decomposed using the complex Morlet wavelet (PyWavelets `cmor1.5-1.0`, bandwidth $B = 1.5$, center frequency $C = 1.0$, equivalent to $\omega_0 = 2 \pi \approx 6.28$, a close approximation to the Nowak et al. (2011) / Torrence and Compo (1998) convention of $\omega_0 = 6$). The PyWavelets `cmor` family uses a different normalization than T&C's unit-energy Morlet; the empirically calibrated $C_\delta$ and the explicit $\kappa$ scaling on the significance threshold compensate for this normalization difference. Scales are constructed geometrically following Torrence and Compo (1998) Eq. 9-10:

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

where $\gamma$ is the wavelet decorrelation factor ($\gamma = 2.32$ for complex Morlet with $\omega_0 \approx 6$, T&C 1998 Table 2). The significance threshold at confidence level $p$ is

$$
\bar W_{\text{thr}}(a_j) = \sigma^2 \, \kappa \, P_{k_j} \, \frac{\chi^2_{\nu_j}(p)}{\nu_j}
$$

where $\sigma^2 = \mathrm{Var}(Q_t - \bar Q)$ is the variance of the mean-centered observed series, $\kappa = \int |\psi|^2 \, dt$ is the integrated squared modulus of the mother wavelet ($\kappa = 1$ for T&C's unit-energy Morlet, $\kappa = 1/\sqrt{2\pi B} \approx 0.326$ for PyWavelets `cmor1.5-1.0` with $B=1.5$), and the theoretical background spectrum is

$$
P_k = \frac{1 - \alpha^2}{1 + \alpha^2 - 2 \alpha \cos(2 \pi k)}, \qquad k = 1 / \lambda
$$

with lag-1 coefficient $\alpha$ either set to zero (white-noise background) or estimated from the data (red-noise background). The $\sigma^2 \kappa$ scaling on the threshold is essential: T&C 1998 Eq. 18 expresses the null distribution as $|W|^2 / \sigma^2 \sim (1/2) P_k \chi^2_2$ for a unit-energy wavelet; for a wavelet with $\kappa \neq 1$ the comparable normalization is $|W|^2 / (\sigma^2 \kappa)$. Bands are identified as contiguous runs of scales for which $\bar W(a_j) > \bar W_{\text{thr}}(a_j)$. Runs shorter than `min_band_scales` are discarded. Users may instead supply explicit period bands $[\lambda_{\min}, \lambda_{\max}]$, in which case all scales whose Fourier period lies within the interval form one band.

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

For the default complex Morlet `cmor1.5-1.0`, $\psi_0(0) = 1 / \sqrt{1.5 \pi} \approx 0.4607$ (analytic) and $C_\delta \approx 0.5587$ (calibrated empirically against the PyWavelets normalization via delta-function reconstruction, T&C 1998 Eq. 12). With these constants, $S^{(b)}_t$ is a variance-preserving reconstruction in the units of the original (mean-centered) flow series; no post-hoc moment matching is required.

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

An AR$(p^{(\eta)})$ model is fit to $\eta_t$ using the same Yule-Walker procedure. Following the Nowak et al. (2011) Section 4 discussion, this residual contains the stochastic high-frequency content that should not be smoothed by a bandwise wavelet treatment. The standardized empirical innovations $\hat\varepsilon_t = (\eta_t - \sum_k \phi_k^{(\eta)} \eta_{t-k}) / \sigma^{(\eta)}$ are retained for the default bootstrap innovation mode (see Synthesis).

### Variance Correction Factor (Eq. 7)

Independent simulation of bands and noise introduces two structural differences between observed and synthesized variance. First, the small cross-covariances between components observed in the data (Nowak et al. 2011 Eq. 6) are lost. Second, within each band the synthesized variance is $\mathrm{Var}(\hat{\tilde S}^{(b)}_t) \cdot \mathbb{E}[\hat P^{(b)}_t] \approx \mathrm{Var}(\tilde S^{(b)}_t) \cdot \overline{\bar P^{(b)}}$ -- the product of the stationary AR variance and the mean SAWP, because $\hat{\tilde S}^{(b)}_t$ and $\hat P^{(b)}_t$ are drawn independently in synthesis. The observed band variance $\mathrm{Var}(S^{(b)}_t)$ differs from this product because the observed stationary signal and observed SAWP are structurally coupled.

The centered synthetic is therefore scaled by the variance correction factor

$$
vf = \sqrt{ \frac{ \mathrm{Var}(Q_t - \bar Q) }{ \sum_b \mathrm{Var}(\tilde S^{(b)}_t) \cdot \overline{\bar P^{(b)}} + \mathrm{Var}(\eta_t) } }
$$

so that the ensemble total variance matches the observed total variance. The denominator is the expected variance of the independently-simulated sum, not the in-sample variance of the observed reconstructions. The square root reflects that the time series (not the variance) is multiplied by $vf$; the resulting variance scales by $vf^2$ to recover $\mathrm{Var}(Q_t - \bar Q)$.

### Synthesis Procedure

For each realization of length $T$:

1. **Per-band AR simulation.** For each band $b$, simulate a synthetic stationary series with Gaussian innovations:

$$
\hat{\tilde S}^{(b)}_t = \mu^{(b)} + \sum_{k=1}^{p^{(b)}} \phi_k^{(b)} \left( \hat{\tilde S}^{(b)}_{t-k} - \mu^{(b)} \right) + \sigma^{(b)} \, \varepsilon_t
$$

   with a burn-in to reach stationarity.

2. **Historical SAWP resampling.** Read the historical SAWP cyclically with a uniformly random integer offset $k \sim \mathrm{Uniform}\{0, \ldots, N-1\}$:

$$
\hat P^{(b)}_t = \bar P^{(b)}_{(t + k) \bmod N}, \qquad t = 1, \ldots, T
$$

   This preserves the full autocorrelation structure (and therefore the non-stationary envelope) of the historical SAWP series. The random offset varies the envelope phase across realizations without destroying its temporal structure. (An earlier implementation used i.i.d. bootstrap of SAWP, which destroyed the SAWP autocorrelation and produced a scale-mixture-of-Gaussians marginal with heavy tails; cyclic resampling resolves that bias.)

3. **Re-introduce non-stationarity.** Multiply by the square root of the historical SAWP to restore the time-varying envelope:

$$
\hat S^{(b)}_t = \hat{\tilde S}^{(b)}_t \cdot \sqrt{\hat P^{(b)}_t}
$$

4. **Noise simulation.** Simulate $\hat \eta_t$ from the noise AR model. With `noise_model='ar_bootstrap'` (default, following Nowak et al. 2011 Section 4 for the Lee's Ferry case), innovations are drawn by resampling with replacement from the empirical standardized residuals $\hat\varepsilon$, then rescaled by $\sigma^{(\eta)}$, preserving any non-normal features (skew, heavy tails). With `noise_model='ar_gaussian'`, innovations are drawn from $\mathcal{N}(0, \sigma^{(\eta) 2})$.

5. **Aggregate, apply variance correction, and add the historical mean:**

$$
\hat Q_t = \bar Q + vf \cdot \left( \sum_b \hat S^{(b)}_t + \hat \eta_t \right)
$$

6. **Apply the physical lower bound:** $\hat Q_t \leftarrow \max(\hat Q_t, L)$ with $L = 0$ by default. The combination of Eq. 7 variance correction and bootstrap noise innovations rarely produces negative annual sums, so the clamp is essentially inactive on typical streamflow records.

## Multi-site Simulation via Composition

WARM as published is a univariate generator. Nowak et al. (2011) Section 2.4 achieves multi-site simulation through a two-stage composition:

1. Apply WARM to an aggregate gauge time series (often the most-downstream gauge in the network, or a synthetic basin total constructed by summing contributing gauges).
2. Disaggregate the resulting WARM realizations spatially across upstream gauges using the proportional KNN method of Nowak et al. (2010).

In SynHydro, this composition is implemented by chaining `WARMGenerator` with `synhydro.methods.disaggregation.spatial.NowakDisaggregator`. The `WARMGenerator` class itself remains univariate. Cross-site spectral consistency depends on the spatial homogeneity of the basin: tributaries with substantially different spectral signatures (Nowak et al. 2011 Section 3.3, the San Juan example) may not inherit the aggregate band structure, in which case those gauges should be modeled independently and recombined.

## Statistical Properties

- **Mean.** Preserved by explicit re-addition of the historical mean after variance-corrected band+noise summation.
- **Variance.** Preserved by the Nowak et al. (2011) Eq. 7 variance correction factor applied to the centered synthetic.
- **Marginal distribution.** Reproduced well for streamflow records typical of temperate basins. The default bootstrap noise innovations preserve any skewness or heavy-tail features present in the residual; the band components themselves use Gaussian innovations on the standardized stationary series and are therefore approximately symmetric per band.
- **Lag-1 autocorrelation.** Captured at the band scale through per-band AR fits.
- **Spectral structure.** The non-stationary spectral envelope of significant bands is reproduced through the cyclic historical SAWP resampling and per-band reconstruction. The global spectrum is reproduced through the variance budget across bands and noise.
- **Higher moments.** Skewness in the noise residual is reproduced via bootstrap of empirical innovations (`noise_model='ar_bootstrap'`, default). Setting `noise_model='ar_gaussian'` falls back to symmetric innovations and may underrepresent the upper tail.

## Tail behavior

With Eq. 7 variance correction, historical SAWP resampling, and bootstrap noise innovations, the synthetic flow distribution closely matches the observed marginal across the full Flow Duration Curve, including the upper and lower tails. The lower-bound clamp $\hat Q_t \leftarrow \max(\hat Q_t, 0)$ is essentially inactive on typical perennial records: clamping events are rare because the variance correction prevents the symmetric-Gaussian heavy-tail behavior that an i.i.d. SAWP bootstrap would otherwise produce. Users who require strictly positive flows for non-perennial rivers may pass a small positive value via `lower_bound`.

## Limitations

- Annual frequency only; monthly or daily output requires a downstream temporal disaggregator.
- Univariate; multi-site requires composition with `NowakDisaggregator` as described above.
- Edge effects in the CWT (the cone of influence) degrade band identification near the start and end of the record. Records shorter than 30 years are discouraged.
- The chi-squared significance test assumes a smoothly varying background spectrum; multi-modal spectra may produce noisy band identification near the threshold.
- Cyclic resampling of SAWP preserves its full autocorrelation but produces a finite number of distinct envelope realizations (equal to the historical record length); for very large ensembles, neighboring realizations may share long-window SAWP patterns up to a circular shift.
- Bootstrap innovations cannot generate values outside the empirical residual support; extreme tail behavior beyond the observed record is not extrapolated.

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

**Implementation:** `src/synhydro/methods/generation/hybrid/warm.py`
