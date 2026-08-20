# WARM -- Wavelet Auto-Regressive Method (Nowak et al., 2011)

| | |
|---|---|
| **Type** | Hybrid (wavelet decomposition + parametric AR + nonparametric residual bootstrap) |
| **Resolution** | Annual |
| **Sites** | Univariate (the paper's spatial proportion disaggregation is not implemented in SynHydro) |

## Overview

WARM generates synthetic annual streamflow that preserves both the marginal distribution and the non-stationary spectral structure of an observed record. The observed series is decomposed by the continuous wavelet transform (CWT) into time-frequency space; significant spectral bands are identified by chi-squared significance testing of the global wavelet spectrum against a white-noise (default) or red-noise background (Torrence and Compo 1998). For each significant band the time-domain reconstruction is obtained from the band-restricted inverse CWT, divided by the square root of its Scale-Averaged Wavelet Power (SAWP) to remove the time-varying envelope, and modelled with an autoregressive process. A residual "noise" component captures everything outside the significant bands and is also fit with an AR model. Synthesis simulates the AR processes, restores each band's non-stationary envelope by multiplying the stationary signal by the square root of the historical SAWP (in observed time order by default), sums bands and noise in the time domain, and applies a variance correction in the spirit of Nowak et al. (2011) Eqs. 6-7.

This implementation follows the structure of Nowak et al. (2011) Sections 2.1-2.3 (per-band SAWP, AR fitting on the band-reconstructed series after envelope removal, inverse CWT in the form of Eq. 4, bootstrap of empirical noise innovations) with several deliberate departures, each listed in the "Deviations from the Paper" section below: a per-band amplitude factor plus a ratio-form variance correction instead of the literal Eq. 7, Burg rather than Yule-Walker AR estimation, and AIC order selection.

**Choosing `sawp_resampling`.** The central feature of Nowak et al. (2011) is that the synthetic ensemble reproduces the *non-stationary* spectrum of the record, i.e. the timing of epochs of high and low spectral power (their Fig. 7). This is delivered only by the default `sawp_resampling='historical'`, which re-applies the observed SAWP envelope in historical time order, so every realization (and hence the ensemble mean) carries the observed epoch timing. The alternative `sawp_resampling='random_offset'` reads the envelope cyclically from a uniformly random starting year: each trace still has a realistic time-varying envelope, but the epoch timing differs between traces and the ensemble-mean local wavelet spectrum is stationary. Use `'random_offset'` when you do not want to condition the ensemble on the historical epoch timing (for example, when the record's epoch structure is not expected to recur at the same calendar positions); use the default when you want to reproduce the paper's behaviour.

**AR order matters.** An AR(1) process has a monotone spectrum and cannot carry a spectral peak, so a band modelled with AR(1) is simulated as red noise and the observed peak is lost. The default therefore selects the order by AIC over $[1, n_{\text{ar,max}}]$ (which picks $p \geq 2$ for any quasi-periodic band). Passing `ar_order` explicitly without `ar_select` switches to that fixed order; `ar_order=1` is accepted for backward compatibility but logs a warning.

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
| $c^{(b)}$ | Per-band amplitude factor matching synthetic band variance to $\mathrm{Var}(S^{(b)}_t)$ |
| $\bar P^{(b)}_t$ | Band-restricted SAWP for band $b$ at time $t$ |
| $S^{(b)}_t$ | Band-restricted time-domain reconstruction |
| $\tilde S^{(b)}_t$ | Stationary signal $S^{(b)}_t / \sqrt{\bar P^{(b)}_t}$ |
| $\eta_t$ | Noise residual (everything outside significant bands) |
| $\phi_k^{(b)}$ | AR coefficient at lag $k$ for band $b$ |
| $\sigma^{(b)}$ | Innovation standard deviation for band $b$ |
| $p^{(b)}$ | AR order for band $b$ |
| $\alpha$ | Lag-1 autocorrelation of $Q_t$ used for the optional red-noise background (zero for the default white-noise background) |
| $\nu$ | Equivalent degrees of freedom of the global spectrum estimator |
| $P_k$ | Theoretical background spectrum at Fourier wavenumber $k$ |
| $vf$ | Total variance correction factor (ratio form of Nowak et al. 2011 Eqs. 6-7) |

## Formulation

### Continuous Wavelet Transform

The observed annual flow series $\{Q_t\}_{t=1}^{N}$ is internally mean-centered for numerical stability and decomposed using the complex Morlet wavelet (PyWavelets `cmor1.5-1.0`, bandwidth $B = 1.5$, center frequency $C = 1.0$). The PyWavelets `cmor` envelope is $\exp(-t^2/B)$ rather than T&C's $\exp(-\eta^2/2)$, so in the dimensionless Torrence and Compo convention this wavelet has $\omega_0 = 2 \pi C \sqrt{B/2} \approx 5.44$ (not $2 \pi C \approx 6.28$), a close approximation to the Nowak et al. (2011) / Torrence and Compo (1998) choice of $\omega_0 = 6$. The PyWavelets `cmor` family uses a different normalization than T&C's unit-energy Morlet; the empirically calibrated $C_\delta$ and the explicit $\kappa$ scaling on the significance threshold compensate for this normalization difference. Scales are constructed geometrically following Torrence and Compo (1998) Eq. 9-10:

$$
a_j = s_0 \, 2^{j \delta_j}, \qquad j = 0, 1, \ldots, J
$$

with default smallest scale $s_0 = 2 \delta_t$, voice spacing $\delta_j = 1/8$, and largest scale capped at $N \delta_t / 2$. The CWT yields a coefficient matrix $W(a_j, t)$ of dimension $(J + 1) \times N$. Each scale $a_j$ corresponds to a Fourier period $\lambda_j$ taken from PyWavelets (`scale2frequency`: $\lambda_j = a_j \delta_t / C = a_j \delta_t$). The T&C 1998 Table 1 relation $\lambda = 4 \pi s / (\omega_0 + \sqrt{2 + \omega_0^2})$ evaluated for $\omega_0 = 5.44$ (with $s = a_j \sqrt{B/2}$ in T&C units) gives $\lambda_j \approx 0.98 \, a_j \delta_t$; the 2% difference is well inside one voice spacing ($2^{1/8} \approx 9\%$) and is ignored.

### Significance Testing and Band Identification

The global wavelet spectrum is the time average of the local power:

$$
\bar W(a_j) = \frac{1}{N} \sum_{t=1}^{N} |W(a_j, t)|^2
$$

Following Torrence and Compo (1998) Eqs. 16-23, the global-spectrum estimator at scale $a_j$ is chi-squared distributed with equivalent degrees of freedom

$$
\nu_j = 2 \sqrt{1 + \left(\frac{N \delta_t}{\gamma a_j}\right)^2}
$$

where $\gamma$ is the wavelet decorrelation factor ($\gamma = 2.32$, the T&C 1998 Table 2 value for the $\omega_0 = 6$ complex Morlet, used as an approximation for the $\omega_0 \approx 5.44$ `cmor1.5-1.0`). The significance threshold at confidence level $p$ is

$$
\bar W_{\text{thr}}(a_j) = \sigma^2 \, \kappa \, P_{k_j} \, \frac{\chi^2_{\nu_j}(p)}{\nu_j}
$$

where $\sigma^2 = \mathrm{Var}(Q_t - \bar Q)$ is the variance of the mean-centered observed series, $\kappa = \int |\psi|^2 \, dt$ is the integrated squared modulus of the mother wavelet ($\kappa = 1$ for T&C's unit-energy Morlet, $\kappa = 1/\sqrt{2\pi B} \approx 0.326$ for PyWavelets `cmor1.5-1.0` with $B=1.5$), and the theoretical background spectrum is

$$
P_k = \frac{1 - \alpha^2}{1 + \alpha^2 - 2 \alpha \cos(2 \pi k)}, \qquad k = 1 / \lambda
$$

with lag-1 coefficient $\alpha$ either set to zero (white-noise background, `background_spectrum='white'`, the default) or estimated from the data as the sample lag-1 autocorrelation clipped to $[0, 0.999)$, so a negative estimate reverts to the white background (red-noise background, `background_spectrum='red'`). The default white-noise test at the 95% level is the procedure of Nowak et al. (2011). The red background is the more conservative test for persistent streamflow, but it can suppress every band: on the bundled annual example (USGS-01434000, 1950-2025 annual means) the white test finds bands at 17 yr and 22-32 yr, while the red test finds no significant band, in which case the generator degenerates to a single AR noise model. The $\sigma^2 \kappa$ scaling on the threshold is essential: T&C 1998 Eq. 18 expresses the null distribution as $|W|^2 / \sigma^2 \sim (1/2) P_k \chi^2_2$ for a unit-energy wavelet; for a wavelet with $\kappa \neq 1$ the comparable normalization is $|W|^2 / (\sigma^2 \kappa)$. Bands are identified as contiguous runs of scales for which $\bar W(a_j) > \bar W_{\text{thr}}(a_j)$. Runs shorter than `min_band_scales` are discarded. Users may instead supply explicit period bands $[\lambda_{\min}, \lambda_{\max}]$, in which case all scales whose Fourier period lies within the interval form one band.

### Per-Band SAWP

For each band $b$ defined by the scale-index set $J_b = \{j_1, \ldots, j_2\}$, the Scale-Averaged Wavelet Power (Nowak et al. 2011 Eq. 5; Torrence and Compo 1998 Eq. 24) is computed with summation limits restricted to $J_b$:

$$
\bar P^{(b)}_t = \frac{\delta_j \, \delta_t}{C_\delta \, \kappa} \sum_{j \in J_b} \frac{|W(a_j, t)|^2}{a_j}
$$

Each band has its own SAWP time series, capturing how the strength of that band evolves over time. The $1/\kappa$ factor is not in the paper's Eq. 5, which assumes T&C's unit-energy Morlet ($\kappa = 1$); with PyWavelets' normalization it is required for the SAWP to be in variance units, so that the whole-spectrum SAWP averages over time to $\mathrm{Var}(Q_t)$ (T&C 1998 Eq. 14). Because the synthesis divides and then multiplies by $\sqrt{\bar P^{(b)}_t}$, $\kappa$ cancels in the generated flows; it only affects the reported `sawp_` values.

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

where $\epsilon$ is a small floor preventing divide-by-zero where the band power vanishes. An AR$(p^{(b)})$ model is fit to $\tilde S^{(b)}_t$. The order $p^{(b)}$ is either fixed (`ar_select='fixed'`) or chosen by Akaike's information criterion over $[1, n_{\text{ar,max}}]$ (`ar_select='aic'`, default). A stationary AR(1) cannot represent a spectral peak, so $p^{(b)} \geq 2$ is required for any quasi-periodic band.

**Estimator.** By default (`ar_method='burg'`) the coefficients $\boldsymbol{\phi}^{(b)}$ are estimated with Burg's recursion. The paper does not state the estimator. Burg does not constrain the lag-0 autocovariance, so the innovation variance is set from the estimated coefficients as

$$
\sigma^{(b) 2} = \gamma_0^{(b)} \left( 1 - \boldsymbol{\phi}^{(b) \top} \boldsymbol{\rho}^{(b)} \right)
$$

with $\gamma_0^{(b)}$ the sample variance of $\tilde S^{(b)}_t$ and $\boldsymbol{\rho}^{(b)}$ the fitted model's own theoretical lag-$1, \ldots, p^{(b)}$ autocorrelation, so the simulated stationary variance equals the sample variance.

The option `ar_method='yule_walker'` instead solves the Yule-Walker equations

$$
R^{(b)} \, \boldsymbol{\phi}^{(b)} = \mathbf{r}^{(b)}, \qquad
\sigma^{(b) 2} = \gamma_0^{(b)} \left( 1 - \boldsymbol{\phi}^{(b) \top} \mathbf{r}^{(b)} \right)
$$

where $R^{(b)}$ is the Toeplitz autocorrelation matrix and $\mathbf{r}^{(b)}$ is the sample lag-$1, \ldots, p^{(b)}$ autocorrelation vector. Yule-Walker estimates based on the biased sample autocovariance pull the complex pole pair of a narrow-band series well inside the unit circle and therefore broaden and lower the spectral peak (Kay and Marple 1981); on the 25-yr test case of Nowak et al. (2011) Section 2, Yule-Walker AR(4) reproduced about 0.6 of the peak power of the stationary component while Burg AR(4) reproduced about 0.8, which is why Burg is the default. Simulations are started from zero with a burn-in lengthened to $\log(10^{-3}) / \log r_{\max}$ samples when the largest pole radius $r_{\max}$ is close to one.

### Noise Residual

The noise component captures variability outside any significant band. It is obtained by subtracting the sum of all band reconstructions from the mean-centered observed series:

$$
\eta_t = (Q_t - \bar Q) - \sum_b S^{(b)}_t
$$

An AR$(p^{(\eta)})$ model is fit to $\eta_t$ with the same estimator and order selection as the bands (Burg by default, Yule-Walker if `ar_method='yule_walker'`). Following the Nowak et al. (2011) Section 4 discussion, this residual contains the stochastic high-frequency content that should not be smoothed by a bandwise wavelet treatment. The standardized empirical innovations $\hat\varepsilon_t = (\eta_t - \sum_k \phi_k^{(\eta)} \eta_{t-k}) / \sigma^{(\eta)}$ are retained for the default bootstrap innovation mode (see Synthesis).

### Per-Band Amplitude Factor and Variance Correction

Independent simulation of bands and noise introduces two structural differences between observed and synthesized variance.

**Within a band.** In synthesis $\hat{\tilde S}^{(b)}_t$ and $\hat P^{(b)}_t$ are drawn independently, so the synthetic band variance is $\mathrm{Var}(\tilde S^{(b)}_t) \cdot \overline{\bar P^{(b)}}$. The observed band variance $\mathrm{Var}(S^{(b)}_t)$ differs from this product because the observed stationary signal and SAWP are coupled: the ratio $S^{(b)2}_t / \bar P^{(b)}_t$ is larger in epochs where a coherent oscillation dominates the band (the real parts of $W$ add coherently across the band's scales in Eq. 4, while Eq. 5 sums their powers) than in epochs where band-limited noise dominates, so $\tilde S^{(b)}_t$ is not perfectly stationary and its large-amplitude epochs coincide with large SAWP. On the Section 2 test case this product under-states $\mathrm{Var}(S^{(b)}_t)$ by 15-35%. Each synthetic band is therefore multiplied by

$$
c^{(b)} = \sqrt{ \frac{ \mathrm{Var}(S^{(b)}_t) }{ \mathrm{Var}(\tilde S^{(b)}_t) \cdot \overline{\bar P^{(b)}} } }
$$

so that it reproduces the variance of its observed reconstruction, which is what the paper's component-wise simulation (Section 2.1, "the band passed components add up to the original data") presumes.

**Across components.** The small cross-covariances between the observed components (Nowak et al. 2011 Eq. 6) are lost under independent simulation. The paper writes the correction as $vf = 1 + 2\sigma_{xy} / \sigma^2_{x+y}$ (Eq. 7) and multiplies the combined trace by it. Taken literally that expression scales the variance by $vf^2$ and does not equal the ratio of observed to independent-sum variance, so the code applies the equivalent ratio form

$$
vf = \sqrt{ \frac{ \mathrm{Var}(Q_t - \bar Q) }{ \sum_b \mathrm{Var}(S^{(b)}_t) + \mathrm{Var}(\eta_t) } }
$$

to the centered synthetic series. This is not Eq. 7 verbatim; it is the factor that makes the ensemble total variance equal $\mathrm{Var}(Q_t - \bar Q)$ when the components are independent, which is the stated purpose of Eq. 7. On typical records $vf$ is within a few percent of one.

### Synthesis Procedure

For each realization of length $T$:

1. **Per-band AR simulation.** For each band $b$, simulate a synthetic stationary series with Gaussian innovations:

$$
\hat{\tilde S}^{(b)}_t = \mu^{(b)} + \sum_{k=1}^{p^{(b)}} \phi_k^{(b)} \left( \hat{\tilde S}^{(b)}_{t-k} - \mu^{(b)} \right) + \sigma^{(b)} \, \varepsilon_t
$$

   with a burn-in to reach stationarity.

2. **Historical SAWP re-application.** Read the historical SAWP cyclically from an integer offset $k$:

$$
\hat P^{(b)}_t = \bar P^{(b)}_{(t + k) \bmod N}, \qquad t = 1, \ldots, T
$$

   With the default `sawp_resampling='historical'`, $k = 0$: the observed envelope is applied in historical order (cyclically extended when $T > N$), as in Nowak et al. (2011) step (iii). Every realization then shares the observed timing of high- and low-power epochs, so the ensemble reproduces the early-epoch concentration of power shown in the paper's Fig. 7, and the peak-power ratio on the Section 2 test case is slightly higher (0.70 versus 0.65). With `sawp_resampling='random_offset'`, $k \sim \mathrm{Uniform}\{0, \ldots, N-1\}$ is drawn per realization: the SAWP autocorrelation is preserved (apart from one discontinuity at the wrap point), but **the timing of high- and low-power epochs is randomized**, so the ensemble-mean local wavelet spectrum is stationary and the paper's Fig. 7 is reproduced only trace by trace, at random positions. (An earlier implementation used i.i.d. bootstrap of SAWP, which destroyed the SAWP autocorrelation and produced a scale-mixture-of-Gaussians marginal with heavy tails.)

3. **Re-introduce non-stationarity.** Multiply by the square root of the historical SAWP to restore the time-varying envelope:

$$
\hat S^{(b)}_t = c^{(b)} \, \hat{\tilde S}^{(b)}_t \cdot \sqrt{\hat P^{(b)}_t}
$$

4. **Noise simulation.** Simulate $\hat \eta_t$ from the noise AR model. With `noise_model='ar_bootstrap'` (default, following Nowak et al. 2011 Section 4 for the Lee's Ferry case), innovations are drawn by resampling with replacement from the empirical standardized residuals $\hat\varepsilon$, then rescaled by $\sigma^{(\eta)}$, preserving any non-normal features (skew, heavy tails). With `noise_model='ar_gaussian'`, innovations are drawn from $\mathcal{N}(0, \sigma^{(\eta) 2})$.

5. **Aggregate, apply variance correction, and add the historical mean:**

$$
\hat Q_t = \bar Q + vf \cdot \left( \sum_b \hat S^{(b)}_t + \hat \eta_t \right)
$$

6. **Apply the physical lower bound:** $\hat Q_t \leftarrow \max(\hat Q_t, L)$ with $L = 0$ by default. The variance correction and bootstrap noise innovations rarely produce negative annual sums, so the clamp is essentially inactive on typical streamflow records.

## Multi-site Simulation (Not Implemented)

WARM as published is a univariate generator. Nowak et al. (2011) Section 2.4 achieves multi-site simulation through a two-stage composition:

1. Apply WARM to an aggregate gauge time series (often the most-downstream gauge in the network, or a synthetic basin total constructed by summing contributing gauges).
2. Disaggregate the resulting WARM realizations spatially across upstream gauges using the proportion method of Nowak et al. (2010): for each historical year compute the vector of site proportions of the aggregate flow, find KNN analog years of each simulated aggregate value, resample one with a distance-weighted kernel, and apply its proportion vector.

**SynHydro does not implement this spatial proportion disaggregation.** `synhydro.methods.disaggregation.temporal.nowak.NowakDisaggregator` implements only the temporal (annual to daily) KNN disaggregation of Nowak et al. (2010); it cannot be used for the Section 2.4 spatial step, and there is no `synhydro.methods.disaggregation.spatial` module. Multi-site WARM therefore requires the user to supply a spatial disaggregator. Nowak et al. (2011) Section 3.3 also caution that tributaries with substantially different spectral signatures (the San Juan example) do not inherit the aggregate band structure under proportion disaggregation.

## Statistical Properties

- **Mean.** Preserved by explicit re-addition of the historical mean after variance-corrected band+noise summation.
- **Variance.** Preserved by the per-band amplitude factor and the total variance correction factor applied to the centered synthetic.
- **Marginal distribution.** Reproduced well for streamflow records typical of temperate basins. The default bootstrap noise innovations preserve any skewness or heavy-tail features present in the residual; the band components themselves use Gaussian innovations on the standardized stationary series and are therefore approximately symmetric per band.
- **Lag-1 autocorrelation.** Captured at the band scale through per-band AR fits.
- **Spectral structure.** Each band's time-varying envelope is reproduced through the historical SAWP (in observed time order by default) and the band's AR model. On the Nowak et al. (2011) Section 2 test case (101 yr, 25-yr sinusoid with amplitude 1.0 for the first 50 yr and 0.2 after, plus N(0, 1) noise, eight observed seeds, 100 realizations each, white background) the ensemble-median global wavelet power at 25 yr is on average 0.70 of the observed value with the defaults (0.65 with `sawp_resampling='random_offset'`, 0.77 when the band is specified explicitly as 16-32 yr with `'random_offset'`), compared with 0.16 for the former AR(1) Yule-Walker default and 0.49 for AIC with Yule-Walker. The remaining shortfall is intrinsic to representing a near-deterministic oscillation by a stochastic AR process: a single 101-yr AR realization spreads its power over neighbouring scales and its peak power is right-skewed, so the ensemble median sits below the observed single-realization value.
- **Higher moments.** Skewness in the noise residual is reproduced via bootstrap of empirical innovations (`noise_model='ar_bootstrap'`, default). Setting `noise_model='ar_gaussian'` falls back to symmetric innovations and may underrepresent the upper tail.

## Tail behavior

With Eq. 7 variance correction, historical SAWP resampling, and bootstrap noise innovations, the synthetic flow distribution closely matches the observed marginal across the full Flow Duration Curve, including the upper and lower tails. The lower-bound clamp $\hat Q_t \leftarrow \max(\hat Q_t, 0)$ is essentially inactive on typical perennial records: clamping events are rare because the variance correction prevents the symmetric-Gaussian heavy-tail behavior that an i.i.d. SAWP bootstrap would otherwise produce. Users who require strictly positive flows for non-perennial rivers may pass a small positive value via `lower_bound`.

## Deviations from the Paper

| Item | Nowak et al. (2011) | SynHydro | Reason |
|---|---|---|---|
| AR order | "lower order AR models" (unspecified) | AIC over $[1, 5]$ by default; fixed order when `ar_order` is given | AR(1) cannot carry a peak |
| AR estimator | unspecified | Burg with variance matching; Yule-Walker optional | Yule-Walker damps narrow-band poles |
| SAWP normalization | Eq. 5 | Eq. 5 divided by $\kappa$ | PyWavelets' non-unit-energy Morlet; cancels in synthesis |
| Envelope re-application | historical SAWP in observed order | same, by default (`sawp_resampling='historical'`); `'random_offset'` reads the envelope from a random cyclic start | Default reproduces the paper's epoch timing; the option gives a stationary ensemble-mean spectrum when conditioning on historical epoch timing is not wanted |
| Wavelet | complex Morlet, $\omega_0 = 6$ | PyWavelets `cmor1.5-1.0` ($\omega_0 \approx 5.44$ in T&C units); $\gamma = 2.32$ tabulated for $\omega_0 = 6$; Fourier period $= a_j \delta_t$ from PyWavelets rather than T&C Table 1 ($\approx 0.98 \, a_j \delta_t$) | Closest PyWavelets wavelet; differences are far below one voice spacing |
| Red-noise lag-1 | not used (white test) | sample lag-1 clipped to $[0, 0.999)$ | Negative persistence reverts to white background; avoids degenerate AR(1) spectra |
| Variance correction | $vf = 1 + 2\sigma_{xy}/\sigma^2_{x+y}$ | per-band $c^{(b)}$ and ratio-form $vf$ | Matches the stated goal (observed total variance) exactly |
| Cone of influence | not discussed | not handled | Edge coefficients enter the global spectrum, SAWP and reconstruction unweighted |
| Multi-site | proportion disaggregation (Sec. 2.4) | not implemented | No spatial disaggregator in SynHydro |

## Limitations

- Annual frequency only; monthly or daily output requires a downstream temporal disaggregator.
- Univariate; the Section 2.4 spatial proportion disaggregation is not implemented (see above).
- No cone-of-influence handling: wavelet coefficients near the start and end of the record are affected by zero padding, and they enter the global spectrum, the SAWP and the band reconstruction with full weight. Band identification and the SAWP envelope are therefore least reliable within roughly $\sqrt{2} a_j$ years of either end at scale $a_j$. Records shorter than 30 years are discouraged.
- The chi-squared significance test assumes a smoothly varying background spectrum; multi-modal spectra may produce noisy band identification near the threshold.
- The default white-noise background (as in the paper) can flag low-frequency power that is explained by simple AR(1) persistence as a significant band; `background_spectrum='red'` tests against the AR(1) spectrum instead but may find no band on short persistent records.
- With the default `'historical'` re-application every realization shares the observed SAWP envelope, so the ensemble is conditioned on the historical epoch timing; with `'random_offset'` the cyclic shift preserves the SAWP autocorrelation but produces only $N$ distinct envelopes, randomizes the timing of wet/dry-power epochs, and introduces one envelope discontinuity at the wrap point. Either way, when $T > N$ the envelope repeats cyclically.
- The synthetic peak power of a strongly periodic band is typically 0.6-0.8 of the observed single-realization value (see Statistical Properties); bands that are near-deterministic oscillations are not reproduced as sharply as in the paper's figures.
- Bootstrap innovations cannot generate values outside the empirical residual support; extreme tail behavior beyond the observed record is not extrapolated.

## References

**Primary:**
Nowak, K., Rajagopalan, B., and Zagona, E. (2011). A Wavelet Auto-Regressive Method (WARM) for multi-site streamflow simulation of data with non-stationary spectra. *Journal of Hydrology*, 410(1-2), 1-12. https://doi.org/10.1016/j.jhydrol.2011.08.051

**Significance testing methodology and inverse-CWT constants:**
Torrence, C., and Compo, G.P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79(1), 61-78. https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2

**Spatial disaggregation used by the paper for multi-site simulation (not implemented in SynHydro):**
Nowak, K., Prairie, J., Rajagopalan, B., and Lall, U. (2010). A nonparametric stochastic approach for multisite disaggregation of annual to daily streamflow. *Water Resources Research*, 46(8). https://doi.org/10.1029/2009WR008530

**AR estimation:**
Kay, S.M., and Marple, S.L. (1981). Spectrum analysis: A modern perspective. *Proceedings of the IEEE*, 69(11), 1380-1419. https://doi.org/10.1109/PROC.1981.12184

**See also:**
- Erkyihun, S.T., Rajagopalan, B., Zagona, E., Lall, U., and Nowak, K. (2016). Wavelet-based time series bootstrap model for multidecadal streamflow simulation using climate indicators. *Water Resources Research*, 52(5), 4061-4077. https://doi.org/10.1002/2016WR018696
- Kwon, H.-H., Lall, U., and Khalil, A.F. (2007). Stochastic simulation model for nonstationary time series using an autoregressive wavelet decomposition. *Water Resources Research*, 43(5), W05407. https://doi.org/10.1029/2006WR005258

---

**Implementation:** `src/synhydro/methods/generation/hybrid/warm.py`
