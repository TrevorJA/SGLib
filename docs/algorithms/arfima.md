# ARFIMA -- Autoregressive Fractionally Integrated Moving Average (Hosking, 1984)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly / Annual |
| **Sites** | Univariate |

## Overview

The ARFIMA(p, d, q) model extends classical ARMA by allowing the differencing parameter $d$ to take fractional values in $(-0.5, 0.5)$; persistence (the Hurst phenomenon) corresponds to $d > 0$. This enables the model to reproduce the long-range dependence (Hurst phenomenon) observed in many hydrologic time series, where the autocorrelation function decays hyperbolically rather than exponentially. The short-memory ARMA(p, q) component captures local temporal structure, while the fractional integration parameter $d$ governs the rate of low-frequency spectral divergence. The relationship $H = d + 0.5$ links the model directly to the Hurst exponent.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_t$ | Observed streamflow at time $t$ |
| $\hat{Q}_t$ | Synthetic streamflow at time $t$ |
| $Y_t$ | Shifted-log-transformed streamflow, $Y_t = \ln(Q_t - \tau_{m(t)})$ |
| $X_t$ | Standardized stationary residual, fit by ARFIMA |
| $W_t$ | Fractionally differenced series |
| $\tau_m$ | Stedinger-Taylor lower bound for month $m$ |
| $d$ | Fractional differencing parameter, $d \in (-0.5, 0.5)$ |
| $H$ | Hurst exponent, $H = d + 0.5$ |
| $p, q$ | Orders of the AR and MA components |
| $\phi_k$ | AR coefficients, $k = 1, \ldots, p$ |
| $\theta_k$ | MA coefficients, $k = 1, \ldots, q$ |
| $\varepsilon_t$ | White noise innovation, $\varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)$ |
| $\pi_k$ | Fractional differencing coefficients (forward) |
| $\psi_k$ | Inverse fractional differencing coefficients |
| $B$ | Backshift operator, $B X_t = X_{t-1}$ |
| $K$ | Truncation lag for the infinite coefficient series (generation side) |
| $M$ | Number of backcast presample values (default 30) |
| $e_t$ | One-step ARMA innovation (residual) |
| $\mu_m, \sigma_m$ | Monthly mean and standard deviation of $Y_t$ (log space) |
| $N$ | Length of the observed record |

## Formulation

### Model Structure

The ARFIMA(p, d, q) process is defined by:

$$
\Phi(B)\,(1 - B)^d\,X_t = \Theta(B)\,\varepsilon_t
$$

where $\Phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ is the AR polynomial, $\Theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ is the MA polynomial, and $(1 - B)^d$ is the fractional differencing operator. The process is stationary and invertible when $-0.5 < d < 0.5$ and the roots of $\Phi$ and $\Theta$ lie outside the unit circle.

The fractional differencing operator is expanded as an infinite-order filter:

$$
(1 - B)^d = \sum_{k=0}^{\infty} \pi_k\,B^k
$$

with coefficients defined recursively:

$$
\pi_0 = 1, \qquad \pi_k = \pi_{k-1} \cdot \frac{k - 1 - d}{k}, \quad k \geq 1
$$

Hosking (1984) writes the exact operator applied to an observed record as Eq. 7, which involves the unobserved presample values $X_0, X_{-1}, \ldots$. Replacing all presample values by the mean gives his Eq. 8,

$$
\nabla_0^d X_t = \sum_{j=0}^{t-1} \pi_j X_{t-j},
$$

and replacing only the values before $t = -M$ by the mean while backcasting $X_{-M}, \ldots, X_{-1}$ gives Eq. 9,

$$
W_t = \nabla_M^d X_t = \sum_{j=0}^{t+M-1} \pi_j X_{t-j}, \qquad t = 1, \ldots, N.
$$

Under the ARFIMA model, $\{W_t\}$ follows a stationary zero-mean ARMA(p, q) process. The default estimator uses Eq. 9 with $M = 30$ (the value Hosking adopts after the sensitivity study in his Table 1) and the full-length filter; the two-stage estimators use Eq. 8 truncated at lag $K$.

### Preprocessing

Fitting an ARFIMA model with Gaussian innovations directly on raw streamflow produces unphysical negative simulated values, and clipping these to zero introduces upward bias in the mean and distorts the marginal distribution (Stedinger and Taylor, 1982; Salas et al., 1980). The implemented preprocessing applies a two-stage transformation so that the back-transformed synthetic series is strictly positive by construction.

**Stage 1 -- shifted-log transformation** (Stedinger and Taylor, 1982). For each calendar month $m$ a lower bound $\tau_m$ is estimated:

$$
\tau_m = \frac{q_{\max,m}\, q_{\min,m} - q_{\text{med},m}^2}{q_{\max,m} + q_{\min,m} - 2\,q_{\text{med},m}}
$$

with the safety fallback $\tau_m = 0$ when the formula yields $\tau_m < 0$ or $\tau_m \geq q_{\min,m}$ (i.e., the lognormal assumption is invalid for that month). The transformed series is:

$$
Y_t = \ln\!\left(Q_t - \tau_{m(t)}\right)
$$

**Stage 2 -- per-month z-score.** Let $\mu_m$ and $\sigma_m$ denote the sample mean and standard deviation of $Y_t$ for month $m$. The standardized residual fit by ARFIMA is:

$$
X_t = \frac{Y_t - \mu_{m(t)}}{\sigma_{m(t)}}
$$

For annual data ($_\text{is\_monthly} = \text{False}$), the same two stages are applied with a single global $\tau$, $\mu$, and $\sigma$ (no per-month branching).

### Parameter Estimation

#### Joint approximate maximum likelihood (default, `d_method='mle'`)

Hosking (1984, Sec. 4.2) estimates $d$ and the ARMA coefficients jointly. The exact Gaussian likelihood (his Eq. 5) requires inverting the $N \times N$ covariance matrix at every iteration and is impractical beyond $N \approx 100$, so the likelihood of $X$ is approximated by the likelihood of the fractionally differenced series (Eq. 6):

$$
\tilde{L}(X;\, d, \phi, \theta, \sigma_\varepsilon^2) = L_{\text{ARMA}}(W(d);\, \phi, \theta, \sigma_\varepsilon^2), \qquad W_t(d) = \nabla_M^d X_t .
$$

The dependence on $d$ enters only through the differenced series $W(d)$, and for a given $W$ the ARMA likelihood can be evaluated by any standard algorithm. The Jacobian of $X \to W$ is ignored (it is close to one unless $d$ is near $0.5$ and is asymptotically negligible).

The implementation evaluates $\tilde{L}$ as follows. Let $X_t$ be the standardized series with its sample mean removed ($\hat{\mu}$ is the sample mean rather than a free likelihood parameter; the per-period standardization already centres $X_t$).

1. **Backcasting (Eq. 9).** An AR($M$) model is fitted to $X_t$ by Yule-Walker, and the time-reversed series is forecast $M$ steps ahead to obtain $\hat{X}_{-M}, \ldots, \hat{X}_{-1}$ (Box and Jenkins, 1976, p. 199; a stationary Gaussian process is time-reversible). $M$ = `backcast_length` (default 30, reduced to $\lfloor N/4 \rfloor$ for short records; `backcast_length=0` gives Eq. 8). The backcasts do not depend on $d$ and are computed once.
2. **Fractional differencing.** For a candidate $d$, $W_t(d)$ is computed from the extended series $(\hat{X}_{-M}, \ldots, \hat{X}_{-1}, X_1, \ldots, X_N)$ with the full-length $\pi_j(d)$ filter (no truncation at $K$).
3. **ARMA innovations.** The one-step innovations are obtained from the recursion with zero presample values,

$$
e_t = W_t - \sum_{k=1}^{p} \phi_k W_{t-k} - \sum_{j=1}^{q} \theta_j e_{t-j},
$$

and the ARMA likelihood is replaced by its conditional-sum-of-squares (CSS) form, so that maximizing $\tilde{L}$ is equivalent to

$$
(\hat{d}, \hat{\phi}, \hat{\theta}) = \arg\min_{d, \phi, \theta} \; \sum_{t = p+1}^{N} e_t(d, \phi, \theta)^2 .
$$

4. **Optimisation.** L-BFGS-B over $d \in$ `d_bounds` (default $(-0.49, 0.49)$, the stationary-invertible range of Hosking's model; use $(0.01, 0.49)$ to force persistence) and coefficients in $(-0.99, 0.99)$, with candidate ARMA polynomials whose roots lie on or inside the unit circle rejected. The search starts from the profile Whittle estimate of $d$ (below) and from $d_0 \in \{0.1, 0.3\}$, with ARMA coefficients at zero, and keeps the best local optimum. The innovation variance $\hat{\sigma}_\varepsilon^2$ is the variance of $e_t$, $t > \max(p, q, 1)$, at the optimum.

The ARMA part is estimated by CSS rather than the exact ARMA likelihood (Hosking used Ansley's, 1979, algorithm). The two are asymptotically equivalent and Hosking notes (Sec. 4.3) that the approximate-versus-exact differences he observed were "the same as, or perhaps slightly larger than, one observes between the conditional least squares and exact ML methods of estimating an ARMA(p, q) model".

**Behaviour on simulated data.** The table below reports the mean (standard deviation) of $\hat{d}$ and of the ARMA coefficient over 30 exact ARFIMA samples of length $N = 600$ (generated by Cholesky factorisation of the autocovariance computed from Hosking's Eq. 3), for the default joint estimator and for the two-stage Whittle procedure (`d_method='whittle'`):

| Model | Two-stage $\hat{d}$ | Two-stage coef. | Joint $\hat{d}$ | Joint coef. |
|---|---|---|---|---|
| ARFIMA(1, 0.20, 0), $\phi = 0.5$ | 0.490 (0.000) | 0.20 (0.03) | 0.116 (0.135) | 0.56 (0.12) |
| ARFIMA(1, 0.20, 0), $\phi = -0.3$ | 0.029 (0.024) | -0.14 (0.03) | 0.175 (0.060) | -0.29 (0.07) |
| ARFIMA(1, 0.35, 0), $\phi = 0.5$ | 0.490 (0.000) | 0.34 (0.03) | 0.265 (0.144) | 0.56 (0.13) |
| ARFIMA(1, 0.35, 0), $\phi = -0.3$ | 0.172 (0.037) | -0.13 (0.03) | 0.327 (0.058) | -0.29 (0.06) |
| ARFIMA(0, 0.20, 1), $\theta = 0.5$ | 0.474 (0.017) | 0.30 (0.05) | 0.185 (0.049) | 0.49 (0.05) |
| ARFIMA(0, 0.20, 1), $\theta = -0.3$ | 0.019 (0.016) | -0.11 (0.03) | 0.161 (0.078) | -0.27 (0.08) |
| ARFIMA(0, 0.35, 1), $\theta = 0.5$ | 0.490 (0.000) | 0.40 (0.04) | 0.334 (0.050) | 0.49 (0.05) |
| ARFIMA(0, 0.35, 1), $\theta = -0.3$ | 0.148 (0.038) | -0.09 (0.02) | 0.316 (0.076) | -0.28 (0.08) |

The two-stage estimate of $d$ absorbs the short-memory structure (it hits the upper bound for a positive AR or MA root and collapses towards zero for a negative root) and the ARMA coefficient fitted to the residual is correspondingly shrunk. The joint estimator recovers both parameters. The one case with a noticeable remaining bias, ARFIMA(1, d, 0) with $\phi = 0.5$, reflects the weak separate identifiability of $d$ and a positive AR(1) root at this sample size ($\hat{d} + \hat{\phi}$ is well determined while the split is not): the exact Gaussian maximum likelihood estimator evaluated on the same samples gives $\hat{d} = 0.135$ (sd 0.11) for $d = 0.2$, so this is a property of the likelihood, not of the approximation.

#### Order selection

When `auto_order=True`, the joint fit is repeated for every $(p, q) \in \{0, 1, 2\}^2$ and the order minimising an information criterion is selected. Hosking (Sec. 5.1) uses

$$
\text{AIC} = -2 \ln L_{\max} + 2 \ln L_0 + 2\,(p + q + \delta_d),
$$

where $L_0$ is the likelihood of a white-noise model and $\delta_d = 1$ if $d$ is estimated. With Gaussian innovations $-2 \ln L_{\max} = n_{\text{eff}} \ln \hat{\sigma}_\varepsilon^2 + \text{const}$, and the $L_0$ term is common to all candidates, so the implemented criterion is

$$
\text{AIC} \doteq n_{\text{eff}} \ln \hat{\sigma}_\varepsilon^2 + 2\,(p + q + 1), \qquad
\text{BIC} \doteq n_{\text{eff}} \ln \hat{\sigma}_\varepsilon^2 + (p + q + 1) \ln n_{\text{eff}},
$$

with $\hat{\sigma}_\varepsilon^2$ evaluated on the common sample $t \geq 3$ ($n_{\text{eff}} = N - 2$) so that all candidates are compared on the same residuals. `order_criterion='aic'` (default) follows Hosking; `'bic'` is available because it is consistent for ARFIMA order selection (Huang et al., 2022). $d$ is estimated in every candidate, so $\delta_d$ is constant across the grid; models with $d$ fixed at zero are not included in the search.

#### Two-stage estimators (`d_method='whittle'`, `'gph'`, `'rs'`)

These estimate $d$ from the series alone, then apply Eq. 8 truncated at $K$ and fit the ARMA part to the residual $W_t$ (Yule-Walker for pure AR, CSS for mixed ARMA, with the same order-selection criterion applied to the differenced series when `auto_order=True`). They are retained for comparison and for the `whittle` starting value of the joint fit; as the table above shows, their $\hat{d}$ is contaminated whenever $p + q > 0$ (Hosking's Table 4 illustrates the same effect on the Nile flows).

**Profile Whittle** (Fox and Taqqu, 1986). With $g(\omega;\,d) = [2(1 - \cos \omega)]^{-d}$, the ARFIMA(0, d, 0) spectral shape, and the scale profiled out,

$$
\hat{d} = \arg\min_{d \in [0.01, 0.49]} \left\{ m \ln\!\left[\frac{1}{m}\sum_{j=1}^{m} \frac{I(\omega_j)}{g(\omega_j;\,d)}\right] + \sum_{j=1}^{m} \ln g(\omega_j;\,d) \right\},
$$

where $I(\omega_j)$ is the periodogram at the $m = \lfloor N/2 \rfloor - 1$ Fourier frequencies. The profile form is invariant to the scale of $X_t$; the non-profiled form with the scale fixed at one biases $\hat{d}$ upward by $+0.04$ to $+0.06$.

**GPH log-periodogram regression** (Geweke and Porter-Hudak, 1983). OLS of $\ln I(\omega_j)$ on $-\ln[2(1 - \cos \omega_j)]$ over the $\lfloor\sqrt{N}\rfloor$ lowest Fourier frequencies. The implementation regresses on $+\ln[2(1 - \cos \omega_j)]$ and negates the slope, which is equivalent.

**Rescaled range (R/S).** $\hat{d} = \hat{H} - 0.5$ with $\hat{H}$ from a log-linear regression of the R/S statistic on subsample size.

### Synthesis Procedure

1. Estimate $(\hat{d}, \hat{\phi}, \hat{\theta}, \hat{\sigma}_\varepsilon^2)$ as described above.
2. Compute the inverse fractional differencing coefficients $\{\psi_k\}$, $k = 0, \ldots, K$.
3. Generate $n + n_{\text{burn}}$ synthetic ARMA innovations and the differenced series, where $n_{\text{burn}} = K + \max(50, 10\max(p, q))$:

$$
\hat{W}_t = \sum_{k=1}^{p} \hat{\phi}_k \hat{W}_{t-k} + \sum_{j=1}^{q} \hat{\theta}_j \varepsilon_{t-j} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \hat{\sigma}_\varepsilon^2)
$$

4. Invert the fractional differencing via a finite impulse response (FIR) convolution:

$$
\hat{X}_t = \sum_{k=0}^{K} \psi_k\,\hat{W}_{t-k}
$$

where the inverse coefficients are:

$$
\psi_0 = 1, \qquad \psi_k = \psi_{k-1} \cdot \frac{k - 1 + d}{k}, \quad k \geq 1
$$

5. Discard the first $n_{\text{burn}}$ values of $\hat{X}_t$. The ARMA recursion starts from $\hat{W}_0 = \varepsilon_0$ and the FIR filter only reaches its full length at $t = K$, so without burn-in the variance of the first output step is $\sigma_\varepsilon^2$ rather than $\sigma_\varepsilon^2 \sum_k \psi_k^2$ (about 79% of the steady value at $d = 0.3$, 70% at $d = 0.35$). The extra draws come from the same seeded generator, so results remain reproducible for a given `seed`.
6. Reverse the standardization in log space: $\hat{Y}_t = \hat{X}_t \cdot \sigma_{m(t)} + \mu_{m(t)}$.
7. Reverse the shifted-log transformation: $\hat{Q}_t = \tau_{m(t)} + \exp(\hat{Y}_t)$.

Because $\tau_m \geq 0$ and $\exp(\hat{Y}_t) > 0$, the simulated flows are strictly positive without any hard-clipping step.

### Deviations from Hosking (1984)

**CSS in place of the exact ARMA likelihood.** Hosking evaluates $L(w; 0, \beta)$ in Eq. 6 exactly (Ansley, 1979). The implementation uses the conditional sum of squares of the ARMA innovations, i.e. it conditions on zero presample values of $W_t$ and $e_t$. The two are asymptotically equivalent; the difference is of the same order as that between CSS and exact ML for an ordinary ARMA model.

**Mean.** Hosking treats $\mu$ as a free parameter of the likelihood. The implementation removes the sample mean of the standardized series (which is zero for annual data and very close to zero for monthly data after per-month standardization).

**Order selection grid.** The search covers $(p, q) \in \{0, 1, 2\}^2$ with $d$ always estimated; Hosking additionally compared models with $d$ fixed at zero (the $\delta_d$ term in his AIC). A pure ARMA alternative must be checked by the user.

**Finite inverse filter in generation.** Hosking's exact simulation (Sec. 3, algorithm A) uses the Cholesky factor of the full autocovariance matrix, and his faster alternative (Eq. 4) filters an exact ARFIMA(0, d, 0) sample through the ARMA recursion. The implementation generates ARMA innovations first and inverts the fractional differencing with a FIR filter of length $K$ (`truncation_lag`, default 100). The truncation caps the simulated variance of $X_t$ at $\sigma_\varepsilon^2 \sum_{k=0}^{K} \psi_k^2$ instead of the exact $\sigma_\varepsilon^2\,\Gamma(1-2d)/\Gamma(1-d)^2$. With $K = 100$ the ratio is 0.97 at $d = 0.3$ but only 0.55 at $d = 0.45$ (0.65 at $K = 1000$), because $\psi_k \sim k^{d-1}$ decays very slowly as $d \to 0.5$. Users whose estimated $d$ exceeds about 0.4 should raise `truncation_lag` substantially (and accept the corresponding $O(nK)$ cost); the burn-in length grows with $K$ automatically.

**Fit-side truncation in the two-stage estimators.** The `whittle`, `gph` and `rs` paths difference with Eq. 8 truncated at $K$ and without backcasting; the default joint estimator uses Eq. 9 with the full-length filter.

## Statistical Properties

The ARFIMA model directly parameterizes long-range dependence through $d$, reproducing the hyperbolic decay of the autocorrelation function $\rho(k) \sim C k^{2d-1}$ as $k \to \infty$ and the spectral divergence $f(\omega) \sim C' \omega^{-2d}$ near the origin. The short-memory ARMA component captures structure at lags $1$ through $p$ and the moving-average smoothing at lags $1$ through $q$.

Monthly means and standard deviations are preserved through the per-month log-space standardization. The shifted-lognormal preprocessing also preserves the positive support of streamflow and accommodates marginal skewness commonly observed in monthly flow records. The model assumes Gaussian innovations in log space, which may underrepresent extreme events or heavy-tailed behavior. Spatial dependence is not modeled. Truncation of the inverse filter at lag $K$ in generation introduces approximation error that grows as $d$ approaches $0.5$ (see *Deviations from Hosking (1984)* for numbers).

## Limitations

- Univariate only; must be combined with spatial methods for multisite applications.
- Reliable estimation of $d$ requires long records (50+ years recommended).
- Gaussian innovation assumption may underrepresent tail behavior.
- Truncation of the fractional differencing series at finite $K$ is an approximation; it under-reproduces variance for $d > 0.4$ unless `truncation_lag` is raised (see *Deviations*).
- $d$ and a positive low-order AR root are weakly separately identified in samples of a few hundred values; the joint estimate of $d$ can then be biased low with a large standard error (see the simulation table), which is a property of the likelihood rather than of the estimator.
- The two-stage `whittle`, `gph` and `rs` options estimate $d$ ignoring the ARMA part and are contaminated by short-memory structure; they are kept for comparison only.
- CSS estimation exhibits known small-sample bias relative to exact ML.

## References

**Primary:**
Hosking, J.R.M. (1984). Modeling persistence in hydrological time series using fractional differencing. *Water Resources Research*, 20(12), 1898-1908. https://doi.org/10.1029/WR020i012p01898

**See also:**
- Granger, C.W.J., and Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. *Journal of Time Series Analysis*, 1(1), 15-29. https://doi.org/10.1111/j.1467-9892.1980.tb00297.x
- Geweke, J., and Porter-Hudak, S. (1983). The estimation and application of long memory time series models. *Journal of Time Series Analysis*, 4(4), 221-238. https://doi.org/10.1111/j.1467-9892.1983.tb00371.x
- Box, G.E.P., and Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control* (revised ed.). Holden-Day.
- Ansley, C.F. (1979). An algorithm for the exact likelihood of a mixed autoregressive-moving average process. *Biometrika*, 66(1), 59-65. https://doi.org/10.1093/biomet/66.1.59
- Huang, H.-H., Chan, N.H., Chen, K., and Ing, C.-K. (2022). Consistent order selection for ARFIMA processes. *The Annals of Statistics*, 50(3), 1297-1319. https://doi.org/10.1214/21-AOS2149
- Fox, R., and Taqqu, M.S. (1986). Large-sample properties of parameter estimates for strongly dependent stationary Gaussian time series. *The Annals of Statistics*, 14(2), 517-532.
- Montanari, A., Rosso, R., and Taqqu, M.S. (1997). Fractionally differenced ARIMA models applied to hydrologic time series. *Water Resources Research*, 33(5), 1035-1044. https://doi.org/10.1029/97WR00043
- Koutsoyiannis, D. (2002). The Hurst phenomenon and fractional Gaussian noise made easy. *Hydrological Sciences Journal*, 47(4), 573-595.
- Stedinger, J.R., and Taylor, M.R. (1982). Synthetic streamflow generation: 1. Model verification and validation. *Water Resources Research*, 18(4), 909-918. https://doi.org/10.1029/WR018i004p00909
- Salas, J.D., Delleur, J.W., Yevjevich, V., and Lane, W.L. (1980). *Applied Modeling of Hydrologic Time Series*. Water Resources Publications.

---

**Implementation:** `src/synhydro/methods/generation/parametric/arfima.py`
