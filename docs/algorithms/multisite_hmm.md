# Multi-Site Hidden Markov Model (Gold et al., 2024)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual |
| **Sites** | Multisite |

## Overview

The Multi-Site HMM generates synthetic streamflow across multiple sites by coupling a hidden Markov chain of discrete hydrologic regimes (e.g., dry and wet states) with state-specific multivariate Gaussian emission distributions. Temporal persistence arises from the Markov transition structure, while spatial dependence is captured by the full covariance matrix within each state. The model is well suited for representing drought dynamics and spatially compounding water scarcity, where distinct hydroclimatic regimes produce qualitatively different joint flow distributions.

## Notation

| Symbol | Description |
|--------|-------------|
| $\mathbf{Q}_t \in \mathbb{R}^S$ | Observed annual flow vector at time $t$ across $S$ sites |
| $\hat{\mathbf{Q}}_t$ | Synthetic annual flow vector at time $t$ |
| $\mathbf{Y}_t \in \mathbb{R}^S$ | Log-transformed flow vector, $Y_{t,s} = \ln(Q_{t,s} + \delta)$ |
| $K$ | Number of hidden states |
| $s_t \in \{1, \ldots, K\}$ | Hidden state at time $t$ |
| $\boldsymbol{\mu}_k \in \mathbb{R}^S$ | Emission mean vector for state $k$ |
| $\boldsymbol{\Sigma}_k \in \mathbb{R}^{S \times S}$ | Emission covariance matrix for state $k$ |
| $\mathbf{A} \in \mathbb{R}^{K \times K}$ | Transition probability matrix, $A_{ij} = P(s_{t+1} = j \mid s_t = i)$ |
| $\boldsymbol{\pi} \in \mathbb{R}^K$ | Stationary distribution of the Markov chain |
| $\delta$ | Additive offset before log transform (default 1.0) |
| $N$ | Number of years in the historical record |

## Formulation

### Model Structure

The model defines a discrete-time hidden Markov process with $K$ states. The state sequence $\{s_t\}$ evolves according to the transition matrix $\mathbf{A}$:

$$
P(s_{t+1} = j \mid s_t = i) = A_{ij}, \qquad \sum_{j=1}^{K} A_{ij} = 1 \quad \forall\, i
$$

Conditional on the hidden state $s_t = k$, the log-transformed flow vector is drawn from a multivariate Gaussian:

$$
\mathbf{Y}_t \mid s_t = k \;\sim\; \mathcal{N}(\boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k)
$$

States are ordered by the mean of the first site so that state 1 corresponds to the driest regime and state $K$ to the wettest.

### Parameter Estimation

All parameters $\{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \mathbf{A}\}$ are estimated jointly via the Baum-Welch (Expectation-Maximization) algorithm on the log-transformed observations $\{\mathbf{Y}_t\}$:

- **E-step**: Compute state posterior probabilities using the forward-backward algorithm.
- **M-step**: Update emission means, covariances, and transition probabilities to maximize the expected log-likelihood.

The covariance structure may be full ($S(S+1)/2$ free parameters per state), diagonal ($S$ parameters per state), spherical (1 parameter per state), or tied (a single full matrix, $S(S+1)/2$ parameters, shared by all states). Full covariance preserves inter-site spatial correlations within each regime.

### EM Convergence and Restarts

EM is a local optimizer: the fitted parameters depend on the random initialization, and different starts can converge to different local optima with noticeably different log-likelihoods. The implementation therefore supports `n_init` random restarts. Each restart draws its own hmmlearn seed from a NumPy generator seeded by `random_state`, so a given `random_state` always reproduces the same sequence of fits; the restart with the highest log-likelihood is retained.

After fitting, the generator stores `log_likelihood_` (best fit), `log_likelihoods_` (one entry per restart, `nan` for a failed restart), and `converged_`. A `UserWarning` is emitted if the retained fit did not meet hmmlearn's log-likelihood tolerance within `max_iterations`, and a second `UserWarning` is emitted if the restarts' log-likelihoods span more than 1 nat, which indicates that EM found several distinct optima and `n_init` should be increased. The default `n_init=1` preserves the original single-fit behaviour; `n_init` of 5 to 10 is recommended for production use.

### Stationary Distribution

The stationary distribution $\boldsymbol{\pi}$ satisfies $\boldsymbol{\pi}^\top \mathbf{A} = \boldsymbol{\pi}^\top$ with $\sum_k \pi_k = 1$. It is computed as the normalized eigenvector of $\mathbf{A}^\top$ corresponding to eigenvalue 1:

$$
\mathbf{A}^\top \boldsymbol{\pi} = \boldsymbol{\pi}
$$

and provides the initial state distribution for generation.

### Synthesis Procedure

1. Draw the initial state from the stationary distribution: $s_1 \sim \text{Categorical}(\boldsymbol{\pi})$.
2. For each subsequent year $t = 2, \ldots, T$, sample the next state from the transition row:

$$
s_t \sim \text{Categorical}(\mathbf{A}_{s_{t-1}, \cdot})
$$

3. For each year $t$, draw the log-transformed flow vector from the state-specific emission:

$$
\mathbf{Y}_t \sim \mathcal{N}(\boldsymbol{\mu}_{s_t},\, \boldsymbol{\Sigma}_{s_t})
$$

4. Back-transform: $\hat{Q}_{t,s} = \exp(Y_{t,s}) - \delta$, then enforce non-negativity.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_states` | 2 | Number of hidden states $K$ (state 0 is the driest). |
| `offset` | 1.0 | Additive offset $\delta$ applied before the log transform. |
| `max_iterations` | 1000 | Maximum EM iterations per restart; a warning is raised if this is reached without convergence. |
| `n_init` | 1 | Number of random EM restarts; the highest log-likelihood fit is kept. |
| `covariance_type` | `'full'` | Emission covariance structure: `'full'`, `'diag'`, `'spherical'`, or `'tied'`. |
| `fit(random_state=...)` | `None` | Seed controlling all restart initializations for reproducible fitting. |

## Statistical Properties

The model captures regime-dependent multivariate distributions, preserving the distinct mean levels, variances, and spatial correlation structures associated with each hydroclimatic state. Temporal persistence is governed by the diagonal elements of the transition matrix (state self-transition probabilities), which determine the expected duration of dry and wet regimes.

Spatial correlations are preserved within each state through the full covariance matrices. However, the marginal distributions of the overall mixture are constrained to be a weighted sum of log-normals, which may not match the true marginals exactly. Higher-order temporal dependence (lag $> 1$) is not explicitly modeled; any apparent longer-memory structure arises solely from the state persistence. The model assumes stationarity of the transition dynamics.

## Scope and Departures from the Paper

- **Annual HMM only.** Gold et al. (2024) follow the annual multi-site HMM with a KNN-based spatial and annual-to-monthly disaggregation step. That disaggregation is not part of this generator, which returns annual multisite flows only. For temporal disaggregation of the annual output use `NowakDisaggregator`.
- **Viterbi classification not exposed.** The paper classifies historical years into dry/wet states with the Viterbi algorithm for diagnostic plots. This generator fits the HMM and samples from it but does not expose a decoded historical state sequence.
- **Back-transform.** The generator uses $\exp(Y) - \delta$, which exactly inverts the forward transform $\ln(Q + \delta)$; the authors' reference script applies $\exp(Y)$ without subtracting the offset.
- **Emission model.** One multivariate Gaussian per state (an hmmlearn `GaussianHMM`, equivalent to a mixture with a single component). Earlier docstrings described this as a "Gaussian Mixture Model HMM"; no multi-component mixture is fit.

## Limitations

- First-order Markov assumption; multi-year drought persistence depends entirely on state self-transition probabilities.
- Full covariance estimation becomes expensive and data-hungry for more than roughly 20 sites.
- EM algorithm may converge to local optima; results can vary across random initializations. Use `n_init > 1` and inspect `log_likelihoods_`.
- Requires approximately 20+ years for 2 states and 50+ years for additional states.
- Log-normal emissions may not capture heavy-tailed or multi-modal within-state behavior.

## References

**Primary:**
Gold, D.F., Gupta, R.S., and Reed, P.M. (2024). Exploring the spatially compounding multi-sectoral drought vulnerabilities in Colorado's West Slope river basins. *Earth's Future*. https://doi.org/10.1029/2024EF004841

**See also:**
- Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
- Akintug, B., and Rasmussen, P.F. (2005). A Markov switching model for annual hydrologic time series. *Water Resources Research*, 41(9).

---

**Implementation:** `src/synhydro/methods/generation/parametric/multisite_hmm.py`
