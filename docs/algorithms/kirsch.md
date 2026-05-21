# Kirsch Hybrid Bootstrap (Kirsch et al., 2013)

| | |
|---|---|
| **Type** | Hybrid |
| **Resolution** | Weekly or Monthly |
| **Sites** | Multisite |

## Overview

The Kirsch method generates synthetic multi-site streamflow by bootstrapping standardized residuals and imposing fitted intra-annual correlation structure through Cholesky decomposition. A cross-year shifted matrix construction preserves continuity across the year boundary. An optional normal score transform reduces bias when operating in log-transformed space. The method is hybrid: a parametric layer (per-period mean, standard deviation, intra-annual correlation) wraps a non-parametric bootstrap of the standardized residuals.

The published algorithm (Kirsch et al., 2013) is described on weekly timesteps with 52 columns per year. SynHydro implements both **weekly** (52 columns, 26-period half-shift) and **monthly** (12 columns, 6-period half-shift) resolutions from a single code path. The algebra is identical at both resolutions; only the period count and the half-year split change. The implementation derives the number of periods per year and the half-shift from the input data frequency in `preprocessing`, so calling code does not pick the resolution explicitly except through the input timestep.

## Notation

Let $P$ denote the number of periods per year ($P = 52$ for weekly, $P = 12$ for monthly), and let $H = P / 2$ be the half-year split point used in the cross-year shifted matrix.

| Symbol | Description |
|--------|-------------|
| $P$ | Number of periods per year ($52$ weekly, $12$ monthly) |
| $H$ | Half-year split, $H = P / 2$ |
| $Q_{y,p,s}$ | Observed flow for year $y$, period $p \in \{1, \dots, P\}$, site $s$ |
| $\hat{Q}_{y,p,s}$ | Synthetic flow |
| $\mu_{p,s}$ | Sample mean of flows at site $s$ in period $p$ |
| $\sigma_{p,s}$ | Sample standard deviation of flows at site $s$ in period $p$ |
| $Z_{y,p,s}$ | Standardized residual |
| $Y_{y,p,s}$ | Normal score-transformed residual |
| $\mathbf{Y}^{(s)} \in \mathbb{R}^{N \times P}$ | Matrix of normal scores for site $s$ (years by periods) |
| $\mathbf{Y}'^{(s)}$ | Cross-year shifted matrix for site $s$ |
| $\mathbf{R}^{(s)}, \mathbf{R}'^{(s)}$ | $P \times P$ correlation matrices of $\mathbf{Y}^{(s)}$ and $\mathbf{Y}'^{(s)}$ |
| $\mathbf{U}^{(s)}$ | Upper Cholesky factor, $\mathbf{R}^{(s)} = (\mathbf{U}^{(s)})^\top \mathbf{U}^{(s)}$ |
| $N$ | Number of complete years in the record |
| $S$ | Number of sites |

The original paper uses "$j$" for the column (period) index and assumes weekly ($P = 52$); the symbol $p$ used here is interchangeable with the paper's $j$ at either resolution.

## Formulation

### Standardization and Normal Score Transform

Flows are first (optionally) log-transformed: $Q' = \ln(\max(Q, 10^{-6}))$. Standardized residuals are computed for each year $y$, period $p$, and site $s$ (Kirsch et al., 2013, eq. 3):

$$
Z_{y,p,s} = \frac{Q'_{y,p,s} - \mu_{p,s}}{\sigma_{p,s}}
$$

When operating in log space, a normal score transform (NST) is applied to each $(p, s)$ pair. The residuals are ranked and mapped to standard normal quantiles via Hazen plotting positions:

$$
Y_{y,p,s} = \Phi^{-1}\!\left(\frac{r(Z_{y,p,s}) - 0.5}{N}\right)
$$

where $r(\cdot)$ denotes the rank among the $N$ values and $\Phi^{-1}$ is the standard normal inverse CDF.

**Note:** The normal score transform is a SynHydro-specific extension to the original Kirsch (2013) method, which only applies z-score standardization (his eq. 3). NST is added in log-space here to prevent bias in the back-transformed marginal distribution when standardized log-residuals are non-Gaussian; the inverse NST (with linear tail extrapolation) maps Cholesky-mixed values back to the empirical $(p, s)$ marginal. Set `generate_using_log_flow=False` to skip both the log transform and NST and run the algorithm closer to the published version.

### Cross-Year Shifted Matrix

To preserve inter-annual correlation across the year boundary, a shifted matrix $\mathbf{Y}'^{(s)}$ is constructed for each site $s$ by stacking the second half of year $y$ on top of the first half of year $y+1$:

$$
\mathbf{Y}'^{(s)}_{y,\,1:H} = \mathbf{Y}^{(s)}_{y,\,H+1:P}, \qquad \mathbf{Y}'^{(s)}_{y,\,H+1:P} = \mathbf{Y}^{(s)}_{y+1,\,1:H}
$$

For weekly ($H = 26$) this pairs weeks 27-52 of year $y$ with weeks 1-26 of year $y+1$. For monthly ($H = 6$) it pairs July-December of year $y$ with January-June of year $y+1$. Either way the Cholesky factor of the shifted matrix captures correlations that straddle the calendar boundary.

### Cholesky Decomposition

For each site $s$, the $P \times P$ sample correlation matrices $\mathbf{R}^{(s)}$ and $\mathbf{R}'^{(s)}$ are computed from $\mathbf{Y}^{(s)}$ and $\mathbf{Y}'^{(s)}$ respectively. If either matrix is not positive definite, it is repaired (default: spectral projection — negative eigenvalues are clipped to a small positive constant and the matrix is rescaled to unit diagonal). The upper Cholesky factors $\mathbf{U}^{(s)}$ and $\mathbf{U}'^{(s)}$ are then computed.

Per Kirsch et al. (2013, eqs. 4-5), the correlation matrix and its Cholesky factor are intra-annual operators defined for a single site at a time. Cross-site correlation is preserved through the **shared bootstrap index matrix $\mathbf{M}$** (Kirsch et al., 2013, p. 7), not through a joint multi-site correlation matrix. This implementation therefore computes per-site Cholesky factors $\mathbf{U}^{(s)}, \mathbf{U}'^{(s)}$ and reuses one $\mathbf{M}$ across all sites in the synthesis step below.

### Synthesis Procedure

1. **Bootstrap**: For each synthetic realization, draw a single matrix $\mathbf{M} \in \{1, \dots, N\}^{(N_{\text{syn}}+1) \times P}$ of year indices sampled with replacement from the historical record. The same $\mathbf{M}$ is reused across all sites so cross-site correlation is preserved (Kirsch et al., 2013, p. 7). Construct the bootstrap matrix $\mathbf{X}^{(s)}$ by extracting the normal scores of site $s$ at the sampled indices. Construct the corresponding shifted matrix $\mathbf{X}'^{(s)}$ **deterministically** from $\mathbf{X}^{(s)}$ using the same half-year shift that produced $\mathbf{Y}'^{(s)}$ from $\mathbf{Y}^{(s)}$:

$$
\mathbf{X}'^{(s)}_{y,\,1:H} = \mathbf{X}^{(s)}_{y,\,H+1:P}, \qquad \mathbf{X}'^{(s)}_{y,\,H+1:P} = \mathbf{X}^{(s)}_{y+1,\,1:H}
$$

Following Kirsch et al. (2013, p. 6): "The matrix X is converted to X' just as Y was converted to Y'". There is exactly one bootstrap draw per realization; $\mathbf{X}'^{(s)}$ is not resampled independently.

2. **Impose correlation**: Multiply each bootstrap matrix by its Cholesky factor (Kirsch et al., 2013, eq. 5):

$$
\tilde{\mathbf{Z}}^{(s)} = \mathbf{X}^{(s)} \mathbf{U}^{(s)}, \qquad \tilde{\mathbf{Z}}'^{(s)} = \mathbf{X}'^{(s)} \mathbf{U}'^{(s)}
$$

3. **Combine shifted and unshifted results**: The final correlated matrix $\mathbf{Z}_C$ takes the first half of each year from the shifted result and the second half from the unshifted result of the following year:

$$
\mathbf{Z}_{C,\,y,\,1:H,\,s} = \tilde{\mathbf{Z}}'_{y,\,H+1:P,\,s}, \qquad \mathbf{Z}_{C,\,y,\,H+1:P,\,s} = \tilde{\mathbf{Z}}_{y+1,\,H+1:P,\,s}
$$

This stitch is what gives the cross-year boundary correlation: each row of $\mathbf{Z}_C$ inherits its first half from $\tilde{\mathbf{Z}}'$ (which was correlated across the calendar boundary) and its second half from $\tilde{\mathbf{Z}}$ (which was correlated within the standard year).

4. **Inverse normal score transform**: Map back from normal space to the original residual space using the stored rank mappings, with linear extrapolation in the tails for values outside the historical range.

5. **Destandardize and back-transform**:

$$
\hat{Q}'_{y,\,p,\,s} = Z_{C,\,y,\,p,\,s} \cdot \sigma_{p,s} + \mu_{p,s}, \qquad \hat{Q}_{y,\,p,\,s} = \exp(\hat{Q}'_{y,\,p,\,s})
$$

### Output indexing

The synthetic ensemble's DatetimeIndex is constructed so that ordinal position $p$ of synthetic year $y$ lands on a date that falls in calendar period $p$ of $y$:

- **Monthly** ($P = 12$): a `MS`-frequency range produces month-start timestamps that already align period $p$ with calendar month $p$.
- **Weekly** ($P = 52$): each (year, period) pair is mapped to the Sunday of ISO week $p$ of year $y$ via `pd.Timestamp.fromisocalendar(y, p, 7)`. This per-year reset prevents the ~1.25 day/year drift that an unanchored `pd.date_range(freq="W-SUN")` would accumulate (since $52 \times 7 = 364 < 365$ days). ISO week 53 is treated as out-of-domain at both fit and generate time, so each retained year has exactly $P = 52$ periods.

## Statistical Properties

The method preserves per-period (monthly or weekly) means and standard deviations at each site (by construction through standardization and destandardization), spatial cross-site correlations (all sites share the same bootstrap indices for each year), and intra-annual temporal correlation (through the Cholesky decomposition of the $P \times P$ correlation matrix). The cross-year shifted matrix construction maintains continuity across the year boundary.

Because the method resamples from the historical record, the empirical marginal distribution is approximately preserved. The normal score transform and its inverse allow modest extrapolation beyond the observed range in the tails. However, generated values remain close to the historical envelope, and genuinely novel extremes cannot be produced.

## Limitations

- Generated values are bounded near the historical range (bootstrap limitation).
- Requires complete years; the cross-year shifted matrix loses one year of data, and the weekly path additionally drops ISO week 53 from years that have one.
- Sample correlation matrices may need positive-definiteness repair, which can inflate apparent correlations.
- The method does not model long-range persistence or nonstationarity.

## References

**Primary:**
Kirsch, B.R., Characklis, G.W., and Zeff, H.B. (2013). Evaluating the impact of alternative hydro-climate scenarios on transfer agreements: A practical improvement for generating synthetic streamflows. *Journal of Water Resources Planning and Management*, 139(4), 396-406. https://doi.org/10.1061/(ASCE)WR.1943-5452.0000287

---

**Implementation:** `src/synhydro/methods/generation/hybrid/kirsch.py`
