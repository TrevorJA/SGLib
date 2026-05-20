# Algorithms

SynHydro's generators are classified into three bins by the mathematical character of their generative mechanism. The classification follows the parametric / non-parametric distinction set out by Studnicka and Panu (2025), with the third "hybrid" bin reserved for methods whose synthesis path combines both parametric and non-parametric components.

## Generator classification

### Parametric

Fit a probability model (AR, MAR, ARFIMA, PAR, SMA, HMM) to the historical record, then synthesize new flows by drawing random innovations from the fitted model. The generative step samples from a fitted distribution. Parametric generators are interpretable and data-efficient, but bound by the assumed model structure.

| Algorithm | Resolution | Sites |
|-----------|------------|-------|
| [Thomas-Fiering AR(1)](thomas_fiering.md) | Monthly | Univariate |
| [Matalas MAR(1)](matalas.md) | Monthly | Multisite |
| [ARFIMA](arfima.md) | Monthly/Annual | Univariate |
| [SPARTA](sparta.md) | Monthly | Multisite |
| [SMARTA](smarta.md) | Annual | Multisite |
| [Multi-Site HMM](multisite_hmm.md) | Annual | Multisite |

### Hybrid

Combine a parametric structure (standardization, marginal distributions, regime states, AR-per-band) with a non-parametric resampling step (bootstrap, k-NN, phase shuffle, wavelet decomposition). The parametric layer enforces statistical properties such as monthly moments, intra-annual correlation, or marginal shape; the non-parametric layer preserves empirical detail the parametric layer would smooth away.

| Algorithm | Parametric component | Non-parametric component | Resolution | Sites |
|-----------|----------------------|--------------------------|------------|-------|
| [Kirsch Bootstrap](kirsch.md) | Monthly mean/std, intra-annual Cholesky correlation | Bootstrap of standardized residuals | Monthly | Multisite |
| [WARM](warm.md) | AR(p) per spectral band | Continuous wavelet decomposition | Annual | Univariate |
| [Phase Randomization](phase_randomization.md) | Four-parameter kappa marginal per day-of-year | FFT phase shuffle | Daily | Univariate |
| [Multisite Phase Randomization](multisite_phase_randomization.md) | Per-site kappa marginals per day-of-year | Wavelet CWT phase shuffle (shared across sites) | Daily | Multisite |
| [HMM-KNN](hmm_knn.md) | Hidden Markov regime sequencer | k-NN resample within state | Annual | Multisite |

### Non-parametric

Generate new flows by direct resampling of the historical record with no fitted probability distribution. The generative step is empirical. Non-parametric generators preserve the empirical distribution and complex nonlinear dependence structures by construction; the trade-off is that they cannot extrapolate beyond observed values and need an adequate historical record.

| Algorithm | Resolution | Sites |
|-----------|------------|-------|
| [KNN Bootstrap](knn_bootstrap.md) | Daily/Monthly/Annual | Univariate/Multisite |

## Disaggregation Methods

| Algorithm | Type | Resolution |
|-----------|------|------------|
| [Nowak KNN](nowak_disaggregation.md) | Non-parametric | Monthly to Daily |
| [Valencia-Schaake](valencia_schaake.md) | Parametric | Annual to Monthly |

## Key Properties Preserved

| Property | Thomas-Fiering | Matalas | ARFIMA | SMARTA | SPARTA | MS-HMM | Kirsch | WARM | Phase Random | HMM-KNN | KNN-Bootstrap |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Monthly means/stds | x | x | x | - | x | - | x | - | - | - | x |
| Temporal correlation | x | x | x | x | x | x | x | x | x | x | x |
| Spatial correlation | - | x | - | x | x | x | x | - | - | x | x |
| Long-range persistence | - | - | x | x | - | - | - | x | - | - | - |
| Non-stationarity | - | - | - | - | - | - | - | x | - | - | - |
| Drought states | - | - | - | - | - | x | - | - | - | x | - |
| Power spectrum | - | - | x | - | - | - | - | x | x | - | - |
| Arbitrary marginals | - | - | - | x | x | - | - | - | x | - | - |
| Empirical distribution | - | - | - | - | - | - | x | - | - | x | x |

## Reference

Studnicka, S. and Panu, U.S. (2025). Techniques and Developments in Stochastic Streamflow Synthesis: A Comprehensive Review. *Encyclopedia*, 5, 198. [https://doi.org/10.3390/encyclopedia5040198](https://doi.org/10.3390/encyclopedia5040198)
