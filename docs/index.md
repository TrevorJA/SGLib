# SynHydro

**Synthetic Generation Library** - stochastic streamflow generation for hydrologic analysis.

[![Tests](https://github.com/TrevorJA/SynHydro/actions/workflows/tests.yml/badge.svg)](https://github.com/TrevorJA/SynHydro/actions/workflows/tests.yml)
[![Docs](https://github.com/TrevorJA/SynHydro/actions/workflows/docs.yml/badge.svg)](https://github.com/TrevorJA/SynHydro/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/TrevorJA/SynHydro/blob/main/LICENSE)

SynHydro provides parametric, hybrid, and non-parametric stochastic generation methods under a unified API. All generators share the same `fit()` and `generate()` workflow. See the [Algorithms overview](algorithms/index.md) for the classification and plain-language descriptions of each class.

## Generators

| Generator | Class | Frequency | Sites | Reference |
|-----------|-------|-----------|-------|-----------|
| [`ThomasFieringGenerator`][synhydro.methods.generation.parametric.thomas_fiering.ThomasFieringGenerator] | Parametric | Monthly | Single | Thomas & Fiering (1962) |
| [`MatalasGenerator`][synhydro.methods.generation.parametric.matalas.MatalasGenerator] | Parametric | Monthly | Multi | Matalas (1967) |
| [`ARFIMAGenerator`][synhydro.methods.generation.parametric.arfima.ARFIMAGenerator] | Parametric | Monthly/Annual | Single | Hosking (1984) |
| [`SPARTAGenerator`][synhydro.methods.generation.parametric.sparta.SPARTAGenerator] | Parametric | Monthly | Multi | Tsoukalas et al. (2018) |
| [`SMARTAGenerator`][synhydro.methods.generation.parametric.smarta.SMARTAGenerator] | Parametric | Annual | Multi | Tsoukalas et al. (2018) |
| [`MultiSiteHMMGenerator`][synhydro.methods.generation.parametric.multisite_hmm.MultiSiteHMMGenerator] | Parametric | Annual | Multi | Gold et al. (2024) |
| [`KirschGenerator`][synhydro.methods.generation.hybrid.kirsch.KirschGenerator] | Hybrid | Monthly | Multi | Kirsch et al. (2013) |
| [`WARMGenerator`][synhydro.methods.generation.hybrid.warm.WARMGenerator] | Hybrid | Annual | Single | Nowak et al. (2011) |
| [`PhaseRandomizationGenerator`][synhydro.methods.generation.hybrid.phase_randomization.PhaseRandomizationGenerator] | Hybrid | Daily | Single | Brunner et al. (2019) |
| [`MultisitePhaseRandomizationGenerator`][synhydro.methods.generation.hybrid.multisite_phase_randomization.MultisitePhaseRandomizationGenerator] | Hybrid | Daily | Multi | Brunner & Gilleland (2020) |
| [`HMMKNNGenerator`][synhydro.methods.generation.hybrid.hmm_knn.HMMKNNGenerator] | Hybrid | Annual | Multi | Prairie et al. (2008) |
| [`KNNBootstrapGenerator`][synhydro.methods.generation.nonparametric.knn_bootstrap.KNNBootstrapGenerator] | Non-parametric | Daily/Monthly/Annual | Multi | Lall & Sharma (1996) |

## Quick Example

```python
import synhydro

Q_obs = synhydro.load_example_data()                       # daily DataFrame
Q_monthly = Q_obs.resample("MS").sum()                  # resample to monthly

gen = synhydro.KirschGenerator()
gen.fit(Q_monthly)
ensemble = gen.generate(n_realizations=50, n_years=30, seed=42)
```

## Installation

```bash
pip install git+https://github.com/TrevorJA/SynHydro.git
```

See [Getting Started](getting-started.md) for full setup and data format details.
