"""
Generation methods for SynHydro.

Generators are organized into three top-level bins by the mathematical
character of their generative mechanism (Studnicka and Panu, 2025):

- ``parametric``: samples new variability from a fitted probability
  distribution (AR-family linear time-series, Markov chains).
- ``hybrid``: combines parametric and non-parametric components in the
  synthesis path (parametric pre-whitening with bootstrap residuals,
  parametric marginals with non-parametric spectral or state-conditional
  resampling).
- ``nonparametric``: produces new variability by direct empirical
  resampling without a fitted distribution (k-NN bootstrap).
"""

from synhydro.methods.generation.hybrid import (
    HMMKNNGenerator,
    KirschGenerator,
    MultisitePhaseRandomizationGenerator,
    PhaseRandomizationGenerator,
    WARMGenerator,
)
from synhydro.methods.generation.nonparametric import (
    KNNBootstrapGenerator,
)
from synhydro.methods.generation.parametric import (
    ARFIMAGenerator,
    MatalasGenerator,
    MultiSiteHMMGenerator,
    SMARTAGenerator,
    SPARTAGenerator,
    ThomasFieringGenerator,
)

__all__ = [
    "ARFIMAGenerator",
    "HMMKNNGenerator",
    "KirschGenerator",
    "KNNBootstrapGenerator",
    "MatalasGenerator",
    "MultisitePhaseRandomizationGenerator",
    "MultiSiteHMMGenerator",
    "PhaseRandomizationGenerator",
    "SMARTAGenerator",
    "SPARTAGenerator",
    "ThomasFieringGenerator",
    "WARMGenerator",
]
