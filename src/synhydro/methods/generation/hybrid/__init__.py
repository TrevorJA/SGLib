"""
Hybrid (semi-parametric) generation methods for SynHydro.

Generators in this module combine parametric and non-parametric components in
the synthesis path. Examples include parametric pre-whitening followed by
non-parametric bootstrap of residuals (Kirsch), and parametric marginal
distributions combined with non-parametric spectral or state-conditional
resampling (Phase Randomization, HMM-KNN, WARM).

References
----------
Studnicka, S. and Panu, U.S. (2025). Techniques and Developments in Stochastic
Streamflow Synthesis-A Comprehensive Review. Encyclopedia, 5, 198.
"""

from synhydro.methods.generation.hybrid.hmm_knn import HMMKNNGenerator
from synhydro.methods.generation.hybrid.kirsch import KirschGenerator
from synhydro.methods.generation.hybrid.multisite_phase_randomization import (
    MultisitePhaseRandomizationGenerator,
)
from synhydro.methods.generation.hybrid.phase_randomization import (
    PhaseRandomizationGenerator,
)
from synhydro.methods.generation.hybrid.warm import WARMGenerator

__all__ = [
    "HMMKNNGenerator",
    "KirschGenerator",
    "MultisitePhaseRandomizationGenerator",
    "PhaseRandomizationGenerator",
    "WARMGenerator",
]
