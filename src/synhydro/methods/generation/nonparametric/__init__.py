"""
Non-parametric generation methods for SynHydro.

Generators in this module produce new variability by direct resampling of the
observed record, without fitting a parametric probability distribution. The
canonical example is the k-nearest-neighbour bootstrap of Lall and Sharma
(1996).

References
----------
Studnicka, S. and Panu, U.S. (2025). Techniques and Developments in Stochastic
Streamflow Synthesis-A Comprehensive Review. Encyclopedia, 5, 198.
"""

from synhydro.methods.generation.nonparametric.knn_bootstrap import (
    KNNBootstrapGenerator,
)

__all__ = [
    "KNNBootstrapGenerator",
]
