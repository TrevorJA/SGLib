"""
Parametric generation methods for SynHydro.

Generators in this module produce new variability by sampling from a fitted
probability distribution. Marginal distributions and time-series dynamics
(AR, MAR, ARFIMA, PAR, SMA, HMM) are specified as parametric models and fit
to observed data; synthesis then draws random innovations from those fits.

References
----------
Studnicka, S. and Panu, U.S. (2025). Techniques and Developments in Stochastic
Streamflow Synthesis-A Comprehensive Review. Encyclopedia, 5, 198.
"""

from synhydro.methods.generation.parametric.arfima import ARFIMAGenerator
from synhydro.methods.generation.parametric.matalas import MatalasGenerator
from synhydro.methods.generation.parametric.multisite_hmm import MultiSiteHMMGenerator
from synhydro.methods.generation.parametric.smarta import SMARTAGenerator
from synhydro.methods.generation.parametric.sparta import SPARTAGenerator
from synhydro.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator

__all__ = [
    "ARFIMAGenerator",
    "MatalasGenerator",
    "MultiSiteHMMGenerator",
    "SMARTAGenerator",
    "SPARTAGenerator",
    "ThomasFieringGenerator",
]
