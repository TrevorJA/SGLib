"""
SGLib: Synthetic Generation Library

A library for generating synthetic hydrologic timeseries with focus on 
statistical preservation and hydrologic applications.
"""

__version__ = "0.0.1"

# core utilities
from .core.base import Generator
from .utils.load import load_drb_reconstruction, HDF5Manager

# Only import stable, tested implementations
from .methods.nonparametric.kirsch_nowak import KirschNowakGenerator
from .methods.parametric.thomas_fiering import ThomasFieringGenerator
from .methods.nonparametric.kirsch import KirschGenerator
from .methods.temporal_disaggregation.nowak import NowakDisaggregator


# public API
__all__ = [
    "KirschNowakGenerator", 
    "ThomasFieringGenerator",
    "load_drb_reconstruction",
    "KirschGenerator",
    "NowakDisaggregator",
    "Generator",
    "HDF5Manager"
]
