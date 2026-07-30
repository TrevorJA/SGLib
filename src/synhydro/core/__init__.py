"""
SynHydro Core Module

Base classes, utilities, and infrastructure for synthetic generation.
"""

from synhydro.core.base import (
    Generator,
    GeneratorState,
    GeneratorParams,
    FittedParams,
    Disaggregator,
    DisaggregatorState,
    DisaggregatorParams,
)
from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.core.pipeline import GeneratorDisaggregatorPipeline

__all__ = [
    # Generator classes
    "Generator",
    "GeneratorState",
    "GeneratorParams",
    "FittedParams",
    # Disaggregator classes
    "Disaggregator",
    "DisaggregatorState",
    "DisaggregatorParams",
    # Ensemble
    "Ensemble",
    "EnsembleMetadata",
    # Pipeline
    "GeneratorDisaggregatorPipeline",
]
