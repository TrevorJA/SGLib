"""
SynHydro: Synthetic Generation Library

A library for generating synthetic hydrologic timeseries with focus on
statistical preservation and hydrologic applications.
"""

__version__ = "0.0.2"

# Core utilities
from synhydro.core import (
    Generator,
    Disaggregator,
    Ensemble,
    EnsembleMetadata,
    GeneratorDisaggregatorPipeline,
)
from synhydro.utils import load_example_data

# Drought analysis tools
from synhydro.droughts.ssi import SSI, get_drought_metrics
from synhydro.droughts.distributions import (
    list_distributions,
    get_distribution,
    DISTRIBUTION_REGISTRY,
)
from synhydro.droughts.diagnostics import (
    compare_distributions,
    distribution_summary,
)

# Validation
from synhydro.core.validation import (
    validate_ensemble,
    ValidationResult,
    compute_realization_metrics,
    bootstrap_metric_ci,
    compare_methods,
)

# Generators
from synhydro.methods.generation.hybrid.kirsch import KirschGenerator
from synhydro.methods.generation.hybrid.multisite_phase_randomization import (
    MultisitePhaseRandomizationGenerator,
)
from synhydro.methods.generation.hybrid.phase_randomization import (
    PhaseRandomizationGenerator,
)
from synhydro.methods.generation.hybrid.warm import WARMGenerator
from synhydro.methods.generation.nonparametric.knn_bootstrap import (
    KNNBootstrapGenerator,
)
from synhydro.methods.generation.parametric.arfima import ARFIMAGenerator
from synhydro.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator
from synhydro.methods.generation.parametric.matalas import MatalasGenerator
from synhydro.methods.generation.parametric.multisite_hmm import MultiSiteHMMGenerator
from synhydro.methods.generation.parametric.smarta import SMARTAGenerator
from synhydro.methods.generation.parametric.sparta import SPARTAGenerator

# Disaggregators
from synhydro.methods.disaggregation.temporal.nowak import NowakDisaggregator
from synhydro.methods.disaggregation.temporal.valencia_schaake import (
    ValenciaSchaakeDisaggregator,
)

# Pre-built pipelines (recommended for most users)
from synhydro.pipelines import (
    KirschNowakPipeline,
    ThomasFieringNowakPipeline,
)


# Public API
__all__ = [
    # Base classes
    "Generator",
    "Disaggregator",
    # Individual generators
    "ARFIMAGenerator",
    "KirschGenerator",
    "KNNBootstrapGenerator",
    "MatalasGenerator",
    "MultiSiteHMMGenerator",
    "MultisitePhaseRandomizationGenerator",
    "PhaseRandomizationGenerator",
    "SMARTAGenerator",
    "SPARTAGenerator",
    "ThomasFieringGenerator",
    "WARMGenerator",
    # Individual disaggregators
    "NowakDisaggregator",
    "ValenciaSchaakeDisaggregator",
    # Pipeline system
    "GeneratorDisaggregatorPipeline",
    "KirschNowakPipeline",
    "ThomasFieringNowakPipeline",
    # Ensemble management
    "Ensemble",
    "EnsembleMetadata",
    # Data utilities
    "load_example_data",
    # Validation
    "validate_ensemble",
    "ValidationResult",
    "compute_realization_metrics",
    "bootstrap_metric_ci",
    "compare_methods",
    # Drought analysis - SSI calculation
    "SSI",
    "get_drought_metrics",
    # Drought analysis - distributions
    "list_distributions",
    "get_distribution",
    "DISTRIBUTION_REGISTRY",
    # Drought analysis - diagnostics
    "compare_distributions",
    "distribution_summary",
]
