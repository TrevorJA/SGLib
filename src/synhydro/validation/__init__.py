"""
Validation suite: fit-for-purpose evaluation.

Validation, in the sense of Stedinger and Taylor (1982), demonstrates
that generated flows reproduce characteristics not explicitly fit by
the generator. The current suite covers drought behavior: run-theory
events below a flow threshold and events on the Standardized
Streamflow Index. Storage-yield and further run-theory statistics are
planned extensions. For statistical property preservation, see
:mod:`synhydro.verification`.

References
----------
Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow
generation: 1. Model verification and validation. Water Resources
Research, 18(4), 909-918.
"""

import pandas as pd

from synhydro.validation._result import ValidationResult
from synhydro.validation._validate import validate, CATEGORIES
from synhydro.validation.metrics.threshold_drought import (
    metric_inventory as _threshold_inventory,
)
from synhydro.validation.metrics.ssi_drought import (
    metric_inventory as _ssi_inventory,
)

__all__ = [
    "validate",
    "ValidationResult",
    "list_metrics",
    "CATEGORIES",
]


def list_metrics() -> pd.DataFrame:
    """
    List all validation metrics.

    Returns
    -------
    pd.DataFrame
        One row per metric with columns ``name, category, kind, units,
        frequencies, min_years, citation, description``.
    """
    return pd.DataFrame(_threshold_inventory() + _ssi_inventory())
