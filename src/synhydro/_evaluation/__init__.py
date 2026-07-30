"""
Shared infrastructure for the verification and validation suites.

This package is private. It provides the metric registry, frequency
handling, the evaluation runner, and the result base class used by both
``synhydro.verification`` (statistical property preservation) and
``synhydro.validation`` (fit-for-purpose evaluation).

Terminology follows Stedinger and Taylor (1982): verification demonstrates
that generated flows reproduce the statistics the generator was designed
to reproduce; validation demonstrates that characteristics not explicitly
fit (droughts, storage) are also consistent with the historical record.
"""

from synhydro._evaluation._frequency import (
    FrequencyInfo,
    normalize_frequency,
    infer_frequency,
    resolve_frequency,
    check_observed_frequency,
)
from synhydro._evaluation._registry import MetricSpec, MetricRegistry
from synhydro._evaluation._context import MetricContext
from synhydro._evaluation._runner import (
    run_metrics,
    resolve_sites,
    VALUE_COLUMNS,
    SKIP_COLUMNS,
)
from synhydro._evaluation._result import EvaluationResult
from synhydro._evaluation._stats import (
    sample_skewness,
    sample_kurtosis,
    extract_runs,
)
