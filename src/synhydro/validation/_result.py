"""
Result container for the validation suite.
"""

from dataclasses import dataclass

from synhydro._evaluation import EvaluationResult


@dataclass
class ValidationResult(EvaluationResult):
    """
    Result of a validation run (fit-for-purpose evaluation).

    Holds the tidy per-realization metric values produced by
    :func:`synhydro.validation.validate`. See :meth:`to_dataframe` for
    the tidy schema, :meth:`summary` for the per-metric comparison
    against observed, and :meth:`category_summary` for the per-category
    rollup.

    Validation, in the sense of Stedinger and Taylor (1982),
    demonstrates that generated flows reproduce characteristics not
    explicitly fit by the generator, such as drought duration and
    severity.

    References
    ----------
    Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow
    generation: 1. Model verification and validation. Water Resources
    Research, 18(4), 909-918.
    """

    _suite: str = "validation"
