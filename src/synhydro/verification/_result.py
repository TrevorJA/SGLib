"""
Result container for the verification suite.
"""

from dataclasses import dataclass

from synhydro._evaluation import EvaluationResult


@dataclass
class VerificationResult(EvaluationResult):
    """
    Result of a verification run (statistical property preservation).

    Holds the tidy per-realization metric values produced by
    :func:`synhydro.verification.verify`. See
    :meth:`to_dataframe` for the tidy schema, :meth:`summary` for the
    per-metric comparison against observed, and
    :meth:`category_summary` for the per-category rollup.

    Verification, in the sense of Stedinger and Taylor (1982),
    demonstrates that generated flows reproduce the statistics the
    generator was designed to reproduce: moments, correlations, and
    distributional shape.

    References
    ----------
    Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow
    generation: 1. Model verification and validation. Water Resources
    Research, 18(4), 909-918.
    """

    _suite: str = "verification"
