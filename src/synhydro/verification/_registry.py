"""
Metric registry instance for the verification suite.
"""

from synhydro._evaluation import MetricRegistry

VERIFICATION_METRICS = MetricRegistry(suite="verification")
