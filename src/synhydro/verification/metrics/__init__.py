"""
Verification metric functions.

Importing this package registers every built-in verification metric in
the VERIFICATION_METRICS registry. All metric functions are directly
callable on a pd.Series (or pd.DataFrame for matrix metrics).
"""

from synhydro.verification.metrics.marginal import (
    mean,
    std,
    cv,
    skewness,
    kurtosis,
    minimum,
    maximum,
    flow_q10,
    flow_q50,
    flow_q90,
    ks_statistic,
)
from synhydro.verification.metrics.temporal import (
    lag1_autocorrelation,
    lag2_autocorrelation,
    acf,
    hurst,
)
from synhydro.verification.metrics.seasonal import (
    monthly_mean,
    monthly_std,
    monthly_skewness,
    monthly_maximum,
    monthly_minimum,
    monthly_lag1_correlation,
    monthly_ranksum_pvalue,
    monthly_levene_pvalue,
)
from synhydro.verification.metrics.annual import (
    annual_mean,
    annual_sd,
    annual_cv,
    annual_skewness,
    annual_lag1_autocorrelation,
    annual_minimum,
    annual_maximum,
)
from synhydro.verification.metrics.spatial import (
    cross_correlation,
    cross_correlation_lag1,
)
from synhydro.verification.metrics.fdc import fdc, fdc_log_rmse
from synhydro.verification.metrics.lmoments import l_cv, l_skewness, l_kurtosis
from synhydro.verification.metrics.extremes import (
    annual_max_mean,
    annual_max_cv,
    gev_rp10,
    gev_rp50,
    gev_rp100,
    annual_min_mean,
    annual_min_cv,
    seven_day_min_mean,
    seven_day_min_cv,
)
from synhydro.verification.metrics.spectral import (
    spectral_density,
    low_frequency_variance_fraction,
)
