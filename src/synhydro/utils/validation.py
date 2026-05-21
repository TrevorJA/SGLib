"""
Data validation utilities.

Provides common validation functions for generator inputs and outputs.
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, List

logger = logging.getLogger(__name__)


def validate_timeseries(
    data: Union[pd.Series, pd.DataFrame],
    require_datetime_index: bool = True,
    allow_nan: bool = True,
    min_length: int = 10,
    variable_name: str = "data",
) -> None:
    """
    Validate timeseries data for generator input.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input timeseries data to validate.
    require_datetime_index : bool, default=True
        If True, require data to have a DatetimeIndex.
    allow_nan : bool, default=True
        If True, allow NaN values in the data.
    min_length : int, default=10
        Minimum required length of the timeseries.
    variable_name : str, default='data'
        Name of variable for error messages.

    Raises
    ------
    TypeError
        If data is not a pandas Series or DataFrame.
    ValueError
        If data doesn't meet validation criteria.

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.utils.validation import validate_timeseries
    >>> Q = pd.Series([100, 200, 300],
    ...               index=pd.date_range('2020-01-01', periods=3))
    >>> validate_timeseries(Q)  # Passes validation

    >>> Q_short = pd.Series([100, 200])
    >>> validate_timeseries(Q_short, min_length=10)  # Raises ValueError
    """
    # Type validation
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"{variable_name} must be pandas Series or DataFrame, "
            f"got {type(data).__name__}"
        )

    # Length validation
    if len(data) < min_length:
        raise ValueError(
            f"{variable_name} is too short: {len(data)} timesteps "
            f"(minimum required: {min_length})"
        )

    # DatetimeIndex validation
    if require_datetime_index:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                f"{variable_name} must have a DatetimeIndex, "
                f"got {type(data.index).__name__}. "
                "Convert with: data.index = pd.to_datetime(data.index)"
            )

    # NaN validation
    if not allow_nan:
        if isinstance(data, pd.Series):
            n_nan = data.isnull().sum()
        else:
            n_nan = data.isnull().sum().sum()

        if n_nan > 0:
            raise ValueError(
                f"{variable_name} contains {n_nan} NaN values. "
                "Remove or interpolate missing data before fitting."
            )

    logger.debug(f"{variable_name} passed validation: {len(data)} timesteps")


def validate_frequency(
    data: Union[pd.Series, pd.DataFrame],
    expected_freq: Optional[str] = None,
    allow_irregular: bool = False,
) -> str:
    """
    Validate and infer the frequency of timeseries data.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Timeseries data with DatetimeIndex.
    expected_freq : str, optional
        Expected frequency code ('D', 'MS', 'YS', etc.).
        If provided, validates data matches this frequency.
    allow_irregular : bool, default=False
        If True, allow irregular time series.

    Returns
    -------
    str
        Inferred frequency code.

    Raises
    ------
    ValueError
        If frequency doesn't match expected or can't be inferred.

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.utils.validation import validate_frequency
    >>> Q = pd.Series([100, 200, 300],
    ...               index=pd.date_range('2020-01-01', periods=3, freq='D'))
    >>> freq = validate_frequency(Q)
    >>> print(freq)
    'D'
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data must have DatetimeIndex for frequency validation")

    # Try to get frequency from index
    freq = data.index.freq

    # If not set, try to infer
    if freq is None:
        try:
            freq = pd.infer_freq(data.index)
        except (ValueError, TypeError):
            freq = None

    # Handle case where frequency can't be inferred
    if freq is None:
        if not allow_irregular:
            raise ValueError(
                "Could not infer frequency from data. "
                "Data may have irregular spacing. "
                "Set allow_irregular=True to bypass this check."
            )
        logger.warning("Could not infer frequency - data may be irregular")
        return None

    # Convert to string if it's a pandas offset
    freq_str = str(freq) if not isinstance(freq, str) else freq

    # Validate against expected frequency
    if expected_freq is not None:
        if freq_str != expected_freq:
            raise ValueError(
                f"Data frequency ({freq_str}) does not match "
                f"expected frequency ({expected_freq})"
            )

    logger.debug(f"Validated frequency: {freq_str}")
    return freq_str


def validate_positive(
    data: Union[pd.Series, pd.DataFrame, np.ndarray],
    variable_name: str = "data",
    allow_zero: bool = True,
    strict: bool = False,
) -> None:
    """
    Validate that data contains only positive values.

    Useful for flow data which should always be non-negative.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, or np.ndarray
        Data to validate.
    variable_name : str, default='data'
        Name of variable for error messages.
    allow_zero : bool, default=True
        If True, allow zero values.
    strict : bool, default=False
        If True, raise error on any negative values.
        If False, only log warning for negative values.

    Raises
    ------
    ValueError
        If data contains negative values (when strict=True).

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.utils.validation import validate_positive
    >>> Q = pd.Series([100, 200, 300])
    >>> validate_positive(Q)  # Passes

    >>> Q_neg = pd.Series([100, -50, 300])
    >>> validate_positive(Q_neg, strict=True)  # Raises ValueError
    """
    # Get min value
    if isinstance(data, (pd.Series, pd.DataFrame)):
        min_val = data.min().min() if isinstance(data, pd.DataFrame) else data.min()
        n_negative = (
            (data < 0).sum().sum()
            if isinstance(data, pd.DataFrame)
            else (data < 0).sum()
        )
    else:
        min_val = np.min(data)
        n_negative = np.sum(data < 0)

    # Check for negative values
    if min_val < 0:
        msg = f"{variable_name} contains {n_negative} negative values (min: {min_val:.2f})"

        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    # Check for zero values if not allowed
    if not allow_zero:
        if isinstance(data, (pd.Series, pd.DataFrame)):
            n_zero = (
                (data == 0).sum().sum()
                if isinstance(data, pd.DataFrame)
                else (data == 0).sum()
            )
        else:
            n_zero = np.sum(data == 0)

        if n_zero > 0:
            logger.warning(f"{variable_name} contains {n_zero} zero values")


def validate_columns(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    allow_extra: bool = True,
) -> None:
    """
    Validate DataFrame columns.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to validate.
    required_columns : List[str], optional
        List of required column names.
    allow_extra : bool, default=True
        If True, allow extra columns beyond required ones.

    Raises
    ------
    ValueError
        If required columns are missing or extra columns present (when not allowed).

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.utils.validation import validate_columns
    >>> df = pd.DataFrame({'site_A': [1, 2], 'site_B': [3, 4]})
    >>> validate_columns(df, required_columns=['site_A'])  # Passes
    """
    if required_columns is None:
        return

    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(data.columns)}"
        )

    if not allow_extra:
        extra = set(data.columns) - set(required_columns)
        if extra:
            raise ValueError(
                f"Unexpected columns found: {sorted(extra)}. "
                f"Expected only: {sorted(required_columns)}"
            )


__all__ = [
    "validate_timeseries",
    "validate_frequency",
    "validate_positive",
    "validate_columns",
]
