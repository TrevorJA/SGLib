"""
Pre-built pipeline configurations.

This module provides convenience wrapper classes for common generator-disaggregator
combinations. These classes internally use the GeneratorDisaggregatorPipeline but
provide a simplified interface for frequently used combinations.

Data is not passed at construction time. Use ``preprocessing(Q_obs)`` or
``fit(Q_obs)`` to supply observed flow data.
"""

from typing import Optional

from synhydro.core.pipeline import GeneratorDisaggregatorPipeline
from synhydro.methods.generation.hybrid.kirsch import KirschGenerator
from synhydro.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator
from synhydro.methods.disaggregation.temporal.nowak import NowakDisaggregator


class KirschNowakPipeline(GeneratorDisaggregatorPipeline):
    """
    Pre-configured pipeline combining Kirsch generator with Nowak disaggregator.

    This pipeline generates monthly synthetic flows using the Kirsch nonparametric
    bootstrap method, then disaggregates them to daily flows using the Nowak
    KNN-based temporal disaggregation method.

    Data is not passed at construction time. Use ``preprocessing(Q_obs)`` or
    ``fit(Q_obs)`` to supply observed flow data.

    Parameters
    ----------
    generate_using_log_flow : bool, default=True
        Whether to generate in log-space (Kirsch parameter).
    matrix_repair_method : str, default='spectral'
        Method for repairing correlation matrices (Kirsch parameter).
    n_neighbors : int, default=5
        Number of KNN neighbors for disaggregation (Nowak parameter).
    max_month_shift : int, default=7
        Maximum day shift for monthly profiles (Nowak parameter).
    name : str, optional
        Name for this pipeline instance.
    debug : bool, default=False
        Enable debug logging.

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.pipelines import KirschNowakPipeline
    >>>
    >>> # Load daily historic flows
    >>> Q_daily = pd.read_csv('daily_flows.csv', index_col=0, parse_dates=True)
    >>>
    >>> # Create pipeline (no data)
    >>> pipeline = KirschNowakPipeline()
    >>>
    >>> # Fit and generate
    >>> pipeline.preprocessing(Q_daily)
    >>> pipeline.fit()
    >>> daily_ensemble = pipeline.generate(n_realizations=100, n_years=50)

    Notes
    -----
    This pipeline is equivalent to creating:
    ```python
    generator = KirschGenerator(generate_using_log_flow=True)
    disaggregator = NowakDisaggregator(n_neighbors=5)
    pipeline = GeneratorDisaggregatorPipeline(generator, disaggregator)
    pipeline.fit(Q_obs)
    ```

    References
    ----------
    Kirsch generator: Nonparametric bootstrap with correlation preservation
    Nowak disaggregator: KNN-based temporal disaggregation (Nowak et al., 2010)
    """

    def __init__(
        self,
        *,
        generate_using_log_flow: bool = True,
        matrix_repair_method: str = "spectral",
        n_neighbors: int = 5,
        max_month_shift: int = 7,
        name: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the Kirsch-Nowak pipeline.

        Parameters
        ----------
        generate_using_log_flow : bool, default=True
            Generate in log-space for Kirsch.
        matrix_repair_method : str, default='spectral'
            Correlation matrix repair method for Kirsch.
        n_neighbors : int, default=5
            Number of KNN neighbors for Nowak.
        max_month_shift : int, default=7
            Day shift for Nowak monthly profiles.
        name : str, optional
            Pipeline name.
        debug : bool, default=False
            Enable debug logging.
        """
        # Create Kirsch generator
        generator = KirschGenerator(
            generate_using_log_flow=generate_using_log_flow,
            matrix_repair_method=matrix_repair_method,
            debug=debug,
        )

        # Create Nowak disaggregator
        disaggregator = NowakDisaggregator(
            n_neighbors=n_neighbors,
            max_month_shift=max_month_shift,
            debug=debug,
        )

        # Initialize parent pipeline
        super().__init__(
            generator=generator,
            disaggregator=disaggregator,
            name=name or "KirschNowakPipeline",
            debug=debug,
        )


class ThomasFieringNowakPipeline(GeneratorDisaggregatorPipeline):
    """
    Pre-configured pipeline combining Thomas-Fiering generator with Nowak disaggregator.

    This pipeline generates monthly synthetic flows using the Thomas-Fiering AR(1)
    parametric method with Stedinger-Taylor normalization, then disaggregates them
    to daily flows using the Nowak KNN-based temporal disaggregation method.

    Note: Thomas-Fiering is a univariate method, so only single-site generation
    is supported. For multisite, use KirschNowakPipeline instead.

    Data is not passed at construction time. Use ``preprocessing(Q_obs)`` or
    ``fit(Q_obs)`` to supply observed flow data.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of KNN neighbors for disaggregation (Nowak parameter).
    max_month_shift : int, default=7
        Maximum day shift for monthly profiles (Nowak parameter).
    name : str, optional
        Name for this pipeline instance.
    debug : bool, default=False
        Enable debug logging.

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.pipelines import ThomasFieringNowakPipeline
    >>>
    >>> # Load daily historic flows (single site)
    >>> Q_daily = pd.read_csv('daily_flows.csv', index_col=0, parse_dates=True)
    >>>
    >>> # Create pipeline (no data)
    >>> pipeline = ThomasFieringNowakPipeline()
    >>>
    >>> # Fit and generate
    >>> pipeline.preprocessing(Q_daily['site_1'])
    >>> pipeline.fit()
    >>> daily_ensemble = pipeline.generate(n_realizations=100, n_years=50)

    Notes
    -----
    This pipeline is equivalent to creating:
    ```python
    generator = ThomasFieringGenerator()
    disaggregator = NowakDisaggregator(n_neighbors=5)
    pipeline = GeneratorDisaggregatorPipeline(generator, disaggregator)
    pipeline.fit(Q_obs)
    ```

    References
    ----------
    Thomas-Fiering: AR(1) with Stedinger-Taylor normalization
    Nowak disaggregator: KNN-based temporal disaggregation (Nowak et al., 2010)
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        max_month_shift: int = 7,
        name: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the Thomas-Fiering-Nowak pipeline.

        Parameters
        ----------
        n_neighbors : int, default=5
            Number of KNN neighbors for Nowak.
        max_month_shift : int, default=7
            Day shift for Nowak monthly profiles.
        name : str, optional
            Pipeline name.
        debug : bool, default=False
            Enable debug logging.
        """
        # Create Thomas-Fiering generator
        generator = ThomasFieringGenerator(
            debug=debug,
        )

        # Create Nowak disaggregator
        disaggregator = NowakDisaggregator(
            n_neighbors=n_neighbors,
            max_month_shift=max_month_shift,
            debug=debug,
        )

        # Initialize parent pipeline
        super().__init__(
            generator=generator,
            disaggregator=disaggregator,
            name=name or "ThomasFieringNowakPipeline",
            debug=debug,
        )
