"""
Valencia-Schaake temporal disaggregation.

Joint multisite formulation per Valencia and Schaake (1973). Given a
multivariate aggregate (per-site annual totals) ``X``, the disaggregated
sub-period vector ``Y`` is modeled as

    Y = mu_Y + A (X - mu_X) + B V,    V ~ N(0, I)

with parameter matrices

    A    = S_yx S_xx^{-1}
    BB^T = S_yy - S_yx S_xx^{-1} S_xy

estimated from historical data. ``B`` is computed by rank-aware spectral
factorization (rank at most ``N - 1`` per the paper, where ``N`` is the
number of fitted years). With ``transform='none'`` the factorization is
carried out in the null space of the aggregation operator ``C``, which
preserves the paper's exact-additivity identities ``C A = I`` and
``C B = 0`` to floating-point precision and guarantees ``C Y = X`` for
any draw. With a log or Box-Cox transform those identities do not hold
in the fitted space, so the full conditional covariance is factored and
additivity is restored by a per-site proportional rescale (Grygier and
Stedinger, 1988, Eq. 14).

Within a year the sub-period vector ``Y`` is ordered site-major, matching
paper Equation (3): the first ``n_subperiods`` entries are site 1, then
site 2, and so on.

References
----------
Valencia, R.D., and Schaake, J.C. (1973).
Disaggregation processes in stochastic hydrology.
Water Resources Research, 9(3), 580-585.
https://doi.org/10.1029/WR009i003p00580

Grygier, J.C., and Stedinger, J.R. (1988).
Condensed disaggregation procedures and conservation corrections for
stochastic hydrology.
Water Resources Research, 24(10), 1574-1584.
"""

import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from synhydro.core.base import Disaggregator, FittedParams
from synhydro.core.ensemble import Ensemble, EnsembleMetadata


logger = logging.getLogger(__name__)


# Sub-period configurations: maps n_subperiods -> (starting calendar months,
# pandas frequency string). Months are 1-indexed and span an even partition
# of a calendar year. Only configurations with an exact month-stride are
# supported, which keeps the time axis unambiguous.
_SUBPERIOD_LAYOUTS: dict = {
    2: ((1, 7), "2QS"),
    3: ((1, 5, 9), "4MS"),
    4: ((1, 4, 7, 10), "QS"),
    6: ((1, 3, 5, 7, 9, 11), "2MS"),
    12: ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), "MS"),
}


def _subperiod_layout(n_subperiods: int) -> Tuple[Tuple[int, ...], str]:
    """Return the (starting-months, pandas-frequency) layout for the
    configured number of sub-periods. Raises ``ValueError`` for
    unsupported counts."""
    try:
        return _SUBPERIOD_LAYOUTS[n_subperiods]
    except KeyError as e:
        raise ValueError(
            f"n_subperiods={n_subperiods} is not supported; "
            f"choose one of {sorted(_SUBPERIOD_LAYOUTS)}."
        ) from e


class ValenciaSchaakeDisaggregator(Disaggregator):
    """
    Joint multisite Valencia-Schaake temporal disaggregator.

    Disaggregates an aggregate flow vector (e.g., per-site annual totals)
    into sub-period flows (e.g., monthly) using the linear model

        Y = mu_Y + A (X - mu_X) + B V,    V ~ N(0, I)

    where ``A`` and ``B`` are fit jointly across all sites and sub-periods.
    The fit preserves the paper's exact-additivity identity
    ``C Y = X`` (where ``C`` is the per-site summation operator) for the
    untransformed model.

    Parameters
    ----------
    n_subperiods : int, default=12
        Number of sub-periods per aggregate period. Supported values are
        2, 3, 4, 6, and 12 (each must divide the calendar year into
        equal month-strided segments).
    transform : str, default='log'
        Transformation applied to sub-period flows before fitting:
        'log', 'boxcox', or 'none'. The paper assumes Gaussian data is
        "convenient but not absolutely necessary"; the log option is a
        common practical choice for hydrologic flows.
    conservation_method : str, default='proportional'
        Method to enforce per-site sum consistency: 'proportional' or
        'none'. Required when a nonlinear transform is used; for
        ``transform='none'`` the linear model already satisfies
        additivity exactly. The combination ``transform != 'none'`` with
        ``conservation_method='none'`` is rejected because it produces
        silently non-conservative output.
    name : str, optional
        Name identifier for this disaggregator instance.
    debug : bool, default=False
        Enable debug logging.

    Attributes
    ----------
    mu_Y_ : np.ndarray
        Sub-period mean vector, shape (n_subperiods * n_sites,),
        site-major ordering.
    mu_X_ : np.ndarray
        Aggregate mean vector, shape (n_sites,).
    S_yy_ : np.ndarray
        Sub-period covariance, shape
        (n_subperiods * n_sites, n_subperiods * n_sites).
    S_xx_ : np.ndarray
        Aggregate covariance, shape (n_sites, n_sites).
    S_yx_ : np.ndarray
        Cross-covariance ``Cov(Y, X)``, shape
        (n_subperiods * n_sites, n_sites).
    A_ : np.ndarray
        Regression matrix ``A = S_yx S_xx^{-1}``, shape
        (n_subperiods * n_sites, n_sites).
    B_ : np.ndarray
        Rank-aware factor with ``B B^T = S_yy - S_yx S_xx^{-1} S_xy``,
        shape (n_subperiods * n_sites, r). With ``transform='none'``
        ``B`` is factored in the null space of ``C`` so that ``C B = 0``
        and ``r <= min(n_years - 1, (n_subperiods - 1) * n_sites)`` (the
        null-space constraint tightens the paper's ``N - 1`` rank bound).
        With a nonlinear transform the full conditional covariance is
        factored and ``r <= min(n_years - 1, n_subperiods * n_sites)``.
    """

    def __init__(
        self,
        *,
        n_subperiods: int = 12,
        transform: str = "log",
        conservation_method: str = "proportional",
        name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(name=name, debug=debug)

        _months, _freq = _subperiod_layout(n_subperiods)

        valid_transforms = {"log", "boxcox", "none"}
        if transform not in valid_transforms:
            raise ValueError(
                f"transform={transform!r} is not supported; "
                f"choose one of {sorted(valid_transforms)}."
            )

        valid_conservation = {"proportional", "none"}
        if conservation_method not in valid_conservation:
            raise ValueError(
                f"conservation_method={conservation_method!r} is not "
                f"supported; choose one of {sorted(valid_conservation)}."
            )

        if transform != "none" and conservation_method == "none":
            raise ValueError(
                "conservation_method='none' is incompatible with a "
                "nonlinear transform: per-site annual sums would not be "
                "preserved. Use conservation_method='proportional' when "
                "transform is 'log' or 'boxcox', or set transform='none'."
            )

        self.n_subperiods = n_subperiods
        self.transform = transform
        self.conservation_method = conservation_method
        self._subperiod_months = _months
        self._output_freq = _freq

        self.init_params.algorithm_params = {
            "method": "Valencia-Schaake Disaggregation",
            "n_subperiods": n_subperiods,
            "transform": transform,
            "conservation_method": conservation_method,
        }

        self.mu_Y_ = None
        self.mu_X_ = None
        self.S_yy_ = None
        self.S_xx_ = None
        self.S_yx_ = None
        self.A_ = None
        self.B_ = None
        self.transform_params_ = {"type": "none"}

    @property
    def input_frequency(self) -> str:
        return "YS"

    @property
    def output_frequency(self) -> str:
        return self._output_freq

    def preprocessing(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Validate and store observed sub-period flows.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed flow data at sub-period (e.g., monthly) resolution.
        sites : list of str, optional
            Sites to retain. If None, uses all columns.
        """
        Q_obs_validated = self._store_obs_data(Q_obs, sites)
        self.Q_obs = Q_obs_validated
        self.Q_annual = Q_obs_validated.resample("YS").sum()

        self.logger.info(
            f"Preprocessing complete: {self.n_sites} sites, "
            f"{len(self.Q_obs)} sub-period observations, "
            f"{len(self.Q_annual)} annual observations"
        )
        self.update_state(preprocessed=True)

    def fit(
        self,
        Q_obs: Optional[Union[pd.Series, pd.DataFrame]] = None,
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the joint multisite Valencia-Schaake model.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed data; if provided, preprocessing runs automatically.
        sites : list of str, optional
            Sites to use (only when Q_obs is provided).
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)

        self.validate_preprocessing()

        Y_block_orig = self._organize_subperiods()
        n_years = Y_block_orig.shape[0]
        if n_years == 0:
            raise ValueError(
                "No complete aggregate periods found in data. "
                "Ensure data has at least one full year."
            )
        if n_years < 2:
            raise ValueError(f"Need at least 2 complete years to fit; got {n_years}.")

        if self.transform != "none":
            Y_block_working = self._apply_transform(Y_block_orig)
        else:
            Y_block_working = Y_block_orig

        self._compute_statistics(Y_block_orig, Y_block_working)
        self._compute_noise_factor()

        rank_bound = (self.n_subperiods - 1) * self.n_sites
        if n_years - 1 < rank_bound:
            self.logger.warning(
                "Short record: %d fitted years - 1 < (n_subperiods - 1) * n_sites = %d. "
                "Noise factor B will be heavily rank-deficient (rank <= %d).",
                n_years,
                rank_bound,
                n_years - 1,
            )

        self.update_state(fitted=True)
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            f"Fitting complete: {n_years} years x {self.n_subperiods} "
            f"sub-periods x {self.n_sites} sites; B rank = {self.B_.shape[1]}"
        )

    def _organize_subperiods(self) -> np.ndarray:
        """
        Build a ``(n_years, n_subperiods, n_sites)`` array of sub-period
        flows over complete historical years using the layout configured
        by ``n_subperiods``. Single-site data gets a trailing site axis
        of size 1.
        """
        years = self.Q_obs.index.year.unique()
        Y_list = []

        for year in years:
            year_mask = self.Q_obs.index.year == year
            if year_mask.sum() < self.n_subperiods:
                continue

            year_start = pd.Timestamp(year=year, month=1, day=1)
            year_end = pd.Timestamp(year=year + 1, month=1, day=1)
            starts = [
                pd.Timestamp(year=year, month=m, day=1) for m in self._subperiod_months
            ]
            edges = starts + [year_end]

            # Observed data is monthly, so a complete year has exactly
            # 12 / n_subperiods observations in every sub-period. Requiring
            # the exact count (not just mask.any()) prevents partial leading
            # or trailing years from biasing the moments when
            # n_subperiods < 12.
            expected_per_subperiod = 12 // self.n_subperiods
            rows = []
            complete = True
            for i in range(self.n_subperiods):
                mask = (self.Q_obs.index >= edges[i]) & (
                    self.Q_obs.index < edges[i + 1]
                )
                if mask.sum() != expected_per_subperiod:
                    complete = False
                    break
                rows.append(self.Q_obs.loc[mask].sum().values)
            if not complete:
                continue
            Y_list.append(np.asarray(rows))

        if not Y_list:
            raise ValueError("No complete periods found for disaggregation.")

        Y = np.array(Y_list)
        if Y.ndim == 2:
            Y = Y[:, :, np.newaxis]

        self.logger.debug(
            f"Organized {Y.shape[0]} complete years into sub-period array"
        )
        return Y

    def _apply_transform(self, Y_block: np.ndarray) -> np.ndarray:
        """Apply the configured transformation to sub-period flows."""
        if self.transform == "log":
            epsilon = 1e-6
            Y_transformed = np.log(Y_block + epsilon)
            self.transform_params_ = {"type": "log", "epsilon": epsilon}
        elif self.transform == "boxcox":
            Y_flat = np.clip(Y_block.flatten(), a_min=1e-6, a_max=None)
            Y_transformed_flat, lambda_bc = boxcox(Y_flat)
            Y_transformed = Y_transformed_flat.reshape(Y_block.shape)
            self.transform_params_ = {"type": "boxcox", "lambda": lambda_bc}
        else:
            Y_transformed = Y_block.copy()
            self.transform_params_ = {"type": "none"}

        self.logger.debug(f"Applied {self.transform} transformation")
        return Y_transformed

    def _inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Invert the fitted transformation."""
        ttype = self.transform_params_.get("type", "none")
        if ttype == "log":
            epsilon = self.transform_params_.get("epsilon", 1e-6)
            Y_inv = np.exp(Y) - epsilon
        elif ttype == "boxcox":
            Y_inv = inv_boxcox(Y, self.transform_params_["lambda"])
        else:
            return Y.copy()
        # inv_boxcox can produce NaN when lambda*Y + 1 is outside the domain
        # of the inverse; np.clip does not strip NaN, so handle explicitly
        # before the lower-bound clip.
        Y_inv = np.where(np.isfinite(Y_inv), Y_inv, 0.0)
        return np.clip(Y_inv, a_min=0, a_max=None)

    def _compute_statistics(
        self, Y_block_orig: np.ndarray, Y_block_working: np.ndarray
    ) -> None:
        """
        Estimate the joint multisite moments per paper Eqs. 13-15.

        ``X`` is always computed from the **original-scale** sub-period flows
        so it matches the annual aggregates passed to ``disaggregate()``.
        ``Y`` is computed from the working-space flows (possibly transformed).
        When ``transform='none'`` the two block arrays are identical and the
        paper's ``X = CY`` identity holds exactly; otherwise the transform
        breaks linear additivity and the proportional adjustment restores
        per-site conservation.

        Parameters
        ----------
        Y_block_orig : np.ndarray
            Original-scale sub-period flows, shape
            ``(n_years, n_subperiods, n_sites)``.
        Y_block_working : np.ndarray
            Possibly transformed sub-period flows, same shape.
        """
        n_years, n_sub, n_sites = Y_block_working.shape

        Y_data = Y_block_working.transpose(0, 2, 1).reshape(n_years, n_sites * n_sub)
        X_data = Y_block_orig.sum(axis=1)

        self.mu_Y_ = Y_data.mean(axis=0)
        self.mu_X_ = X_data.mean(axis=0)

        Y_centered = Y_data - self.mu_Y_
        X_centered = X_data - self.mu_X_

        self.S_yy_ = (Y_centered.T @ Y_centered) / (n_years - 1)
        self.S_xx_ = (X_centered.T @ X_centered) / (n_years - 1)
        self.S_yx_ = (Y_centered.T @ X_centered) / (n_years - 1)

        S_xx_inv = np.linalg.pinv(self.S_xx_)
        self.A_ = self.S_yx_ @ S_xx_inv

        self.logger.debug(
            f"Joint statistics: S_yy={self.S_yy_.shape}, S_xx={self.S_xx_.shape}, "
            f"S_yx={self.S_yx_.shape}, A={self.A_.shape}"
        )

    def _compute_noise_factor(self) -> None:
        """
        Construct the noise factor ``B`` such that
        ``B B^T = S_yy - S_yx S_xx^{-1} S_xy`` (paper Eq. 19).

        Two factorization paths are used:

        * ``transform='none'``: the paper's identities ``C A = I`` and
          ``C B = 0`` hold (Eqs. 39-40, ``C`` the per-site aggregation
          operator), so ``BB^T`` lives entirely in the null space of
          ``C``. We factor in an orthonormal basis ``N`` of that null
          space: any vector ``B v`` then lies in the null space by
          construction, ``C B = 0`` to floating-point precision, and the
          exact-additivity identity ``C Y = X`` holds for any draw. The
          retained rank is at most ``min((n_subperiods - 1) * n_sites,
          n_years - 1)`` per the paper's rank bound (p.584).
        * ``transform='log'`` or ``'boxcox'``: ``X`` is the original-scale
          aggregate while ``Y`` is transformed, so ``C BB^T != 0`` and a
          null-space projection would discard real conditional variance
          (about 2-5 percent of the trace on typical monthly data). The
          full symmetric ``BB^T`` is therefore factored directly after
          clipping numerically non-positive eigenvalues; additivity is
          restored afterwards by the proportional rescale in
          ``disaggregate``.
        """
        from scipy.linalg import null_space

        n_sub = self.n_subperiods
        n_sites = self.n_sites
        n = n_sub * n_sites

        S_xx_inv = np.linalg.pinv(self.S_xx_)
        BBt = self.S_yy_ - self.S_yx_ @ S_xx_inv @ self.S_yx_.T

        if self.transform == "none":
            ones_row = np.ones((1, n_sub))
            block_basis = null_space(ones_row)  # (n_sub, n_sub - 1)
            N = np.zeros((n, (n_sub - 1) * n_sites))
            for s in range(n_sites):
                N[
                    s * n_sub : (s + 1) * n_sub, s * (n_sub - 1) : (s + 1) * (n_sub - 1)
                ] = block_basis
            M2 = N.T @ BBt @ N
            rank_max = (n_sub - 1) * n_sites
        else:
            N = np.eye(n)
            M2 = BBt
            rank_max = n
        M2 = 0.5 * (M2 + M2.T)

        eigvals, eigvecs = np.linalg.eigh(M2)
        scale = max(abs(eigvals[-1]), 1.0) if eigvals.size else 1.0
        tol = scale * max(eigvals.size, 1) * np.finfo(eigvals.dtype).eps * 10.0
        keep = eigvals > tol

        if not keep.any():
            # Degenerate: the conditional distribution has no residual
            # variance (e.g., perfectly cyclic input). The conditional mean
            # alone defines the disaggregation.
            self.B_ = np.zeros((n, 0))
            self.logger.debug(
                "Conditional covariance is numerically zero; noise factor B has rank 0."
            )
            return

        M = eigvecs[:, keep] * np.sqrt(eigvals[keep])
        self.B_ = N @ M

        self.logger.debug(
            f"Noise factor B: shape={self.B_.shape}, "
            f"rank={self.B_.shape[1]}/{rank_max} max, tol={tol:.2e}"
        )

    def _compute_fitted_params(self) -> FittedParams:
        """Package fitted statistics in the framework's FittedParams object."""
        m = self.n_sites
        s = self.n_subperiods
        # Count only the independent synthesis-model parameters:
        # mu_Y (s*m) + mu_X (m) + A (s*m, m) + B (s*m, r).
        # Covariance matrices are derived from these, not independent.
        n_params = s * m + m + s * m * m + self.B_.size

        labels = [f"{site}_sub{j}" for site in self._sites for j in range(s)]
        means_series = pd.Series(self.mu_Y_, index=labels)
        stds_series = pd.Series(np.sqrt(np.diag(self.S_yy_)), index=labels)

        return FittedParams(
            means_=means_series,
            stds_=stds_series,
            correlations_=self._get_correlation_matrix(),
            distributions_={
                "type": "multivariate_normal",
                "method": "Valencia-Schaake (joint multisite)",
            },
            fitted_models_={
                "n_subperiods": self.n_subperiods,
                "n_sites": self.n_sites,
                "transform": self.transform,
                "conservation_method": self.conservation_method,
                "B_rank": int(self.B_.shape[1]),
            },
            n_parameters_=int(n_params),
            sample_size_=len(self.Q_obs),
            n_sites_=self.n_sites,
            training_period_=(
                str(self.Q_obs.index[0].date()),
                str(self.Q_obs.index[-1].date()),
            ),
        )

    def _get_correlation_matrix(self) -> np.ndarray:
        """Correlation matrix derived from S_yy_."""
        stds = np.sqrt(np.diag(self.S_yy_))
        denom = np.outer(stds, stds)
        denom[denom == 0] = 1.0
        return self.S_yy_ / denom

    def disaggregate(
        self, ensemble: Ensemble, seed: Optional[int] = None, **kwargs: Any
    ) -> Ensemble:
        """
        Disaggregate an annual ensemble to monthly (or n_subperiods) flows.

        Parameters
        ----------
        ensemble : Ensemble
            Input ensemble at annual resolution. Must have frequency 'YS'
            and the same site set as the fitted disaggregator.
        seed : int, optional
            Random seed for reproducibility. A single
            ``np.random.default_rng(seed)`` is consumed sequentially in
            ``ensemble.data_by_realization`` insertion order, so changing
            the number of realizations or their iteration order will
            change every realization's draws.

        Returns
        -------
        Ensemble
            Disaggregated ensemble at sub-period resolution.
        """
        self.validate_fit()
        self.validate_input_ensemble(ensemble)

        rng = np.random.default_rng(seed)

        sub_realization_dict = {}
        for realization_id, annual_df in ensemble.data_by_realization.items():
            sub_realization_dict[realization_id] = (
                self._disaggregate_single_realization(annual_df, rng=rng)
            )

        first_df = sub_realization_dict[next(iter(sub_realization_dict))]
        metadata = EnsembleMetadata(
            generator_class=ensemble.metadata.generator_class,
            generator_params=ensemble.metadata.generator_params,
            n_realizations=len(sub_realization_dict),
            n_sites=len(self._sites),
            time_resolution=self.output_frequency,
            time_period=(str(first_df.index[0].date()), str(first_df.index[-1].date())),
        )

        out_ensemble = Ensemble(sub_realization_dict, metadata=metadata)
        self.logger.info(
            f"Disaggregated {len(sub_realization_dict)} realizations "
            f"from annual to sub-period resolution"
        )
        return out_ensemble

    def _disaggregate_single_realization(
        self, annual_df: pd.DataFrame, *, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Disaggregate one realization. ``annual_df`` columns must match the
        fitted site order; we reindex to enforce that.
        """
        annual_df = annual_df.reindex(columns=self._sites)
        X_syn = annual_df.values  # (n_years, n_sites)
        n_years, n_sites = X_syn.shape
        n_sub = self.n_subperiods

        out_rows = np.empty((n_years * n_sub, n_sites), dtype=float)
        out_dates: List[pd.Timestamp] = []

        for t in range(n_years):
            year = annual_df.index[t].year
            X_t = X_syn[t]

            mu_Y_given_X = self.mu_Y_ + self.A_ @ (X_t - self.mu_X_)
            V = rng.standard_normal(self.B_.shape[1])
            Y_t = mu_Y_given_X + self.B_ @ V

            if self.transform != "none":
                Y_t = self._inverse_transform(Y_t)

            # Site-major unpacking per paper Eq. (3): rows = sites, cols = sub-periods.
            Y_sitewise = Y_t.reshape(n_sites, n_sub)

            if self.conservation_method == "proportional":
                # Clip then rescale so per-site annual sums match X_t exactly
                # even when the nonlinear transform or clipping breaks linear
                # additivity. For transform='none' with a paper-faithful B
                # (rank-aware, C B = 0) the pre-clip Y_sitewise already sums
                # to X_t; the rescale absorbs any clip-induced error.
                Y_sitewise = np.clip(Y_sitewise, 0, None)
                for s in range(n_sites):
                    s_sum = Y_sitewise[s].sum()
                    if s_sum > 0:
                        Y_sitewise[s] = Y_sitewise[s] * (X_t[s] / s_sum)
                    else:
                        # All entries were non-positive and got clipped to
                        # zero; fall back to a uniform split to preserve the
                        # site's annual total.
                        Y_sitewise[s, :] = X_t[s] / n_sub

            out_rows[t * n_sub : (t + 1) * n_sub, :] = Y_sitewise.T
            for month in self._subperiod_months:
                out_dates.append(pd.Timestamp(year=year, month=month, day=1))

        return pd.DataFrame(
            out_rows,
            index=pd.DatetimeIndex(np.array(out_dates, dtype="datetime64[s]")),
            columns=self._sites,
        )
