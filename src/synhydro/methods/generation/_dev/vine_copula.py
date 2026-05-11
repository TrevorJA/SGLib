"""
Vine Copula Generator for multi-site monthly streamflow.

NOT EXPORTED FROM THE PUBLIC API. NOT BUILT INTO THE DOCUMENTATION SITE.

Status: needs substantial refactor to align faithfully to a primary publication
before public release. The current implementation cites Pereira et al. (2017)
but does not implement Pereira's actual method; it is closer to a generic
"periodic vine on PAR(1) Gaussianized residuals" model in the spirit of
Yu et al. (2025) / Wang & Shen (2023).

This module is retained on disk for a future rewrite. It is excluded from
``synhydro/__init__.py``, ``synhydro/methods/generation/parametric/__init__.py``,
``docs/algorithms/index.md``, ``docs/api/generators.md``, ``mkdocs.yml`` nav,
``README.md``, and the docs-site build (``mkdocs.yml`` ``exclude_docs``).
The existing test file ``tests/test_vine_copula_generator.py`` still imports
the class via its full module path and continues to pass.

================================================================================
TODO: align to Pereira et al. (2017) before re-exporting
================================================================================

A careful read of the paper against the current code identified 9 algorithmic
gaps and 4 documentation gaps. Equation and section references below are to
Pereira et al. (2017), "A periodic spatial vine copula model for multi-site
streamflow simulation," Electric Power Systems Research 152, 9-17.

ALGORITHMIC GAPS

  (1) Marginal pipeline (Eqs. 5-6).
      Pereira: log-transform z -> y = ln(z); standardize y by monthly mean
      mu_m^s and std sigma_m^s INSIDE the PAR equation; PAR residuals are
      already approximately N(0, 1) and used directly.
      Current code: per-(month, site) gamma vs. lognormal by BIC, or empirical
      NormalScoreTransform; PIT through that fitted CDF; norm.ppf to standard
      normal; THEN PAR(1) on the normal scores. This is a different paradigm
      (marginal-fit-then-AR-on-normal-scores) and is not what Pereira does.

  (2) PAR order (Eq. 6).
      Pereira: PAR(p) with order p_m^s explicitly varying by month and site
      (paper says "the orders, vary according to the month").
      Current code: PAR(1) hardcoded; no support for higher orders or
      per-month order selection.

  (3) Tree structure across months (Section 3.3, 4.1).
      Pereira: "the pair-copula families vary according to the months, while
      we keep the tree structure fixed." Tree structure is selected ONCE
      (Dissmann on pooled residuals); only families and parameters vary by
      month.
      Current code: fits a complete vine -- structure + families + parameters
      -- independently per calendar month. This produces 12 different tree
      structures.

  (4) Spatial T1 reparametrization (Eqs. 7, 10, 11).
      Pereira's distinguishing contribution. T1 Kendall taus are regressed on
      log(distance), a same-river-no-other-plant-between indicator D_ij, and
      seasonal dummies for May-September:
          g_z(tau_{ij,m}) = a_0 + a_1 ln(Dist_ij) + a_2 D_ij
                          + sum_{q=5}^{9} a_{q-2} S_{qm}
      with g_z(r) = 0.5 ln((1+r)/(1-r)) the Fisher z-transform. T1 has 8
      parameters total (vs. 12 (d-1) unconstrained).
      Implementation requires user-supplied n_sites x n_sites distance and
      river-network matrices passed to ``fit()``; joint OLS across all months;
      then per-pair-per-month copula parameter via theta = g_tau^{-1}(tau; b)
      for the chosen pair-copula family b.
      Current code: not implemented at all.

  (5) Family set (Section 3.3, 4.1).
      Pereira: Clayton, Gumbel, Normal, Product (independence), Student-t --
      with 90, 180, 270 degree rotations of Clayton and Gumbel.
      Current code: when ``family_set='all'`` includes Frank, Joe, BB1, BB6,
      BB7, BB8 -- none of which are in Pereira.

  (6) Family selection criterion (Section 4.1).
      Pereira: BIC.
      Current code: AIC by default.

  (7) Student-t df handling (Section 4.2).
      Pereira: "the degrees of freedom of the Student-t copula are not
      estimated. Instead, we use the monthly average of the degree of freedom
      estimates obtained while selecting the pair-copulas."
      Current code: per-pair, per-month estimation by pyvinecopulib; no
      averaging across months.

  (8) Truncation level (Section 4.1).
      Pereira: L = 4 in their case study (95% of total log-likelihood).
      Current code: ``trunc_level=None`` default (no truncation).

  (9) Log transform (Eq. 5).
      Pereira: mandatory.
      Current code: optional via ``log_transform=False`` default.

DOCUMENTATION GAPS (in docs/algorithms/vine_copula.md, currently excluded
from the docs build)

  (10) Title cites "Yu et al., 2025; Wang and Shen, 2023" rather than Pereira.
  (11) Module docstring lists Yu/Wang first; Pereira buried at the end.
  (12) References section places Pereira under "See also" rather than primary.
  (13) Formulation section describes the gamma/lognormal pipeline that the
       code currently implements, not Pereira's log-then-standardize-then-PAR
       pipeline.

REFACTOR PLAN (to be confirmed with user before resuming work)

  Source code:
   - Replace the marginal pipeline (drop _fit_parametric_marginals,
     _fit_empirical_marginals, _pit_to_normal). Implement Pereira's:
       y = ln(z); standardize within PAR; residuals ~ N(0, 1); u = Phi(eps).
   - Add PAR(p) with method-of-moments / Yule-Walker per Salas et al. (1980),
     order p_m^s either user-specified or auto-selected per (month, site) by
     BIC over a small range (e.g., 1..3).
   - Restructure ``fit`` to: (a) fit ONE tree structure on stacked residuals,
     (b) apply spatial T1 reparam (joint OLS across months), (c) sequentially
     fit T_2..T_L families+parameters per month with structure fixed,
     (d) post-fit step: average Student-t df estimates across months.
   - Add ``distance_matrix`` and ``river_network`` arguments to ``fit()``.
     Open question: hard requirement (raises) or optional with documented
     fallback to per-month T1 (Yu/Wang style) plus a warning?
   - Default ``family_set`` to Pereira's set (clayton, gumbel, gaussian,
     indep, student_t) plus rotated clayton/gumbel.
   - Default ``selection_criterion='bic'``.
   - Default ``trunc_level=4``.
   - Default log transform on (and remove the parameter, since Pereira makes
     it mandatory).
   - Update ``_generate_one`` to invert Pereira's pipeline (vine sample ->
     Phi^{-1} -> PAR(p) recursion -> de-standardize by mu_m, sigma_m ->
     exp).

  Documentation (docs/algorithms/vine_copula.md):
   - Retitle "Vine Copula Generator (Pereira et al., 2017)".
   - Rewrite Notation table to use Pereira's symbols.
   - Rewrite Formulation to follow Eqs. 5, 6, 7, 10, 11, 13.
   - Add a dedicated "Spatial reparametrization for T1" subsection.
   - Cite Pereira primary; Yu (2025), Wang & Shen (2023), Erhardt-Czado-
     Schepsmeier (2015), Dissmann et al. (2013), Aas et al. (2009),
     Salas et al. (1980) as supporting references.

OPEN QUESTIONS FOR THE USER

  Q1: Replace the gamma/lognormal/empirical marginal options with Pereira's
      log-then-standardize-then-PAR pipeline, or preserve them as non-Pereira
      opt-in alternatives with a prominent doc note?

  Q2: Make ``distance_matrix`` and ``river_network`` HARD requirements for
      ``fit()`` (strict Pereira alignment), or OPTIONAL with explicit
      fallback to per-month T1 fitting (and a warning)?

  Q3: PAR order: auto-select per (month, site) by BIC over {1, 2, 3}, or
      user-specified global order with default p = 1?

================================================================================

References
----------
Pereira, G.A.A., Veiga, A., Erhardt, T., and Czado, C. (2017). A periodic
    spatial vine copula model for multi-site streamflow simulation. Electric
    Power Systems Research, 152, 9-17.
    https://doi.org/10.1016/j.epsr.2017.06.017
Yu, X., Xu, Y.-P., Guo, Y., Chen, S., and Gu, H. (2025). Synchronization
    frequency analysis and stochastic simulation of multi-site flood flows
    based on the complicated vine copula structure. Hydrology and Earth System
    Sciences, 29, 179-214. https://doi.org/10.5194/hess-29-179-2025
Wang, X., and Shen, Y.-M. (2023). R-statistic based predictor variables
    selection and vine structure determination approach for stochastic
    streamflow generation. Journal of Hydrology, 617, 129093.
    https://doi.org/10.1016/j.jhydrol.2023.129093
Wang, W., Dong, Z., Zhang, T., Ren, L., Xue, L., Wu, T. (2024). Mixed
    D-vine copula-based conditional quantile model for stochastic monthly
    streamflow simulation. Water Science and Engineering, 17(1), 13-20.
    https://doi.org/10.1016/j.wse.2023.05.004
"""

import logging
from typing import Optional, Dict, List, Tuple, Union, Any

# pyvinecopulib must be imported before pandas/pyarrow on Windows to avoid
# a C++ runtime DLL conflict between pyarrow and pyvinecopulib's C++ layer.
try:
    import pyvinecopulib as _pyvinecopulib_preload  # noqa: F401
except ImportError:
    pass

import numpy as np
import pandas as pd
from scipy.stats import norm, gamma as gamma_dist, lognorm

from synhydro.core.base import Generator, FittedParams
from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.core.statistics import NormalScoreTransform

logger = logging.getLogger(__name__)

_LOG2PI = np.log(2.0 * np.pi)


def _bic(n: int, neg_loglik: float, k: int) -> float:
    """Bayesian Information Criterion: BIC = 2*NLL + k*ln(n)."""
    return 2.0 * neg_loglik + k * np.log(n)


def _require_pyvinecopulib():
    """Import pyvinecopulib; raise ImportError with install hint if missing."""
    try:
        import pyvinecopulib as pv
    except ImportError:
        raise ImportError(
            "pyvinecopulib is required for VineCopulaGenerator. "
            "Install it with:  pip install synhydro[vine]"
        ) from None
    return pv


class VineCopulaGenerator(Generator):
    """Multi-site monthly generator based on vine copulas.

    Fits per-(month, site) marginal distributions and a vine copula per
    calendar month on the PAR(1) residuals.  Supports R-vine, C-vine, and
    D-vine structures with automatic family and structure selection via
    pyvinecopulib.

    Parameters
    ----------
    vine_type : str, default="rvine"
        ``"rvine"``, ``"cvine"``, or ``"dvine"``.  Controls the vine tree
        structure constraint.
    family_set : str or list of str, default="all"
        Bivariate copula families to consider.  ``"all"`` uses all available
        parametric families.  A list of family names can also be provided
        (e.g., ``["gaussian", "clayton", "gumbel", "frank"]``).
    selection_criterion : str, default="aic"
        ``"aic"`` or ``"bic"`` for bivariate copula family selection at
        each edge.
    marginal_method : str, default="parametric"
        ``"parametric"`` fits gamma and log-normal per (month, site) and
        selects the winner by BIC.  ``"empirical"`` uses the Hazen
        plotting-position CDF via :class:`NormalScoreTransform`.
    log_transform : bool, default=False
        Apply ``log(Q + offset)`` before fitting.
    offset : float, default=1.0
        Additive offset for the log transform.
    trunc_level : int or None, default=None
        Truncation level for the vine tree.  ``None`` means no truncation
        (all trees are fitted).  Setting ``trunc_level=1`` truncates after
        the first tree, replacing higher trees with independence copulas.
    name : str, optional
        Instance name.
    debug : bool, default=False
        Enable debug logging.

    References
    ----------
    Yu et al. (2025), Wang & Shen (2023), Wang et al. (2024),
    Pereira et al. (2017).
    """

    supports_multisite = True
    supported_frequencies = ("MS",)

    _VALID_VINE_TYPES = ("rvine", "cvine", "dvine")
    _VALID_MARGINALS = ("parametric", "empirical")
    _VALID_CRITERIA = ("aic", "bic")

    def __init__(
        self,
        *,
        vine_type: str = "rvine",
        family_set: Union[str, List[str]] = "all",
        selection_criterion: str = "aic",
        marginal_method: str = "parametric",
        log_transform: bool = False,
        offset: float = 1.0,
        trunc_level: Optional[int] = None,
        name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(name=name, debug=debug)

        if vine_type not in self._VALID_VINE_TYPES:
            raise ValueError(
                f"vine_type must be one of {self._VALID_VINE_TYPES}, "
                f"got '{vine_type}'"
            )
        if marginal_method not in self._VALID_MARGINALS:
            raise ValueError(
                f"marginal_method must be one of {self._VALID_MARGINALS}, "
                f"got '{marginal_method}'"
            )
        if selection_criterion not in self._VALID_CRITERIA:
            raise ValueError(
                f"selection_criterion must be one of {self._VALID_CRITERIA}, "
                f"got '{selection_criterion}'"
            )

        self.vine_type = vine_type
        self.family_set = family_set
        self.selection_criterion = selection_criterion
        self.marginal_method = marginal_method
        self.log_transform = log_transform
        self.offset = offset
        self.trunc_level = trunc_level

        self.init_params.algorithm_params = {
            "method": f"Vine Copula ({vine_type})",
            "vine_type": vine_type,
            "family_set": (
                family_set if isinstance(family_set, str) else list(family_set)
            ),
            "selection_criterion": selection_criterion,
            "marginal_method": marginal_method,
            "log_transform": log_transform,
            "offset": offset,
            "trunc_level": trunc_level,
        }

        # Fitted attributes (populated by fit)
        self._marginal_params: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._nst: Optional[NormalScoreTransform] = None
        self._monthly_vines: Dict[int, Any] = {}  # month -> pv.Vinecop or None
        self._Q_monthly: Optional[pd.DataFrame] = None
        self._par_rho: Dict[Tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_frequency(self) -> str:
        return "MS"

    # ------------------------------------------------------------------
    # Preprocessing (shared with GaussianCopulaGenerator)
    # ------------------------------------------------------------------

    def preprocessing(
        self,
        Q_obs,
        *,
        sites: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Validate input and prepare monthly flow data.

        Parameters
        ----------
        Q_obs : pd.DataFrame or pd.Series
            Monthly streamflow with DatetimeIndex.
        sites : list of str, optional
            Subset of sites to use.
        """
        Q = self._store_obs_data(Q_obs, sites)
        self._n_sites = len(self._sites)

        # Resample daily/weekly to monthly if needed
        inferred = pd.infer_freq(Q.index[: min(30, len(Q))])
        if inferred is not None and (
            inferred.startswith("D") or inferred.startswith("W")
        ):
            self.logger.info("Resampling from %s to monthly (sum)", inferred)
            Q = Q.resample("MS").sum()

        if self._n_sites == 1:
            self.logger.warning(
                "Only 1 site provided. Vine copula dependence structure is "
                "trivial. Consider a univariate generator instead."
            )

        # Optional log transform
        if self.log_transform:
            Q = np.log(Q + self.offset)

        self._Q_monthly = Q.copy()
        self.update_state(preprocessed=True)
        self.logger.info(
            "Preprocessing complete: %d months, %d sites",
            len(Q),
            self._n_sites,
        )

    # ------------------------------------------------------------------
    # Marginal fitting helpers (shared with GaussianCopulaGenerator)
    # ------------------------------------------------------------------

    def _fit_parametric_marginals(self) -> None:
        """Fit gamma and log-normal per (month, site); select by BIC."""
        Q = self._Q_monthly
        self._marginal_params = {}

        for m in range(1, 13):
            mask = Q.index.month == m
            month_data = Q.loc[mask]

            for s_idx, site in enumerate(self._sites):
                vals = month_data[site].values
                vals = vals[np.isfinite(vals)]
                vals_pos = np.maximum(vals, 1e-6)

                best_bic = np.inf
                best_params: Dict[str, Any] = {}

                # Candidate 1: Gamma
                try:
                    ga, gloc, gscale = gamma_dist.fit(vals_pos, floc=0)
                    nll_gamma = -np.sum(
                        gamma_dist.logpdf(vals_pos, ga, loc=0, scale=gscale)
                    )
                    bic_gamma = _bic(len(vals_pos), nll_gamma, k=2)
                    if bic_gamma < best_bic:
                        best_bic = bic_gamma
                        best_params = {
                            "dist": "gamma",
                            "shape": float(ga),
                            "loc": 0.0,
                            "scale": float(gscale),
                            "bic": float(bic_gamma),
                        }
                except Exception:
                    self.logger.debug("Gamma fit failed for month=%d, site=%s", m, site)

                # Candidate 2: Log-normal
                try:
                    ls, lloc, lscale = lognorm.fit(vals_pos, floc=0)
                    nll_ln = -np.sum(lognorm.logpdf(vals_pos, ls, loc=0, scale=lscale))
                    bic_ln = _bic(len(vals_pos), nll_ln, k=2)
                    if bic_ln < best_bic:
                        best_bic = bic_ln
                        best_params = {
                            "dist": "lognorm",
                            "shape": float(ls),
                            "loc": 0.0,
                            "scale": float(lscale),
                            "bic": float(bic_ln),
                        }
                except Exception:
                    self.logger.debug(
                        "Log-normal fit failed for month=%d, site=%s", m, site
                    )

                if not best_params:
                    self.logger.warning(
                        "All parametric fits failed for month=%d, site=%s; "
                        "falling back to empirical normal scores.",
                        m,
                        site,
                    )
                    best_params = {"dist": "empirical_fallback"}

                self._marginal_params[(m, s_idx)] = best_params

    def _fit_empirical_marginals(self) -> None:
        """Fit NormalScoreTransform per (month, site)."""
        Q = self._Q_monthly
        values_by_group: Dict[tuple, np.ndarray] = {}

        for m in range(1, 13):
            mask = Q.index.month == m
            month_data = Q.loc[mask]
            for s_idx, site in enumerate(self._sites):
                vals = month_data[site].values
                vals = vals[np.isfinite(vals)]
                values_by_group[(m, s_idx)] = vals

        self._nst = NormalScoreTransform()
        self._nst.fit(values_by_group)

    def _pit_to_normal(self) -> Dict[int, np.ndarray]:
        """Apply PIT and normal score transform; return normal scores by month.

        Returns
        -------
        Dict[int, np.ndarray]
            Mapping from month (1-12) to array of shape (n_years, n_sites).
        """
        Q = self._Q_monthly
        scores_by_month: Dict[int, np.ndarray] = {}

        for m in range(1, 13):
            mask = Q.index.month == m
            month_data = Q.loc[mask].values  # (n_years, n_sites)
            z = np.zeros_like(month_data)

            for s_idx in range(self._n_sites):
                vals = month_data[:, s_idx]

                if self.marginal_method == "parametric":
                    params = self._marginal_params.get((m, s_idx), {})
                    dist_name = params.get("dist", "empirical_fallback")

                    if dist_name == "gamma":
                        u = gamma_dist.cdf(
                            np.maximum(vals, 1e-6),
                            params["shape"],
                            loc=params["loc"],
                            scale=params["scale"],
                        )
                    elif dist_name == "lognorm":
                        u = lognorm.cdf(
                            np.maximum(vals, 1e-6),
                            params["shape"],
                            loc=params["loc"],
                            scale=params["scale"],
                        )
                    else:
                        # Fallback to empirical
                        if self._nst is None:
                            self._fit_empirical_marginals()
                        z[:, s_idx] = self._nst.transform(vals, (m, s_idx))
                        continue

                    # Clip to avoid +/- inf from norm.ppf
                    u = np.clip(u, 1e-6, 1.0 - 1e-6)
                    z[:, s_idx] = norm.ppf(u)

                else:
                    # Empirical path uses NormalScoreTransform directly
                    z[:, s_idx] = self._nst.transform(vals, (m, s_idx))

            scores_by_month[m] = z

        return scores_by_month

    # ------------------------------------------------------------------
    # PAR(1) temporal model (shared with GaussianCopulaGenerator)
    # ------------------------------------------------------------------

    def _fit_par_residuals(
        self, scores_by_month: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Fit PAR(1) per (month, site) and return residuals.

        For each month transition m -> m+1, estimates the lag-1
        autocorrelation rho_m(s) from the normal scores.  The PAR(1)
        residuals are approximately i.i.d. and capture only spatial
        dependence (Pereira et al. 2017, Section 3.1).

        Parameters
        ----------
        scores_by_month : Dict[int, np.ndarray]
            Normal scores per month, shape (n_years, n_sites).

        Returns
        -------
        Dict[int, np.ndarray]
            PAR residuals per month, shape (n_years, n_sites).
        """
        self._par_rho = {}

        # Compute lag-1 autocorrelation for each month transition
        for m in range(1, 13):
            m_prev = 12 if m == 1 else m - 1
            z_curr = scores_by_month[m]
            z_prev = scores_by_month[m_prev]

            # Align: for Jan(m=1), pair with Dec of previous year
            if m == 1:
                z_prev = z_prev[:-1]
                z_curr = z_curr[1:]
                n_min = min(len(z_prev), len(z_curr))
                z_prev = z_prev[:n_min]
                z_curr = z_curr[:n_min]
            else:
                n_min = min(len(z_curr), len(z_prev))
                z_curr = z_curr[:n_min]
                z_prev = z_prev[:n_min]

            for s in range(self._n_sites):
                if len(z_curr) > 2:
                    std_c = np.std(z_curr[:, s])
                    std_p = np.std(z_prev[:, s])
                    if std_c > 1e-10 and std_p > 1e-10:
                        rho = float(np.corrcoef(z_prev[:, s], z_curr[:, s])[0, 1])
                        rho = np.clip(rho, -0.99, 0.99)
                    else:
                        rho = 0.0
                else:
                    rho = 0.0
                self._par_rho[(m, s)] = rho

        # Compute PAR residuals
        residuals_by_month: Dict[int, np.ndarray] = {}
        for m in range(1, 13):
            m_prev = 12 if m == 1 else m - 1
            z_curr = scores_by_month[m]
            z_prev = scores_by_month[m_prev]

            if m == 1:
                z_prev = z_prev[:-1]
                z_curr = z_curr[1:]
                n_min = min(len(z_prev), len(z_curr))
                z_prev = z_prev[:n_min]
                z_curr = z_curr[:n_min]
            else:
                n_min = min(len(z_curr), len(z_prev))
                z_curr = z_curr[:n_min]
                z_prev = z_prev[:n_min]

            e = np.zeros_like(z_curr)
            for s in range(self._n_sites):
                rho = self._par_rho[(m, s)]
                scale = np.sqrt(max(1.0 - rho**2, 1e-6))
                e[:, s] = (z_curr[:, s] - rho * z_prev[:, s]) / scale

            residuals_by_month[m] = e

        return residuals_by_month

    # ------------------------------------------------------------------
    # Vine copula helpers
    # ------------------------------------------------------------------

    def _resolve_family_set(self) -> list:
        """Resolve family_set parameter to pyvinecopulib BicopFamily enums.

        Returns
        -------
        list
            List of pyvinecopulib.BicopFamily members.
        """
        pv = _require_pyvinecopulib()

        FAMILY_MAP = {
            "gaussian": pv.BicopFamily.gaussian,
            "student_t": pv.BicopFamily.student,
            "student": pv.BicopFamily.student,
            "t": pv.BicopFamily.student,
            "clayton": pv.BicopFamily.clayton,
            "gumbel": pv.BicopFamily.gumbel,
            "frank": pv.BicopFamily.frank,
            "joe": pv.BicopFamily.joe,
            "bb1": pv.BicopFamily.bb1,
            "bb6": pv.BicopFamily.bb6,
            "bb7": pv.BicopFamily.bb7,
            "bb8": pv.BicopFamily.bb8,
            "independence": pv.BicopFamily.indep,
            "tll": pv.BicopFamily.tll,
        }

        ALL_PARAMETRIC = [
            pv.BicopFamily.gaussian,
            pv.BicopFamily.student,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,
            pv.BicopFamily.joe,
            pv.BicopFamily.bb1,
            pv.BicopFamily.bb6,
            pv.BicopFamily.bb7,
            pv.BicopFamily.bb8,
            pv.BicopFamily.indep,
        ]

        if self.family_set == "all":
            return ALL_PARAMETRIC
        elif isinstance(self.family_set, list):
            families = []
            for name in self.family_set:
                key = name.lower()
                if key not in FAMILY_MAP:
                    raise ValueError(
                        f"Unknown copula family '{name}'. "
                        f"Available: {list(FAMILY_MAP.keys())}"
                    )
                families.append(FAMILY_MAP[key])
            return families
        else:
            raise ValueError(
                f"family_set must be 'all' or a list of family names, "
                f"got '{self.family_set}'"
            )

    def _build_structure(self, n_sites: int):
        """Build a vine structure constraint for the given vine_type.

        Parameters
        ----------
        n_sites : int
            Number of variables.

        Returns
        -------
        pyvinecopulib structure or None
            Structure constraint, or None for R-vine (auto-selected).
        """
        pv = _require_pyvinecopulib()
        order = list(range(1, n_sites + 1))

        if self.vine_type == "dvine":
            return pv.DVineStructure(order=order)
        elif self.vine_type == "cvine":
            return pv.CVineStructure(order=order)
        else:
            return None  # R-vine: auto structure selection

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, Q_obs=None, *, sites=None, **kwargs) -> None:
        """Fit marginal distributions and vine copula dependence structure.

        Parameters
        ----------
        Q_obs : pd.DataFrame or pd.Series, optional
            If provided, calls preprocessing automatically.
        sites : list of str, optional
            Sites to use.
        """
        pv = _require_pyvinecopulib()

        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        # Step 1: Fit marginals (identical to GaussianCopulaGenerator)
        if self.marginal_method == "parametric":
            self._fit_parametric_marginals()
        else:
            self._fit_empirical_marginals()

        # Step 2: PIT + normal score transform
        scores_by_month = self._pit_to_normal()

        # Step 3: PAR(1) temporal model (Pereira et al. 2017)
        residuals_by_month = self._fit_par_residuals(scores_by_month)

        # Step 4: Vine copula on PAR residuals per month
        family_set = self._resolve_family_set()
        structure = self._build_structure(self._n_sites)

        controls_kwargs: Dict[str, Any] = {
            "family_set": family_set,
            "selection_criterion": self.selection_criterion,
        }
        if self.trunc_level is not None:
            controls_kwargs["trunc_lvl"] = self.trunc_level

        controls = pv.FitControlsVinecop(**controls_kwargs)

        # TODO: Pereira et al. (2017) spatial reparametrization of T1.
        # The current fit lets each per-month vine learn an unconstrained tree
        # 1 (T1) of d - 1 pair copulas. Pereira (2017) Eq. 11 instead
        # parametrizes T1 Kendall taus as
        #   g_z(tau) = beta_0 + beta_1 * ln(distance) + beta_2 * D
        #              + sum_q beta_q * S_q
        # where distance is inter-site distance, D is a basin indicator, and
        # S_q are seasonal harmonics. This reduces T1 from 12 * (d - 1) free
        # parameters to 8 across all months and is the distinguishing
        # contribution of Pereira (2017) for sparse-data multisite settings.
        # Implementing it requires an inter-site distance matrix as input,
        # constrained MLE across all 12 months jointly for T1, and per-month
        # unconstrained fits for higher trees only.
        self._monthly_vines = {}
        for m in range(1, 13):
            e = residuals_by_month[m]  # (n_years, n_sites)

            if self._n_sites == 1:
                self._monthly_vines[m] = None
                self.logger.debug("Month %d: single site, skipping vine", m)
                continue

            # Transform PAR residuals to uniform [0,1] via standard normal CDF
            u = norm.cdf(e)
            u = np.clip(u, 1e-6, 1.0 - 1e-6)

            try:
                vine = pv.Vinecop.from_data(u, controls=controls, structure=structure)
                self._monthly_vines[m] = vine
                self.logger.info(
                    "Month %d: fitted %s vine with %d parameters "
                    "on %d obs x %d sites",
                    m,
                    self.vine_type,
                    int(vine.npars),
                    u.shape[0],
                    u.shape[1],
                )
            except Exception as exc:
                self.logger.warning(
                    "Vine fitting failed for month %d: %s. "
                    "Falling back to independence copula.",
                    m,
                    exc,
                )
                self._monthly_vines[m] = None

        self.update_state(fitted=True)
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            "Fit complete: vine=%s, marginals=%s, %d sites",
            self.vine_type,
            self.marginal_method,
            self._n_sites,
        )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Ensemble:
        """Generate synthetic monthly streamflow realizations.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of independent realizations.
        n_years : int, optional
            Years per realization.  Defaults to length of the historic record.
        n_timesteps : int, optional
            Total monthly timesteps; overrides *n_years* if provided.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Ensemble
            Synthetic flow realizations.
        """
        self.validate_fit()

        rng = np.random.default_rng(seed)

        if n_timesteps is not None:
            total_months = n_timesteps
            n_years_gen = int(np.ceil(n_timesteps / 12))
        elif n_years is not None:
            total_months = n_years * 12
            n_years_gen = n_years
        else:
            total_months = len(self._Q_monthly)
            n_years_gen = total_months // 12

        start_date = self._Q_monthly.index[0]

        realizations: Dict[int, pd.DataFrame] = {}
        for r in range(n_realizations):
            Q_syn = self._generate_one(n_years_gen, total_months, rng=rng)
            dates = pd.date_range(start=start_date, periods=len(Q_syn), freq="MS")
            realizations[r] = pd.DataFrame(Q_syn, index=dates, columns=self._sites)

        metadata = EnsembleMetadata(
            generator_class=self.name or self.__class__.__name__,
            n_realizations=n_realizations,
            n_sites=self._n_sites,
            description=(
                f"Vine Copula ({self.vine_type}) with "
                f"{self.marginal_method} marginals"
            ),
        )

        self.logger.info(
            "Generated %d realizations of %d months x %d sites",
            n_realizations,
            total_months,
            self._n_sites,
        )
        return Ensemble(realizations, metadata=metadata)

    def _generate_one(
        self,
        n_years: int,
        total_months: int,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a single realization.

        Returns
        -------
        np.ndarray
            Shape (total_months, n_sites).
        """
        n_sites = self._n_sites
        n_months = n_years * 12
        Q = np.zeros((n_months, n_sites))

        # Pre-generate all vine samples in one batch per calendar month so that
        # pyvinecopulib's internal RNG is invoked exactly once per month type,
        # with a seed derived from the parent rng.  Each calendar month m
        # appears exactly n_years times in the n_months loop.
        _vine_draws: Dict[int, np.ndarray] = {}
        _vine_idx: Dict[int, int] = {}
        for _m in range(1, 13):
            _vine = self._monthly_vines.get(_m)
            if _vine is not None and n_sites > 1:
                _seed_m = int(rng.integers(0, 2**31))
                _vine_draws[_m] = _vine.simulate(n_years, seeds=[_seed_m])
                _vine_idx[_m] = 0

        z_prev = np.zeros(n_sites)

        for t in range(n_months):
            m = (t % 12) + 1  # calendar month 1-12

            vine = self._monthly_vines.get(m)

            if vine is not None and n_sites > 1:
                # Retrieve the pre-generated sample for this month occurrence
                u_new = _vine_draws[m][_vine_idx[m] : _vine_idx[m] + 1]
                _vine_idx[m] += 1
                # Transform uniform to normal: these are the PAR residuals
                e = norm.ppf(np.clip(u_new[0], 1e-8, 1.0 - 1e-8))
            else:
                # Independence fallback (single site or failed fit)
                e = rng.standard_normal(n_sites)

            # PAR(1) temporal recursion
            z = np.zeros(n_sites)
            for s_idx in range(n_sites):
                rho = self._par_rho.get((m, s_idx), 0.0)
                scale = np.sqrt(max(1.0 - rho**2, 1e-6))
                z[s_idx] = rho * z_prev[s_idx] + scale * e[s_idx]

            z_prev = z.copy()

            # Map from normal-score space to uniform
            u = norm.cdf(z)

            # Inverse marginal CDF
            for s_idx in range(n_sites):
                if self.marginal_method == "parametric":
                    params = self._marginal_params.get((m, s_idx), {})
                    dist_name = params.get("dist", "empirical_fallback")

                    if dist_name == "gamma":
                        Q[t, s_idx] = gamma_dist.ppf(
                            np.clip(u[s_idx], 1e-8, 1 - 1e-8),
                            params["shape"],
                            loc=params["loc"],
                            scale=params["scale"],
                        )
                    elif dist_name == "lognorm":
                        Q[t, s_idx] = lognorm.ppf(
                            np.clip(u[s_idx], 1e-8, 1 - 1e-8),
                            params["shape"],
                            loc=params["loc"],
                            scale=params["scale"],
                        )
                    else:
                        Q[t, s_idx] = self._nst.inverse_transform(
                            np.array([z[s_idx]]), (m, s_idx)
                        )[0]
                else:
                    Q[t, s_idx] = self._nst.inverse_transform(
                        np.array([z[s_idx]]), (m, s_idx)
                    )[0]

        # Undo log transform if applied
        if self.log_transform:
            Q = np.exp(Q) - self.offset

        # Enforce non-negativity
        Q = np.maximum(Q, 0.0)

        return Q[:total_months]

    # ------------------------------------------------------------------
    # Fitted params
    # ------------------------------------------------------------------

    def _compute_fitted_params(self) -> FittedParams:
        n = self._n_sites
        n_marginal = 12 * n * 2  # shape + scale per (month, site)
        n_par = 12 * n  # PAR(1) rho per month per site

        # Vine copula parameters: count across all months
        n_vine_params = 0
        for m in range(1, 13):
            vine = self._monthly_vines.get(m)
            if vine is not None:
                n_vine_params += int(vine.npars)

        n_params = n_marginal + n_par + n_vine_params

        training_period = (
            str(self._Q_monthly.index[0].date()),
            str(self._Q_monthly.index[-1].date()),
        )

        # Build vine summary for diagnostics
        vine_summaries = {}
        for m in range(1, 13):
            vine = self._monthly_vines.get(m)
            if vine is not None:
                vine_summaries[f"month_{m}"] = {
                    "n_params": int(vine.npars),
                    "trunc_level": vine.trunc_lvl,
                }
            else:
                vine_summaries[f"month_{m}"] = {
                    "n_params": 0,
                    "trunc_level": 0,
                }

        return FittedParams(
            means_=None,
            stds_=None,
            correlations_=None,
            distributions_={
                "vine_type": self.vine_type,
                "family_set": (
                    self.family_set
                    if isinstance(self.family_set, str)
                    else list(self.family_set)
                ),
                "selection_criterion": self.selection_criterion,
                "marginal_method": self.marginal_method,
                "marginal_params": {
                    str(k): v for k, v in self._marginal_params.items()
                },
                "vine_summaries": vine_summaries,
            },
            transformations_={
                "log_transform": self.log_transform,
                "offset": self.offset,
            },
            n_parameters_=n_params,
            sample_size_=len(self._Q_monthly),
            n_sites_=self._n_sites,
            training_period_=training_period,
        )
