"""
Metric registry for the evaluation suites.

Each metric is a plain function registered with metadata describing how
the runner should evaluate and report it. Four metric kinds exist:

- ``scalar``: ``f(x: pd.Series, **opts) -> float``. One value per site
  per series. Computed on observed and on every realization.
- ``curve``: ``f(x: pd.Series, **opts) -> pd.Series``. One value per
  component (calendar month, lag, exceedance probability, period band).
  Computed on observed and on every realization.
- ``matrix``: ``f(frame: pd.DataFrame, **opts) -> pd.Series`` indexed by
  site-pair label. Computed on observed and on every realization.
- ``comparison``: ``f(x: pd.Series, reference: pd.Series, **opts)``
  returning a float or pd.Series. Measures the divergence of one
  realization from observed; there is no observed value of the metric
  itself.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

import pandas as pd

METRIC_KINDS = frozenset({"scalar", "curve", "matrix", "comparison"})
SUMMARY_MODES = frozenset({"distribution", "reject_rate"})


@dataclass(frozen=True)
class MetricSpec:
    """
    Registered metadata for a single metric.

    Attributes
    ----------
    name : str
        Unique metric name within a registry.
    category : str
        Metric category (e.g. ``'marginal'``, ``'temporal'``).
    kind : str
        One of ``'scalar'``, ``'curve'``, ``'matrix'``, ``'comparison'``.
    func : Callable
        The metric function.
    needs : tuple of str
        MetricContext attribute names injected as keyword arguments.
    frequencies : frozenset of str or None
        Frequencies the metric supports; None means any frequency.
    min_years : float or None
        Minimum observed record length in years; None means no minimum.
    units : str
        Reported units label (e.g. ``'flow'``, ``'dimensionless'``,
        ``'pvalue'``, ``'per_year'``, ``'timesteps'``).
    summary_mode : str
        ``'distribution'`` (default) or ``'reject_rate'`` for p-value
        metrics summarized as a rejection rate at alpha = 0.05.
    citation : str
        Short citation for the metric (author, year).
    description : str
        One-line description used in metric inventories.
    """

    name: str
    category: str
    kind: str
    func: Callable
    needs: tuple = ()
    frequencies: Optional[frozenset] = None
    min_years: Optional[float] = None
    units: str = "dimensionless"
    summary_mode: str = "distribution"
    citation: str = ""
    description: str = ""


class MetricRegistry:
    """
    Name-keyed collection of MetricSpec entries for one suite.

    Parameters
    ----------
    suite : str
        Suite label, e.g. ``'verification'`` or ``'validation'``.
    """

    def __init__(self, suite: str) -> None:
        self.suite = suite
        self._specs: dict[str, MetricSpec] = {}

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        category: str = "custom",
        kind: str = "scalar",
        needs: Iterable[str] = (),
        frequencies: Optional[Iterable[str]] = None,
        min_years: Optional[float] = None,
        units: str = "dimensionless",
        summary_mode: str = "distribution",
        citation: str = "",
        description: str = "",
    ) -> Callable:
        """
        Register a metric function, usable as a decorator or plain call.

        Parameters
        ----------
        func : Callable, optional
            Metric function. If omitted, returns a decorator.
        name : str, optional
            Metric name; defaults to ``func.__name__``.
        category : str, default 'custom'
            Metric category.
        kind : str, default 'scalar'
            Metric kind; see module docstring.
        needs : iterable of str, optional
            MetricContext attributes to inject as keyword arguments.
        frequencies : iterable of str, optional
            Supported base frequencies; None means any.
        min_years : float, optional
            Minimum observed record length in years.
        units : str, default 'dimensionless'
            Units label for reporting.
        summary_mode : str, default 'distribution'
            Summary behavior; ``'reject_rate'`` for p-value metrics.
        citation : str, optional
            Short citation.
        description : str, optional
            One-line description; defaults to the first docstring line.

        Returns
        -------
        Callable
            The registered function (unchanged), or a decorator.

        Raises
        ------
        ValueError
            If the name is already registered, or kind or summary_mode
            is invalid.
        """

        def _register(f: Callable) -> Callable:
            metric_name = name or f.__name__
            if metric_name in self._specs:
                raise ValueError(
                    f"Metric '{metric_name}' is already registered in the "
                    f"{self.suite} registry."
                )
            if kind not in METRIC_KINDS:
                raise ValueError(
                    f"Invalid metric kind '{kind}'. Valid kinds: "
                    f"{sorted(METRIC_KINDS)}."
                )
            if summary_mode not in SUMMARY_MODES:
                raise ValueError(
                    f"Invalid summary_mode '{summary_mode}'. Valid modes: "
                    f"{sorted(SUMMARY_MODES)}."
                )
            desc = description
            if not desc and f.__doc__:
                desc = f.__doc__.strip().splitlines()[0]
            self._specs[metric_name] = MetricSpec(
                name=metric_name,
                category=category,
                kind=kind,
                func=f,
                needs=tuple(needs),
                frequencies=(
                    frozenset(frequencies) if frequencies is not None else None
                ),
                min_years=min_years,
                units=units,
                summary_mode=summary_mode,
                citation=citation,
                description=desc,
            )
            return f

        if func is not None:
            return _register(func)
        return _register

    def names(self) -> list[str]:
        """Return all registered metric names."""
        return list(self._specs)

    def categories(self) -> list[str]:
        """Return the distinct categories in registration order."""
        seen: dict[str, None] = {}
        for spec in self._specs.values():
            seen.setdefault(spec.category, None)
        return list(seen)

    def get(self, name: str) -> MetricSpec:
        """Return the MetricSpec for a metric name."""
        if name not in self._specs:
            raise KeyError(f"Unknown metric '{name}' in the {self.suite} registry.")
        return self._specs[name]

    def select(
        self,
        metrics: Union[str, Iterable[Union[str, Callable]], None],
    ) -> list[MetricSpec]:
        """
        Resolve a metric selection to a list of MetricSpec entries.

        Parameters
        ----------
        metrics : str, iterable, or None
            ``'all'`` selects every registered metric. An iterable may
            mix metric names, category names, and bare callables (a
            callable is wrapped as an unregistered scalar metric of
            category ``'custom'``). None is rejected: the caller must
            select metrics explicitly.

        Returns
        -------
        list of MetricSpec
            Resolved specs, deduplicated, in selection order.

        Raises
        ------
        ValueError
            If metrics is None, empty, or contains an unknown name.
        """
        if metrics is None:
            raise ValueError(
                f"No metrics selected. Pass metrics='all', a list of metric "
                f"or category names, or callables. Available categories: "
                f"{self.categories()}. Available metrics: {self.names()}."
            )
        if isinstance(metrics, str):
            if metrics == "all":
                return list(self._specs.values())
            metrics = [metrics]

        selected: dict[str, MetricSpec] = {}
        category_map: dict[str, list[MetricSpec]] = {}
        for spec in self._specs.values():
            category_map.setdefault(spec.category, []).append(spec)

        for item in metrics:
            if callable(item):
                name = getattr(item, "__name__", "custom_metric")
                desc = ""
                if item.__doc__:
                    desc = item.__doc__.strip().splitlines()[0]
                spec = MetricSpec(
                    name=name,
                    category="custom",
                    kind="scalar",
                    func=item,
                    description=desc,
                )
                selected[spec.name] = spec
            elif item in self._specs:
                selected[item] = self._specs[item]
            elif item in category_map:
                for spec in category_map[item]:
                    selected[spec.name] = spec
            else:
                raise ValueError(
                    f"Unknown metric or category '{item}'. Available "
                    f"categories: {self.categories()}. Available metrics: "
                    f"{self.names()}."
                )
        if not selected:
            raise ValueError(
                "Metric selection resolved to an empty set. Pass "
                "metrics='all', metric names, category names, or callables."
            )
        return list(selected.values())

    def to_frame(self) -> pd.DataFrame:
        """
        Return the metric inventory as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: name, category, kind, units, frequencies,
            min_years, citation, description.
        """
        rows = []
        for spec in self._specs.values():
            rows.append(
                {
                    "name": spec.name,
                    "category": spec.category,
                    "kind": spec.kind,
                    "units": spec.units,
                    "frequencies": (
                        "any"
                        if spec.frequencies is None
                        else ", ".join(sorted(spec.frequencies))
                    ),
                    "min_years": spec.min_years,
                    "citation": spec.citation,
                    "description": spec.description,
                }
            )
        return pd.DataFrame(rows)
