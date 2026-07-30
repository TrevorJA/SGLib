"""Build tutorial figures for the documentation site.

Runs each tutorial's plotting workflow with small ensembles and saves PNGs to
docs/assets/images/tutorials/. Designed for both local use and CI invocation
via the [runfigs] commit-message trigger in the docs workflow.

Usage:
    python scripts/build_tutorial_figures.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import synhydro
from synhydro.plotting import (
    plot_flow_duration_curve,
    plot_metric_curve,
    plot_metric_distributions,
    plot_monthly_distributions,
    plot_spatial_correlation,
    plot_ssi_timeseries,
    plot_timeseries,
    plot_verification_panel,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images" / "tutorials"

N_REALIZATIONS = 50
N_YEARS = 30
SEED = 42
DPI = 150


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", path.relative_to(REPO_ROOT))


def build_quickstart_figures(Q_monthly) -> None:
    logger.info("Tutorial 01: Thomas-Fiering quickstart")
    site = Q_monthly.columns[0]
    Q_single = Q_monthly[[site]]

    gen = synhydro.ThomasFieringGenerator()
    gen.fit(Q_single)
    ensemble = gen.generate(n_realizations=N_REALIZATIONS, n_years=N_YEARS, seed=SEED)

    fig, _ = plot_timeseries(ensemble, observed=Q_monthly[site], show_members=3)
    _save(fig, "01_timeseries.png")

    fig, _ = plot_flow_duration_curve(ensemble, observed=Q_monthly[site])
    _save(fig, "01_fdc.png")


def build_multisite_figure(Q_monthly) -> None:
    logger.info("Tutorial 02: Kirsch multisite")
    gen = synhydro.KirschGenerator()
    gen.fit(Q_monthly)
    ensemble = gen.generate(n_realizations=N_REALIZATIONS, n_years=N_YEARS, seed=SEED)

    fig, _ = plot_spatial_correlation(ensemble, observed=Q_monthly, timestep="monthly")
    _save(fig, "02_spatial_correlation.png")


def build_disaggregator_figure(Q_daily, Q_monthly) -> None:
    logger.info("Tutorial 03: Nowak disaggregator")
    site = Q_monthly.columns[0]

    gen = synhydro.KirschGenerator()
    gen.fit(Q_monthly)
    monthly_ensemble = gen.generate(
        n_realizations=N_REALIZATIONS, n_years=N_YEARS, seed=SEED
    )

    disagg = synhydro.NowakDisaggregator()
    disagg.fit(Q_daily)
    daily_ensemble = disagg.disaggregate(monthly_ensemble, seed=SEED)

    syn_start = monthly_ensemble.data_by_realization[0].index[0]
    start = syn_start.strftime("%Y-%m-%d")
    monthly_end = (syn_start + pd.DateOffset(years=5) - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    daily_end = (syn_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_timeseries(
        monthly_ensemble,
        observed=Q_monthly[site],
        site=site,
        ax=axes[0, 0],
        start_date=start,
        end_date=monthly_end,
        show_members=3,
        title="Monthly ensemble",
    )
    plot_timeseries(
        daily_ensemble,
        observed=Q_daily[site],
        site=site,
        ax=axes[0, 1],
        start_date=start,
        end_date=daily_end,
        show_members=3,
        title="Disaggregated daily ensemble",
    )
    plot_flow_duration_curve(
        monthly_ensemble,
        observed=Q_monthly[site],
        site=site,
        ax=axes[1, 0],
        title="Monthly FDC",
    )
    plot_flow_duration_curve(
        daily_ensemble,
        observed=Q_daily[site],
        site=site,
        ax=axes[1, 1],
        title="Daily FDC",
    )
    fig.tight_layout()
    _save(fig, "03_disaggregator_panels.png")


def build_pipeline_figure(Q_daily) -> None:
    logger.info("Tutorial 04: Kirsch-Nowak pipeline")
    pipeline = synhydro.KirschNowakPipeline()
    pipeline.fit(Q_daily)
    daily_ensemble = pipeline.generate(n_realizations=10, n_years=N_YEARS, seed=SEED)

    site = Q_daily.columns[0]
    sample_index = daily_ensemble.data_by_realization[0].index
    start_date = sample_index[0].strftime("%Y-%m-%d")
    end_date = (sample_index[0] + pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    fig, _ = plot_timeseries(
        daily_ensemble,
        observed=Q_daily[site],
        start_date=start_date,
        end_date=end_date,
        show_members=3,
    )
    _save(fig, "04_daily_timeseries.png")


def build_ssi_figure(Q_monthly) -> None:
    logger.info("Tutorial 05: SSI drought analysis")
    site = Q_monthly.columns[0]

    gen = synhydro.KirschGenerator()
    gen.fit(Q_monthly)
    ensemble = gen.generate(n_realizations=N_REALIZATIONS, n_years=N_YEARS, seed=SEED)

    fig, _ = plot_ssi_timeseries(
        ensemble,
        observed=Q_monthly[site],
        site=site,
        window=12,
        title=f"SSI-12 -- {site}",
    )
    _save(fig, "05_ssi_with_droughts.png")


def build_verification_figures(Q_monthly) -> None:
    logger.info("Tutorial 06: Verification and validation")
    site = Q_monthly.columns[0]

    gen = synhydro.KirschGenerator()
    gen.fit(Q_monthly)
    ensemble = gen.generate(
        n_realizations=N_REALIZATIONS * 2, n_years=N_YEARS, seed=SEED
    )

    fig, _ = plot_verification_panel(
        ensemble, observed=Q_monthly[site], site=site, seed=SEED
    )
    _save(fig, "06_verification_panel.png")

    result = synhydro.verify(
        ensemble, Q_monthly, metrics=["marginal", "fdc"], sites=[site]
    )
    fig, _ = plot_metric_distributions(
        result, metrics=["mean", "std", "skewness"], ncols=3
    )
    _save(fig, "06_metric_distributions.png")

    fig, _ = plot_metric_curve(result, "fdc", site=site)
    _save(fig, "06_fdc_curve.png")


def build_plotting_walkthrough_figures(Q_monthly) -> None:
    logger.info("Tutorial 07: Plotting walkthrough")
    site = Q_monthly.columns[0]

    gen = synhydro.KirschGenerator()
    gen.fit(Q_monthly)
    ensemble = gen.generate(n_realizations=N_REALIZATIONS, n_years=N_YEARS, seed=SEED)

    fig, _ = plot_timeseries(
        ensemble, observed=Q_monthly[site], site=site, show_members=3
    )
    _save(fig, "07_timeseries.png")

    fig, _ = plot_flow_duration_curve(ensemble, observed=Q_monthly[site], site=site)
    _save(fig, "07_fdc.png")

    fig, _ = plot_monthly_distributions(
        ensemble, observed=Q_monthly[site], site=site, plot_type="box"
    )
    _save(fig, "07_monthly_dist.png")

    fig, _ = plot_verification_panel(
        ensemble, observed=Q_monthly[site], site=site, seed=SEED
    )
    _save(fig, "07_verification_panel.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", OUT_DIR.relative_to(REPO_ROOT))

    Q_daily = synhydro.load_example_data()
    Q_monthly = Q_daily.resample("MS").sum()

    build_quickstart_figures(Q_monthly)
    build_multisite_figure(Q_monthly)
    build_disaggregator_figure(Q_daily, Q_monthly)
    build_pipeline_figure(Q_daily)
    build_ssi_figure(Q_monthly)
    build_verification_figures(Q_monthly)
    build_plotting_walkthrough_figures(Q_monthly)

    logger.info("All tutorial figures built.")


if __name__ == "__main__":
    main()
