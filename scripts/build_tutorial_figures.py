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
    plot_monthly_distributions,
    plot_spatial_correlation,
    plot_ssi_timeseries,
    plot_timeseries,
    plot_validation_panel,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images" / "tutorials"

N_REALIZATIONS = 20
N_YEARS = 20
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


def build_pipeline_figure(Q_daily) -> None:
    logger.info("Tutorial 03: Kirsch-Nowak pipeline")
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
    _save(fig, "03_daily_timeseries.png")


def build_ssi_figure(Q_monthly) -> None:
    logger.info("Tutorial 04: SSI drought analysis")
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
    _save(fig, "04_ssi_with_droughts.png")


def build_validation_figure(Q_monthly) -> None:
    logger.info("Tutorial 05: Validation panel")
    site = Q_monthly.columns[0]

    gen = synhydro.KirschGenerator()
    gen.fit(Q_monthly)
    ensemble = gen.generate(
        n_realizations=N_REALIZATIONS * 2, n_years=N_YEARS, seed=SEED
    )

    fig, _ = plot_validation_panel(ensemble, observed=Q_monthly[site], site=site)
    _save(fig, "05_validation_panel.png")


def build_plotting_walkthrough_figures(Q_monthly) -> None:
    logger.info("Tutorial 06: Plotting walkthrough")
    site = Q_monthly.columns[0]

    gen = synhydro.KirschGenerator()
    gen.fit(Q_monthly)
    ensemble = gen.generate(n_realizations=N_REALIZATIONS, n_years=N_YEARS, seed=SEED)

    fig, _ = plot_timeseries(
        ensemble, observed=Q_monthly[site], site=site, show_members=3
    )
    _save(fig, "06_timeseries.png")

    fig, _ = plot_flow_duration_curve(ensemble, observed=Q_monthly[site], site=site)
    _save(fig, "06_fdc.png")

    fig, _ = plot_monthly_distributions(
        ensemble, observed=Q_monthly[site], site=site, plot_type="box"
    )
    _save(fig, "06_monthly_dist.png")

    fig, _ = plot_validation_panel(ensemble, observed=Q_monthly[site], site=site)
    _save(fig, "06_validation_panel.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", OUT_DIR.relative_to(REPO_ROOT))

    Q_daily = synhydro.load_example_data()
    Q_monthly = Q_daily.resample("MS").sum()

    build_quickstart_figures(Q_monthly)
    build_multisite_figure(Q_monthly)
    build_pipeline_figure(Q_daily)
    build_ssi_figure(Q_monthly)
    build_validation_figure(Q_monthly)
    build_plotting_walkthrough_figures(Q_monthly)

    logger.info("All tutorial figures built.")


if __name__ == "__main__":
    main()
