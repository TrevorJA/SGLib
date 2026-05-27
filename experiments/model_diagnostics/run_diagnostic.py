"""Run diagnostic analysis for a single generator.

Usage:
    python run_diagnostic.py --generator Matalas
    python run_diagnostic.py --generator SMARTA --n_realizations 5 --n_years 50
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synhydro.utils.data import load_example_data
from synhydro.core.ensemble import Ensemble
from synhydro.core.validation import validate_ensemble

from config import GENERATORS, N_REALIZATIONS, N_YEARS, SEED, SITE_INDEX, ACF_MAX_LAG
from plotting import (
    fig_marginal_pdf,
    fig_seasonal_cycle,
    fig_seasonal_std,
    fig_acf,
    fig_fdc,
    fig_cross_correlation,
    fig_validation_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_diagnostics")


def _import_generator_class(class_name: str):
    """Dynamically import a generator class by name."""
    # Try parametric first, then nonparametric
    try:
        mod = __import__("synhydro.methods.generation", fromlist=[class_name])
        return getattr(mod, class_name)
    except AttributeError:
        raise ImportError(
            f"Generator class '{class_name}' not found in synhydro.methods.generation"
        )


def _prepare_data(frequency: str, multisite: bool, site_idx: int):
    """Load example data and prepare at the required frequency.

    Returns
    -------
    Q_input : pd.DataFrame
        Data to pass to gen.fit() at the correct frequency.
    Q_daily : pd.DataFrame
        Daily data for validation/plotting of daily generators.
    Q_weekly : pd.DataFrame
        Weekly (W-SUN) data for validation/plotting.
    Q_monthly : pd.DataFrame
        Monthly data for validation/plotting.
    Q_annual : pd.DataFrame
        Annual data for validation/plotting.
    """
    Q_daily = load_example_data()

    Q_weekly = Q_daily.resample("W-SUN").sum()
    Q_monthly = Q_daily.resample("MS").sum()
    Q_annual = Q_daily.resample("YS").sum()

    if frequency == "daily":
        Q_input = Q_daily
    elif frequency == "annual":
        Q_input = Q_annual
    elif frequency == "weekly":
        Q_input = Q_weekly
    else:
        Q_input = Q_monthly

    if not multisite:
        Q_input = Q_input.iloc[:, [site_idx]]

    return Q_input, Q_daily, Q_weekly, Q_monthly, Q_annual


def _compute_validation_summary(ensemble, Q_obs_at_freq):
    """Run validate_ensemble and extract per-category MARE.

    Returns a pd.Series indexed by category name, or None on failure.
    """
    try:
        result = validate_ensemble(
            ensemble,
            Q_obs_at_freq,
            metrics=["marginal", "temporal", "seasonal", "fdc"],
        )
        summary = result.summary
        if hasattr(summary, "category_scores"):
            scores = summary.category_scores
            return pd.Series(scores, name="MARE")
        elif isinstance(summary, dict):
            return pd.Series(summary, name="MARE")
        else:
            return None
    except Exception as e:
        logger.warning("Validation failed: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Run diagnostic for one generator")
    parser.add_argument(
        "--generator",
        "-g",
        required=True,
        help=f"Generator key. Options: {', '.join(GENERATORS.keys())}",
    )
    parser.add_argument("--n_realizations", "-r", type=int, default=N_REALIZATIONS)
    parser.add_argument("--n_years", "-y", type=int, default=N_YEARS)
    parser.add_argument("--seed", "-s", type=int, default=SEED)
    parser.add_argument("--site_index", type=int, default=SITE_INDEX)
    args = parser.parse_args()

    gen_key = args.generator
    if gen_key not in GENERATORS:
        logger.error(
            "Unknown generator '%s'. Available: %s", gen_key, list(GENERATORS.keys())
        )
        sys.exit(1)

    cfg = GENERATORS[gen_key]
    class_name = cfg["class_name"]
    frequency = cfg["frequency"]
    multisite = cfg["multisite"]
    init_kwargs = cfg.get("init_kwargs", {})

    output_dir = Path(__file__).parent / "outputs" / gen_key
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Diagnostic: %s (%s)", gen_key, class_name)
    logger.info(
        "Settings: %d realizations x %d years, seed=%d",
        args.n_realizations,
        args.n_years,
        args.seed,
    )
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    # 1. Load and prepare data
    logger.info("Loading example data...")
    Q_input, Q_daily, Q_weekly, Q_monthly, Q_annual = _prepare_data(
        frequency, multisite, args.site_index
    )
    logger.info("Input data: %s, shape %s", frequency, Q_input.shape)

    # 2. Instantiate and fit
    logger.info("Importing %s...", class_name)
    GenClass = _import_generator_class(class_name)
    gen = GenClass(**init_kwargs)

    logger.info("Fitting...")
    t0 = time.time()
    gen.fit(Q_input)
    fit_time = time.time() - t0
    logger.info("Fit complete in %.1fs", fit_time)

    # 3. Generate
    logger.info(
        "Generating %d realizations x %d years...", args.n_realizations, args.n_years
    )
    t0 = time.time()
    ensemble = gen.generate(
        n_realizations=args.n_realizations,
        n_years=args.n_years,
        seed=args.seed,
    )
    gen_time = time.time() - t0
    logger.info("Generation complete in %.1fs", gen_time)

    # 4. Save ensemble
    h5_path = output_dir / "ensemble.h5"
    try:
        ensemble.to_hdf5(str(h5_path))
        logger.info("Ensemble saved to %s", h5_path)
    except Exception as e:
        logger.warning("Could not save HDF5: %s", e)

    # 5. Determine observation data at ensemble frequency
    # Synthetic and observed must be at the SAME frequency, otherwise groupby-mean
    # plots compare apples (monthly sums) to oranges (daily means).
    sample_df = ensemble.data_by_realization[ensemble.realization_ids[0]]
    ens_freq = pd.infer_freq(sample_df.index[: min(24, len(sample_df))])

    if ens_freq is not None and (ens_freq.startswith("Y") or ens_freq.startswith("A")):
        Q_obs_for_plots = Q_annual
        is_annual = True
        is_daily = False
    elif ens_freq is not None and ens_freq.startswith("W"):
        Q_obs_for_plots = Q_weekly
        is_annual = False
        is_daily = False
    elif frequency == "daily":
        Q_obs_for_plots = Q_daily
        is_annual = False
        is_daily = True
    elif frequency == "weekly":
        Q_obs_for_plots = Q_weekly
        is_annual = False
        is_daily = False
    else:
        Q_obs_for_plots = Q_monthly
        is_annual = False
        is_daily = False

    # Match sites between ensemble and obs
    ens_sites = ensemble.site_names
    obs_sites = [s for s in Q_obs_for_plots.columns if s in ens_sites]
    if not obs_sites:
        # Univariate generator may have renamed; use positional match
        obs_sites = list(Q_obs_for_plots.columns[: len(ens_sites)])
        Q_obs_for_plots = Q_obs_for_plots[obs_sites]
        Q_obs_for_plots.columns = ens_sites
    else:
        Q_obs_for_plots = Q_obs_for_plots[obs_sites]

    site_idx = 0

    # 6. Validation
    logger.info("Running validation metrics...")
    val_summary = _compute_validation_summary(ensemble, Q_obs_for_plots)
    if val_summary is not None:
        val_path = output_dir / "validation.csv"
        val_summary.to_csv(val_path)
        logger.info("Validation saved to %s", val_path)
        logger.info("MARE by category:\n%s", val_summary.to_string())

    # 7. Figures
    logger.info("Generating diagnostic figures...")

    fig_marginal_pdf(ensemble, Q_obs_for_plots, site_idx, output_dir, gen_key)
    logger.info("  [1/7] Marginal PDF")

    if not is_annual:
        fig_seasonal_cycle(ensemble, Q_obs_for_plots, site_idx, output_dir, gen_key)
        logger.info("  [2/7] Seasonal cycle")

        fig_seasonal_std(ensemble, Q_obs_for_plots, site_idx, output_dir, gen_key)
        logger.info("  [3/7] Seasonal std")
    else:
        logger.info("  [2-3/7] Skipped (annual generator)")

    acf_lag = ACF_MAX_LAG if not is_annual else min(ACF_MAX_LAG, args.n_years // 3)
    fig_acf(ensemble, Q_obs_for_plots, site_idx, output_dir, gen_key, max_lag=acf_lag)
    logger.info("  [4/7] ACF")

    fig_fdc(ensemble, Q_obs_for_plots, site_idx, output_dir, gen_key)
    logger.info("  [5/7] FDC")

    if len(ens_sites) > 1:
        fig_cross_correlation(ensemble, Q_obs_for_plots, output_dir, gen_key)
        logger.info("  [6/7] Cross-correlation")
    else:
        logger.info("  [6/7] Skipped (single site)")

    if val_summary is not None:
        fig_validation_summary(val_summary, output_dir, gen_key)
        logger.info("  [7/7] Validation summary")
    else:
        logger.info("  [7/7] Skipped (no validation data)")

    logger.info("=" * 60)
    logger.info("Done. Results in %s", output_dir)
    logger.info("Fit: %.1fs | Generate: %.1fs", fit_time, gen_time)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
