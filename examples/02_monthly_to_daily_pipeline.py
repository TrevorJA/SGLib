"""
Example 2: Monthly-to-Daily Pipeline

This example demonstrates temporal disaggregation using the Kirsch-Nowak pipeline:
1. Load historical daily streamflow data
2. Use KirschNowakPipeline for integrated monthly generation + daily disaggregation
3. Generate daily synthetic flows as an Ensemble
4. Visualize daily flows with uncertainty bands
5. Save and load the ensemble

Key parameters to explore:
- n_neighbors: Number of KNN neighbors for Nowak disaggregation (default: 5)
- generate_using_log_flow: Whether Kirsch generates in log space (default: True)
- max_month_shift: Temporal flexibility in daily pattern matching (default: 7)
"""

import matplotlib.pyplot as plt
from synhydro import load_example_data, Ensemble
from synhydro.pipelines import KirschNowakPipeline
from synhydro.plotting import plot_timeseries, plot_flow_duration_curve

# ============================================================================
# Configuration
# ============================================================================
N_YEARS = 5  # Years of synthetic data to generate
N_REALIZATIONS = 30  # Number of synthetic traces
SEED = 123  # Random seed
N_NEIGHBORS = 5  # KNN neighbors for disaggregation
LOG_SPACE = True  # Generate in log space

# ============================================================================
# Load historical daily data
# ============================================================================
Q_daily = load_example_data("usgs_daily_streamflow_cms")

# ============================================================================
# Create and fit Kirsch-Nowak pipeline
# ============================================================================
pipeline = KirschNowakPipeline(
    generate_using_log_flow=LOG_SPACE, n_neighbors=N_NEIGHBORS, name="KN_Example"
)

pipeline.fit(Q_daily)

# ============================================================================
# Generate daily synthetic flows as Ensemble
# ============================================================================
ensemble = pipeline.generate(n_realizations=N_REALIZATIONS, n_years=N_YEARS, seed=SEED)

# ============================================================================
# Save and load ensemble
# ============================================================================
ensemble.to_hdf5("examples/kn_ensemble_example.h5", compression="gzip")
ensemble_loaded = Ensemble.from_hdf5("examples/kn_ensemble_example.h5")

# ============================================================================
# Visualizations
# ============================================================================
site = Q_daily.columns[0]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: Daily timeseries (subset for clarity)
ax = axes[0]
plot_timeseries(
    ensemble_loaded,
    observed=Q_daily[site],
    site=site,
    start_date=ensemble_loaded.data_by_realization[0].index[0],
    end_date=ensemble_loaded.data_by_realization[0].index[365],  # First year only
    ax=ax,
    title=f"Synthetic Daily Flows - {site} (Year 1)",
    ylabel="Flow (cms)",
    percentiles=[10, 50, 90],
    show_members=3,
)

# Plot 2: Daily Flow Duration Curve
ax = axes[1]
plot_flow_duration_curve(
    ensemble_loaded,
    observed=Q_daily[site],
    site=site,
    ax=ax,
    title="Daily Flow Duration Curve",
    ylabel="Flow (cms)",
    log_scale=True,
)

plt.tight_layout()
plt.savefig(
    "examples/figures/02_monthly_to_daily_pipeline.png", dpi=150, bbox_inches="tight"
)
plt.close()

print(f"Generated {N_REALIZATIONS} realizations of {N_YEARS} years at daily resolution")
print(f"Ensemble saved to examples/kn_ensemble_example.h5")
print(f"Figure saved to examples/figures/02_monthly_to_daily_pipeline.png")
