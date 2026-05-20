"""
Example 1: Basic Synthetic Flow Generation

This example demonstrates the fundamental workflow for generating synthetic streamflow:
1. Load historical daily streamflow data
2. Fit a Thomas-Fiering generator (parametric AR(1) model)
3. Generate synthetic monthly flows as an Ensemble
4. Visualize results with timeseries and flow duration curves

Key parameters to explore:
- n_years: Number of years to generate
- n_realizations: Number of synthetic traces
- seed: Random seed for reproducibility
"""

import matplotlib.pyplot as plt
from synhydro import load_example_data, ThomasFieringGenerator
from synhydro.methods.generation.hybrid.kirsch import KirschGenerator

from synhydro.plotting import plot_timeseries, plot_flow_duration_curve

# ============================================================================
# Configuration
# ============================================================================
N_YEARS = 10  # Years of synthetic data to generate
N_REALIZATIONS = 50  # Number of synthetic traces
SEED = 42  # Random seed for reproducibility

# ============================================================================
# Load historical data
# ============================================================================
Q_daily = load_example_data("usgs_daily_streamflow_cms")
site = Q_daily.columns[0]  # Use first site

# ============================================================================
# Fit Thomas-Fiering generator
# ============================================================================

generator = "Kirsch"  # options: 'ThomasFiering', 'Kirsch', etc.

if generator == "ThomasFiering":
    monthly_gen = ThomasFieringGenerator(name="TF_Example", seed=SEED)
    monthly_gen.fit(Q_daily[site])
elif generator == "Kirsch":

    monthly_gen = KirschGenerator(name="Kirsch_Example", seed=SEED)
    monthly_gen.fit(Q_daily[site])


# ============================================================================
# Generate synthetic flows as Ensemble
# ============================================================================
ensemble = monthly_gen.generate(n_realizations=N_REALIZATIONS, n_years=N_YEARS)

# ============================================================================
# Visualizations
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Timeseries with bands
ax = axes[0]
plot_timeseries(
    ensemble,
    observed=monthly_gen.Q_obs_monthly,
    site=site,
    ax=ax,
    title=f"Synthetic Monthly Flows - {site}",
    ylabel="Flow (cms)",
    percentiles=[10, 25, 75, 90],
)

# Plot 2: Flow Duration Curve
ax = axes[1]
plot_flow_duration_curve(
    ensemble,
    observed=monthly_gen.Q_obs_monthly,
    site=site,
    ax=ax,
    title="Flow Duration Curve Comparison",
    ylabel="Flow (cms)",
)

plt.tight_layout()
plt.savefig("examples/figures/01_basic_generation.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Generated {N_REALIZATIONS} realizations of {N_YEARS} years")
print(f"Figure saved to examples/figures/01_basic_generation.png")
