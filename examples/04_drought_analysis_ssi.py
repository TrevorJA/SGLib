"""
Example 4: Drought Analysis with Standardized Streamflow Index (SSI)

This example demonstrates comprehensive drought analysis using SSI methodology:
1. Load historical daily streamflow data
2. Test and compare different probability distributions
3. Calculate SSI at multiple timescales
4. Extract and analyze drought characteristics
5. Visualize drought events and SSI timeseries

Key concepts:
- SSI standardizes streamflow to identify droughts
- Different distributions affect drought detection sensitivity
- Timescale (window) determines drought duration focus
- Drought metrics: duration, magnitude, severity, timing

Key parameters to explore:
- dist: Probability distribution ('gamma', 'lognorm', 'pearson3', etc.)
- window: Rolling window size (3=seasonal, 6=semi-annual, 12=annual droughts)
- timescale: 'M' for monthly or 'D' for daily analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from synhydro import (
    load_example_data,
    SSI,
    compare_distributions,
    list_distributions,
)
from synhydro.droughts import get_drought_metrics

# ============================================================================
# Configuration
# ============================================================================
SITE_INDEX = 3  # Which site to analyze (0-3)
WINDOW = 12  # Rolling window in months for SSI calculation
TIMESCALE = "M"  # Monthly analysis
ANALYSIS_START = "1950-01-01"  # Start of analysis period
ANALYSIS_END = "2020-12-31"  # End of analysis period

# ============================================================================
# Load and prepare data
# ============================================================================
print("=" * 80)
print("DROUGHT ANALYSIS WITH STANDARDIZED STREAMFLOW INDEX (SSI)")
print("=" * 80)
print()

# Load daily data
Q_daily = load_example_data("usgs_daily_streamflow_cms")
site = Q_daily.columns[SITE_INDEX]
print(f"Analyzing site: {site}")
print(f"Data period: {Q_daily.index.min().date()} to {Q_daily.index.max().date()}")
print(f"Total days: {len(Q_daily):,}")
print()

# Resample to monthly for this analysis
Q_monthly = Q_daily[site].resample("MS").sum()

# Subset to analysis period
Q_analysis = Q_monthly.loc[ANALYSIS_START:ANALYSIS_END]
print(
    f"Analysis period: {Q_analysis.index.min().date()} to {Q_analysis.index.max().date()}"
)
print(f"Total months: {len(Q_analysis):,}")
print()

# ============================================================================
# Step 1: Compare probability distributions
# ============================================================================
print("STEP 1: TESTING PROBABILITY DISTRIBUTIONS")
print("-" * 80)
print("Testing which distribution best fits the flow data...")
print()

# Test common hydrological distributions
comparison = compare_distributions(
    Q_analysis,
    distributions=["gamma", "lognorm", "pearson3", "weibull_min", "norm"],
    window=WINDOW,
)

print("Distribution Comparison Results (ranked by AIC):")
print(
    comparison[["distribution", "aic", "bic", "ks_pvalue", "ks_pass"]].to_string(
        index=False
    )
)
print()

best_dist = comparison.iloc[0]["distribution"]
best_aic = comparison.iloc[0]["aic"]
best_pvalue = comparison.iloc[0]["ks_pvalue"]

print(f"RECOMMENDED DISTRIBUTION: {best_dist}")
print(f"  - Lowest AIC: {best_aic:.2f}")
print(
    f"  - KS p-value: {best_pvalue:.4f} {'(PASS)' if best_pvalue > 0.05 else '(marginal)'}"
)
print()

# ============================================================================
# Step 2: Calculate SSI with recommended distribution
# ============================================================================
print("STEP 2: CALCULATING SSI")
print("-" * 80)
print(f"Using {best_dist} distribution with {WINDOW}-month window...")
print()

# Initialize SSI calculator with best distribution
ssi_calc = SSI(dist=best_dist, timescale=WINDOW, fit_freq="ME")

# Calculate SSI
ssi_values = ssi_calc.fit_transform(Q_analysis)

print(f"SSI Statistics:")
print(f"  Mean: {ssi_values.mean():.3f} (should be ~0)")
print(f"  Std Dev: {ssi_values.std():.3f} (should be ~1)")
print(f"  Min: {ssi_values.min():.3f}")
print(f"  Max: {ssi_values.max():.3f}")
print()

# ============================================================================
# Step 3: Extract drought metrics
# ============================================================================
print("STEP 3: EXTRACTING DROUGHT EVENTS")
print("-" * 80)
print("Identifying droughts (SSI <= -1 indicates critical drought)...")
print()

drought_metrics = get_drought_metrics(ssi_values)

if len(drought_metrics) > 0:
    print(f"Total droughts detected: {len(drought_metrics)}")
    print()

    # Summary statistics
    print("Drought Statistics:")
    print(f"  Average duration: {drought_metrics['duration'].mean():.1f} months")
    print(f"  Max duration: {drought_metrics['duration'].max():.0f} months")
    print(f"  Average severity: {drought_metrics['severity'].mean():.2f}")
    print(
        f"  Max severity: {drought_metrics['severity'].min():.2f}"
    )  # min because severity is negative
    print()

    # Top 5 most severe droughts
    print("Top 5 Most Severe Drought Events:")
    print("-" * 80)
    top_droughts = drought_metrics.nsmallest(5, "severity")
    for idx, row in top_droughts.iterrows():
        print(f"  {idx}. Start: {row['start'].date()}, End: {row['end'].date()}")
        print(
            f"     Duration: {row['duration']:.0f} months, Severity: {row['severity']:.2f}, "
            f"Magnitude: {row['magnitude']:.2f}"
        )
    print()
else:
    print("No critical droughts detected (SSI never dropped below -1)")
    print()

# ============================================================================
# Step 4: Visualizations
# ============================================================================
print("STEP 4: CREATING VISUALIZATIONS")
print("-" * 80)

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

# -----------------------------------------------------------------------------
# Plot 1: Original Flow Timeseries
# -----------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(Q_analysis.index, Q_analysis.values, "b-", linewidth=0.8, alpha=0.7)
ax1.set_ylabel("Flow (cms)", fontsize=10)
ax1.set_title(f"Monthly Streamflow - {site}", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(Q_analysis.index.min(), Q_analysis.index.max())

# -----------------------------------------------------------------------------
# Plot 2: SSI Timeseries with drought shading
# -----------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[1, :])

# Plot SSI line
ax2.plot(ssi_values.index, ssi_values.values, "k-", linewidth=1.2, label="SSI")

# Add drought threshold lines
ax2.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax2.axhline(
    -1, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Moderate drought"
)
ax2.axhline(
    -2, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Severe drought"
)

# Shade drought categories
ax2.fill_between(ssi_values.index, -3, -2, alpha=0.15, color="darkred", label="Extreme")
ax2.fill_between(ssi_values.index, -2, -1, alpha=0.15, color="red", label="Severe")
ax2.fill_between(ssi_values.index, -1, 0, alpha=0.15, color="orange", label="Moderate")
ax2.fill_between(ssi_values.index, 0, 3, alpha=0.08, color="blue", label="Wet")

ax2.set_ylabel(f"SSI ({WINDOW}-month)", fontsize=10)
ax2.set_title(
    f"Standardized Streamflow Index - {best_dist} distribution",
    fontsize=12,
    fontweight="bold",
)
ax2.set_ylim(-3.5, 3.5)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right", fontsize=8, ncol=3)
ax2.set_xlim(Q_analysis.index.min(), Q_analysis.index.max())

# -----------------------------------------------------------------------------
# Plot 3: Distribution comparison (AIC)
# -----------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[2, 0])
comparison_plot = comparison.sort_values("aic")
colors = ["green" if x else "red" for x in comparison_plot["ks_pass"]]
ax3.barh(
    comparison_plot["distribution"], comparison_plot["aic"], color=colors, alpha=0.6
)
ax3.set_xlabel("AIC (lower is better)", fontsize=9)
ax3.set_title("Distribution Fit Comparison", fontsize=10, fontweight="bold")
ax3.grid(True, alpha=0.3, axis="x")
ax3.invert_yaxis()
# Add legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="green", alpha=0.6, label="Passes KS test"),
    Patch(facecolor="red", alpha=0.6, label="Fails KS test"),
]
ax3.legend(handles=legend_elements, fontsize=7, loc="lower right")

# -----------------------------------------------------------------------------
# Plot 4: Drought duration histogram
# -----------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[2, 1])
if len(drought_metrics) > 0:
    ax4.hist(
        drought_metrics["duration"],
        bins=15,
        color="coral",
        alpha=0.7,
        edgecolor="black",
    )
    ax4.set_xlabel("Duration (months)", fontsize=9)
    ax4.set_ylabel("Frequency", fontsize=9)
    ax4.set_title("Drought Duration Distribution", fontsize=10, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.axvline(
        drought_metrics["duration"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {drought_metrics['duration'].mean():.1f} mo",
    )
    ax4.legend(fontsize=8)
else:
    ax4.text(
        0.5,
        0.5,
        "No droughts detected",
        ha="center",
        va="center",
        transform=ax4.transAxes,
        fontsize=12,
    )
    ax4.set_title("Drought Duration Distribution", fontsize=10, fontweight="bold")

# -----------------------------------------------------------------------------
# Plot 5: Drought severity vs duration scatter
# -----------------------------------------------------------------------------
ax5 = fig.add_subplot(gs[3, 0])
if len(drought_metrics) > 0:
    # Color by magnitude
    scatter = ax5.scatter(
        drought_metrics["duration"],
        -drought_metrics["severity"],  # Negative for plotting
        c=drought_metrics["magnitude"].abs(),
        s=100,
        alpha=0.6,
        cmap="YlOrRd",
        edgecolor="black",
        linewidth=0.5,
    )
    ax5.set_xlabel("Duration (months)", fontsize=9)
    ax5.set_ylabel("Peak Severity (|SSI|)", fontsize=9)
    ax5.set_title("Drought Severity vs Duration", fontsize=10, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label("Magnitude (|sum SSI|)", fontsize=8)
else:
    ax5.text(
        0.5,
        0.5,
        "No droughts detected",
        ha="center",
        va="center",
        transform=ax5.transAxes,
        fontsize=12,
    )
    ax5.set_title("Drought Severity vs Duration", fontsize=10, fontweight="bold")

# -----------------------------------------------------------------------------
# Plot 6: SSI histogram
# -----------------------------------------------------------------------------
ax6 = fig.add_subplot(gs[3, 1])
ax6.hist(
    ssi_values.dropna(),
    bins=50,
    color="steelblue",
    alpha=0.7,
    edgecolor="black",
    density=True,
)
# Overlay standard normal for comparison
x = np.linspace(-4, 4, 100)
ax6.plot(
    x,
    (1 / np.sqrt(2 * np.pi)) * np.exp(-(x**2) / 2),
    "r--",
    linewidth=2,
    label="Standard Normal",
)
ax6.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax6.axvline(-1, color="orange", linestyle="--", linewidth=1, alpha=0.7)
ax6.set_xlabel("SSI", fontsize=9)
ax6.set_ylabel("Density", fontsize=9)
ax6.set_title("SSI Distribution", fontsize=10, fontweight="bold")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis="y")

# Save figure
plt.savefig(
    "examples/figures/04_drought_analysis_ssi.png", dpi=150, bbox_inches="tight"
)
plt.close()

print(f"Figure saved to examples/figures/04_drought_analysis_ssi.png")
print()

# ============================================================================
# Step 5: Compare different window sizes
# ============================================================================
print("STEP 5: COMPARING DIFFERENT SSI TIMESCALES")
print("-" * 80)
print("Testing how window size affects drought detection...")
print()

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
windows = [3, 6, 12]
colors_list = ["purple", "blue", "green"]

for idx, (window, color) in enumerate(zip(windows, colors_list)):
    ax = axes[idx]

    # Calculate SSI for this window
    ssi_calc_temp = SSI(dist=best_dist, timescale=window, fit_freq="ME")
    ssi_temp = ssi_calc_temp.fit_transform(Q_analysis)
    droughts_temp = get_drought_metrics(ssi_temp)

    # Plot SSI
    ax.plot(
        ssi_temp.index,
        ssi_temp.values,
        color=color,
        linewidth=1.2,
        label=f"SSI-{window}",
    )

    # Add threshold lines
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(-1, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)

    # Shade drought periods
    ax.fill_between(ssi_temp.index, -1, 0, alpha=0.1, color="orange")
    ax.fill_between(ssi_temp.index, -4, -1, alpha=0.15, color="red")

    ax.set_ylabel(f"SSI-{window}", fontsize=10)
    ax.set_ylim(-3.5, 3.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Add info text
    n_droughts = len(droughts_temp)
    info_text = f"{n_droughts} droughts detected"
    if n_droughts > 0:
        avg_dur = droughts_temp["duration"].mean()
        info_text += f", avg duration: {avg_dur:.1f} months"
    ax.text(
        0.02,
        0.95,
        info_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

axes[-1].set_xlabel("Year", fontsize=10)
axes[0].set_title(
    f"SSI at Different Timescales - {site} ({best_dist} distribution)",
    fontsize=12,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig(
    "examples/figures/04_drought_analysis_timescales.png", dpi=150, bbox_inches="tight"
)
plt.close()

print(f"Comparison figure saved to examples/figures/04_drought_analysis_timescales.png")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Key Findings:")
print(f"  -Best-fit distribution: {best_dist}")
print(
    f"  -Total droughts detected: {len(drought_metrics) if len(drought_metrics) > 0 else 0}"
)
if len(drought_metrics) > 0:
    print(
        f"  -Average drought duration: {drought_metrics['duration'].mean():.1f} months"
    )
    print(f"  -Longest drought: {drought_metrics['duration'].max():.0f} months")
    most_severe = drought_metrics.loc[drought_metrics["severity"].idxmin()]
    print(
        f"  -Most severe drought: {most_severe['start'].date()} to {most_severe['end'].date()}"
    )
    print(f"    (SSI reached {most_severe['severity']:.2f})")
print()
print("Figures saved:")
print("  1. examples/figures/04_drought_analysis_ssi.png - Main analysis")
print("  2. examples/figures/04_drought_analysis_timescales.png - Timescale comparison")
print()
