"""
Demonstration of the improved Ensemble class.

This script shows how to:
1. Create ensembles from generator output
2. Save and load ensembles with metadata
3. Compute statistics and percentiles
4. Subset and resample ensembles
5. Use the new integrated API
"""

import numpy as np
import pandas as pd
from synhydro import ThomasFieringGenerator, EnsembleMetadata, Ensemble

print("=" * 80)
print("Ensemble Class Demonstration")
print("=" * 80)
print()

# =============================================================================
# 1. Create synthetic monthly flow data
# =============================================================================
print("1. Creating Synthetic Historical Data")
print("-" * 80)

np.random.seed(42)
dates = pd.date_range("2000-01-01", "2020-12-31", freq="MS")
flows = (
    1000
    + 500 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12)
    + np.random.normal(0, 100, len(dates))
)
flows = np.maximum(flows, 50)
Q_monthly = pd.Series(flows, index=dates, name="site_A")

print(f"Created {len(dates)} months of historical data")
print()

# =============================================================================
# 2. Fit generator and create ensemble
# =============================================================================
print("2. Fitting Generator and Creating Ensemble")
print("-" * 80)

tf_gen = ThomasFieringGenerator(name="TF_Demo", debug=False)
tf_gen.fit(Q_monthly)

print("Generator fitted successfully")
print()

# Create ensemble using generate() (returns Ensemble directly)
print("Creating ensemble using generator.generate()...")
ensemble = tf_gen.generate(n_realizations=5, n_years=10)

print(f"Ensemble created: {ensemble}")
print()

# =============================================================================
# 3. Explore ensemble structure
# =============================================================================
print("3. Exploring Ensemble Structure")
print("-" * 80)

print("Ensemble string representation:")
print(ensemble)
print()

print(f"Number of realizations: {len(ensemble.realization_ids)}")
print(f"Number of sites: {len(ensemble.site_names)}")
print(f"Site names: {ensemble.site_names}")
print()

print("Ensemble metadata:")
print(f"  Generator: {ensemble.metadata.generator_class}")
print(f"  Created: {ensemble.metadata.creation_timestamp}")
print(f"  Time period: {ensemble.metadata.time_period}")
print()

# =============================================================================
# 4. Access data in different representations
# =============================================================================
print("4. Accessing Data")
print("-" * 80)

# Access by realization
real_0 = ensemble.data_by_realization[0]
print(f"Realization 0 shape: {real_0.shape}")
print(f"Realization 0 head:")
print(real_0.head())
print()

# Access by site
site_data = ensemble.data_by_site["site_A"]
print(f"Site A data shape: {site_data.shape}")
print(f"Site A head (realizations as columns):")
print(site_data.head())
print()

# =============================================================================
# 5. Compute statistics
# =============================================================================
print("5. Computing Statistics")
print("-" * 80)

# Summary statistics
stats = ensemble.summary(by="site")
print("Summary statistics by site:")
print(stats)
print()

# Percentiles
percentiles = ensemble.percentile([10, 50, 90], by="site")
print("Percentiles for site_A (first 5 timesteps):")
print(percentiles["site_A"].head())
print()

# =============================================================================
# 6. Subset ensemble
# =============================================================================
print("6. Subsetting Ensemble")
print("-" * 80)

# Subset by realizations and time period
subset = ensemble.subset(
    realizations=[0, 1, 2], start_date="2000-01-01", end_date="2005-12-31"
)

print(f"Original ensemble: {len(ensemble.realization_ids)} realizations")
print(f"Subset ensemble: {len(subset.realization_ids)} realizations")
print(f"Subset time period: {subset.metadata.time_period}")
print()

# =============================================================================
# 7. Resample ensemble
# =============================================================================
print("7. Resampling Ensemble")
print("-" * 80)

# Resample to annual
annual_ensemble = ensemble.resample("AS")
print(f"Original frequency: Monthly ({len(ensemble.data_by_realization[0])} timesteps)")
print(
    f"Resampled frequency: Annual ({len(annual_ensemble.data_by_realization[0])} timesteps)"
)
print()

# =============================================================================
# 8. Save and load ensemble (with metadata)
# =============================================================================
print("8. Saving and Loading Ensemble")
print("-" * 80)

# Save with metadata
output_file = "test_ensemble.h5"
print(f"Saving ensemble to {output_file}...")
ensemble.to_hdf5(output_file, compression="gzip")
print("Saved successfully")
print()

# Load ensemble
print(f"Loading ensemble from {output_file}...")
loaded_ensemble = Ensemble.from_hdf5(output_file)
print(f"Loaded: {loaded_ensemble}")
print()

print("Verifying loaded metadata:")
print(f"  Generator: {loaded_ensemble.metadata.generator_class}")
print(f"  Realizations: {loaded_ensemble.metadata.n_realizations}")
print(f"  Sites: {loaded_ensemble.metadata.n_sites}")
print()

# =============================================================================
# 9. Demonstrate manual ensemble creation
# =============================================================================
print("9. Manual Ensemble Creation")
print("-" * 80)

# Create ensemble manually from dictionary
manual_data = {
    0: pd.DataFrame(
        {"site_A": np.random.randn(100), "site_B": np.random.randn(100)},
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    ),
    1: pd.DataFrame(
        {"site_A": np.random.randn(100), "site_B": np.random.randn(100)},
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    ),
}

custom_metadata = EnsembleMetadata(
    description="Manually created ensemble for testing", time_resolution="daily"
)

manual_ensemble = Ensemble(manual_data, metadata=custom_metadata)
print(f"Manual ensemble created: {manual_ensemble}")
print()

# =============================================================================
# 10. Demonstrate new data loading utility
# =============================================================================
print("10. Testing Data Loading Utilities")
print("-" * 80)

print("Loading example data with new utility:")
try:
    from synhydro import load_example_data

    print("  Loading default dataset...")
    example_data = load_example_data()
    print(f"  Loaded: {example_data.shape}")
    print(f"  Columns: {list(example_data.columns)[:3]}...")
except Exception as e:
    print(f"  Error: {e}")
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 80)
print("Demonstration Complete!")
print("=" * 80)
print()
print("Key Features Demonstrated:")
print("  -generator.generate() - Create ensemble from fitted generator")
print("  -Ensemble.to_hdf5() / from_hdf5() - Integrated I/O with metadata")
print("  -ensemble.summary() - Statistical summaries")
print("  -ensemble.percentile() - Percentile calculations")
print("  -ensemble.subset() - Flexible subsetting")
print("  -ensemble.resample() - Temporal resampling")
print("  -Dual representations - Access by site or realization")
print("  -Metadata tracking - Generator provenance")
print("  -load_example_data() - Easy example data loading")
print()

# Cleanup
import os

if os.path.exists(output_file):
    os.remove(output_file)
    print(f"Cleaned up {output_file}")
