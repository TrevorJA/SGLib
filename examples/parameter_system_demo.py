"""
Demonstration of the new Generator parameter system.

This script shows how to:
1. Access initialization parameters
2. Access fitted parameters
3. View comprehensive summaries
4. Get complete state information
"""

import numpy as np
import pandas as pd
from synhydro.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator

# Create synthetic monthly flow data for demonstration
np.random.seed(42)
dates = pd.date_range("2000-01-01", "2020-12-31", freq="MS")
flows = (
    1000
    + 500 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12)
    + np.random.normal(0, 100, len(dates))
)
flows = np.maximum(flows, 50)  # Ensure positive
Q_monthly = pd.Series(flows, index=dates, name="site_A")

print("=" * 80)
print("Generator Parameter System Demonstration")
print("=" * 80)
print()

# =============================================================================
# 1. Initialize Generator
# =============================================================================
print("1. Initializing Thomas-Fiering Generator")
print("-" * 80)

tf_gen = ThomasFieringGenerator(name="TF_Demo", debug=False)

print(f"Generator created: {tf_gen}")
print()

# =============================================================================
# 2. Access Initialization Parameters (before fitting)
# =============================================================================
print("2. Accessing Initialization Parameters")
print("-" * 80)

init_params = tf_gen.get_params()
print("Initialization parameters:")
for key, value in init_params.items():
    print(f"  {key}: {value}")
print()

# Can also directly access the dataclass
print("Direct access to init_params:")
print(tf_gen.init_params)
print()

# =============================================================================
# 3. Fit the Generator
# =============================================================================
print("3. Fitting the Generator")
print("-" * 80)

tf_gen.fit(Q_monthly)

print(f"Generator fitted: {tf_gen}")
print()

# =============================================================================
# 4. Access Fitted Parameters (after fitting)
# =============================================================================
print("4. Accessing Fitted Parameters")
print("-" * 80)

try:
    fitted_params = tf_gen.get_fitted_params()
    print("Fitted parameters available:")
    for key, value in fitted_params.items():
        if isinstance(value, (int, str, tuple)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: <{type(value).__name__}>")
    print()
except ValueError as e:
    print(f"Error: {e}")
    print()

# Direct access to fitted_params_ dataclass
print("Direct access to fitted_params_:")
print(tf_gen.fitted_params_)
print()

# =============================================================================
# 5. View Comprehensive Summary
# =============================================================================
print("5. Viewing Comprehensive Summary")
print("-" * 80)
print()

summary = tf_gen.summary()
print(summary)
print()

# =============================================================================
# 6. Get Complete State Information
# =============================================================================
print("6. Getting Complete State Information")
print("-" * 80)

state_info = tf_gen.get_state_info()
print("State information keys:", list(state_info.keys()))
print(f"  - Class: {state_info['class']}")
print(f"  - Is fitted: {state_info['is_fitted']}")
print(f"  - Fit timestamp: {state_info['fit_timestamp']}")
print()

# =============================================================================
# 7. Generate Synthetic Data
# =============================================================================
print("7. Generating Synthetic Data")
print("-" * 80)

Q_syn = tf_gen.generate(n_years=5, n_realizations=3)
first = Q_syn.data_by_realization[Q_syn.realization_ids[0]]
n_timesteps, n_sites = first.shape
print(f"Generated synthetic flows: {len(Q_syn.realization_ids)} realizations")
print(f"  {n_timesteps} timesteps x {n_sites} site(s) per realization")
print()

# =============================================================================
# 8. Access Specific Fitted Parameters
# =============================================================================
print("8. Accessing Specific Fitted Parameters")
print("-" * 80)

print("Monthly means (normalized space):")
print(tf_gen.fitted_params_.means_)
print()

print("Monthly standard deviations (normalized space):")
print(tf_gen.fitted_params_.stds_)
print()

print("Monthly lag-1 correlations:")
print(tf_gen.fitted_params_.correlations_)
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 80)
print("Demonstration Complete!")
print("=" * 80)
print()
print("Key Features:")
print("  -get_params() - Access initialization parameters")
print("  -get_fitted_params() - Access learned parameters")
print("  -summary() - Comprehensive formatted summary")
print("  -get_state_info() - Complete state dictionary")
print("  -Direct access via init_params and fitted_params_ attributes")
print()
