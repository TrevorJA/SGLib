"""Configuration for model diagnostics."""

# Default ensemble settings (small for fast iteration)
N_REALIZATIONS = 3
N_YEARS = 30
SEED = 42
SITE_INDEX = 0
ACF_MAX_LAG = 36

# Generator registry
# Each entry: short key -> class name, input frequency, multisite flag, init_kwargs
GENERATORS = {
    "ThomasFiering": {
        "class_name": "ThomasFieringGenerator",
        "frequency": "monthly",
        "multisite": False,
        "init_kwargs": {},
    },
    "Matalas": {
        "class_name": "MatalasGenerator",
        "frequency": "monthly",
        "multisite": True,
        "init_kwargs": {"log_transform": True},
    },
    "Kirsch": {
        "class_name": "KirschGenerator",
        "frequency": "monthly",
        "multisite": True,
        "init_kwargs": {"generate_using_log_flow": True},
    },
    "Kirsch_Weekly": {
        "class_name": "KirschGenerator",
        "frequency": "weekly",
        "multisite": True,
        "init_kwargs": {"generate_using_log_flow": True},
    },
    "KNNBootstrap": {
        "class_name": "KNNBootstrapGenerator",
        "frequency": "monthly",
        "multisite": True,
        "init_kwargs": {},
    },
    "ARFIMA": {
        "class_name": "ARFIMAGenerator",
        "frequency": "monthly",
        "multisite": False,
        "init_kwargs": {"p": 1, "q": 0, "d_method": "whittle"},
    },
    "HMM": {
        "class_name": "MultiSiteHMMGenerator",
        "frequency": "annual",
        "multisite": True,
        "init_kwargs": {"n_states": 2, "covariance_type": "full"},
    },
    "WARM": {
        "class_name": "WARMGenerator",
        "frequency": "annual",
        "multisite": False,
        "init_kwargs": {
            "wavelet": "morl",
            "scales": 16,
            "ar_order": 1,
            "lower_bound": "obs_min",
        },
    },
    "SMARTA": {
        "class_name": "SMARTAGenerator",
        "frequency": "annual",
        "multisite": True,
        "init_kwargs": {"sma_order": 64},
    },
    "SPARTA": {
        "class_name": "SPARTAGenerator",
        "frequency": "monthly",
        "multisite": True,
        "init_kwargs": {},
    },
    "MultisitePhaseRandomization": {
        "class_name": "MultisitePhaseRandomizationGenerator",
        "frequency": "daily",
        "multisite": True,
        "init_kwargs": {},
    },
    "PhaseRandomization": {
        "class_name": "PhaseRandomizationGenerator",
        "frequency": "daily",
        "multisite": False,
        "init_kwargs": {"marginal": "kappa"},
    },
}
