# SGLib
The Synthetic Generation Library (SGLib) provides a suite of different methods for generating synthetic timeseries, with a focus on hydrologic applications.


Package structure:
.
├── sglib/
│   ├── core/
│   │   ├── base.py
│   │   └── utilities.py
│   ├── methods/
│   │   ├── parametric/
│   │   │   ├── autoregressive.py
│   │   │   ├── arima.py
│   │   │   └── hmm.py
│   │   ├── machine_learning
│   │   └── non_parametric
│   └── plotting/
│       └── timeseries.py
├── tests
├── examples
└── docs