# Tutorials

Step-by-step guides, each focused on a single SynHydro workflow.

| Tutorial | What you'll learn |
|----------|-------------------|
| [01 - Quickstart](01_quickstart.md) | Three-step generator workflow: preprocessing, fit, generate |
| [02 - Multi-Site](02_multisite.md) | Multi-site generation with spatial correlation preservation |
| [03 - Disaggregator](03_disaggregator.md) | Monthly-to-daily disaggregation with `NowakDisaggregator` |
| [04 - Pipeline](04_pipeline.md) | End-to-end monthly-to-daily generation via `KirschNowakPipeline` |
| [05 - Drought Analysis](05_drought_analysis.md) | SSI calculation and drought event extraction |
| [06 - Ensemble Validation](06_validation.md) | `validate_ensemble` metrics and `plot_validation_panel` |
| [07 - Plotting Walkthrough](07_plotting.md) | Default plots for ensemble visualization and validation |

All examples use `synhydro.load_example_data()`, which returns a multi-site
daily streamflow `DataFrame`.
