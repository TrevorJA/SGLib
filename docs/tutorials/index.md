# Tutorials

Step-by-step guides, each focused on a single SynHydro workflow.

Every tutorial is a Jupyter notebook that lives in the
[`examples/`](https://github.com/TrevorJA/SynHydro/tree/main/examples) directory
of the repository. Clone the repo and run them locally; each notebook writes its
figures to `examples/figures/<tutorial name>/` and its data outputs to
`examples/outputs/<tutorial name>/`.

| Tutorial | What you'll learn |
|----------|-------------------|
| [01 - Quickstart](01_quickstart.ipynb) | Three-step generator workflow: preprocessing, fit, generate |
| [02 - Multi-Site](02_multisite.ipynb) | Multi-site generation with spatial correlation preservation |
| [03 - Disaggregator](03_disaggregator.ipynb) | Monthly-to-daily disaggregation with `NowakDisaggregator` |
| [04 - Working with Ensembles](04_ensembles.ipynb) | The `Ensemble` data model: dual views, construction, statistics, HDF5 I/O |
| [05 - Verification & Validation](05_verification.ipynb) | `verify()` and `validate()` metrics, summaries, and plots |
| [06 - Plotting Walkthrough](06_plotting.ipynb) | Default plots for ensemble visualization and verification |
| [07 - Drought Analysis](07_drought_analysis.ipynb) | SSI calculation and drought event extraction |
| [08 - Pipeline](08_pipeline.ipynb) | End-to-end monthly-to-daily generation via `KirschNowakPipeline` |

All examples use `synhydro.load_example_data()`, which returns a multi-site
daily streamflow `DataFrame`.
