# Contributing to SynHydro

Thank you for your interest in contributing to SynHydro. This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

Open an issue on GitHub with a clear description of the bug, including a minimal reproducible example and the version of SynHydro and Python you are using.

### Suggesting New Methods

SynHydro aims to be a comprehensive library for synthetic hydrologic timeseries generation. If you would like to suggest a new generation or disaggregation method, please open an issue with the following information:

- Method name and primary academic reference (journal paper with DOI)
- Brief description of the algorithm
- Properties preserved (e.g., autocorrelation, spatial correlation, marginal distribution)
- Whether you are willing to implement it

### Adding a New Generator or Disaggregator

All generators inherit from `synhydro.core.base.Generator` and follow the preprocessing, fit, generate pattern. All disaggregators inherit from `synhydro.core.base.Disaggregator`.

Before writing code:

1. Write an algorithm description in `docs/algorithms/` following the structure of existing pages (e.g. `thomas_fiering.md`)
2. Add the primary reference to `docs/references/references.md`
3. Verify the reference DOI resolves correctly

Code requirements:

- Type hints on all public methods
- NumPy-style docstrings on all public methods
- Use `logger = logging.getLogger(__name__)` instead of `print()`
- Follow existing naming conventions: `Q_obs`, `Q_syn`, `n_sites`, `n_realizations`, `n_years`
- No commented-out code blocks
- Include a module-level docstring citing the primary reference

Testing:

- Write tests in `tests/` covering initialization, preprocessing, fitting, and generation
- Tests should use the shared fixtures in `tests/conftest.py`
- Run the full test suite with `pytest tests/` before submitting

### Pull Requests

1. Fork the repository and create a feature branch from `main`
2. Make your changes following the coding standards above
3. Add or update tests as appropriate
4. Run `pytest tests/` and ensure all tests pass
5. Update documentation if you changed public APIs
6. Submit a pull request with a clear description of the changes

## Development Setup

```bash
git clone https://github.com/your-username/SynHydro.git
cd SynHydro
pip install -e ".[dev]"
pytest tests/
```

## Questions

If you have questions about contributing, feel free to open a discussion on GitHub or contact the maintainers.
