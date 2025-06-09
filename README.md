# SGLib
The Synthetic Generation Library (SGLib) provides a suite of different methods for generating synthetic timeseries, with a focus on hydrologic applications.

This package seeks to support multiple different synthetic generation models using a standardized API. All of the models are built on a common Abstract Base Class (`sglib.core.Generator`) which facilitates the standardized workflows.  Given the diversity of generation methods, the internal functionality of the different generators is customized, and each class (will eventually) require a corresponding `Option` argument. 

In addition to the basic generation capabilities, this package (will someday) provide:
- Diagnostic tools to verify the statistical properites of the generated series
- Drought identification and quantification tools
- Basic plotting resources
- Ensemble management (e.g., reformatting, transformations, and read/write to memory)

This package is a work in progress, and many of the methods are incomplete or unverified. 

## Currently supported methods:
- Kirsch-Nowak daily streamflow generation (`KirschNowakGenerator`)
- Thomas-Fiering monthly streamflow generation (`ThomasFieringGenerator`)

## Methods to be developed:
- Hidden Markov Model
- Autoregressive models
