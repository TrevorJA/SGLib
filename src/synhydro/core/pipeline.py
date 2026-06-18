"""
Pipeline classes for composing generators and disaggregators.

This module provides the GeneratorDisaggregatorPipeline class that allows
flexible composition of any generator with any compatible disaggregator.
"""

import logging
from typing import Optional, Union, List, Dict, Any
import pickle

import pandas as pd

from synhydro.core.base import Generator, Disaggregator
from synhydro.core.ensemble import Ensemble


class GeneratorDisaggregatorPipeline:
    """
    Pipeline for composing a generator with a disaggregator.

    This class orchestrates the flow from generation to disaggregation,
    ensuring compatibility between components and managing the complete
    workflow.

    Data is not passed at construction time. Use ``preprocessing(Q_obs)``
    or ``fit(Q_obs)`` to supply observed flow data.

    Parameters
    ----------
    generator : Generator
        An unfitted generator instance (constructed without data).
    disaggregator : Disaggregator
        An unfitted disaggregator instance (constructed without data).
    name : str, optional
        Name for this pipeline instance.
    debug : bool, default=False
        Enable debug logging.

    Examples
    --------
    >>> from synhydro.methods.generation.hybrid.kirsch import KirschGenerator
    >>> from synhydro.methods.disaggregation.temporal.nowak import NowakDisaggregator
    >>> from synhydro.core.pipeline import GeneratorDisaggregatorPipeline
    >>>
    >>> # Create components (no data)
    >>> generator = KirschGenerator()
    >>> disaggregator = NowakDisaggregator()
    >>>
    >>> # Create pipeline
    >>> pipeline = GeneratorDisaggregatorPipeline(generator, disaggregator)
    >>>
    >>> # Fit and generate
    >>> pipeline.preprocessing(Q_daily)
    >>> pipeline.fit()
    >>> daily_ensemble = pipeline.generate(n_realizations=10, n_years=50)
    """

    def __init__(
        self,
        generator: Generator,
        disaggregator: Disaggregator,
        name: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the pipeline with generator and disaggregator components.

        Parameters
        ----------
        generator : Generator
            Generator instance for producing synthetic flows.
        disaggregator : Disaggregator
            Disaggregator instance for temporal disaggregation.
        name : str, optional
            Name for this pipeline.
        debug : bool, default=False
            Enable debug logging.

        Raises
        ------
        TypeError
            If components are not proper Generator/Disaggregator instances.
        """
        # Validate component types
        if not isinstance(generator, Generator):
            raise TypeError(
                f"generator must be a Generator instance, got {type(generator)}"
            )
        if not isinstance(disaggregator, Disaggregator):
            raise TypeError(
                f"disaggregator must be a Disaggregator instance, got {type(disaggregator)}"
            )

        # Store components
        self.generator = generator
        self.disaggregator = disaggregator

        # Set name
        self.name = (
            name
            or f"{generator.__class__.__name__}_{disaggregator.__class__.__name__}_Pipeline"
        )

        # Setup logging
        self.debug = debug
        self.logger = logging.getLogger(f"synhydro.{self.name}")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.logger.info(
            f"Pipeline created: {generator.__class__.__name__} -> {disaggregator.__class__.__name__}"
        )

    def validate_compatibility(self) -> None:
        """
        Validate that generator and disaggregator are compatible.

        Checks that the generator's output frequency matches the
        disaggregator's input frequency.

        Raises
        ------
        ValueError
            If frequencies are incompatible.
        """
        gen_output_freq = self.generator.output_frequency
        dis_input_freq = self.disaggregator.input_frequency

        if gen_output_freq != dis_input_freq:
            raise ValueError(
                f"Incompatible frequencies: Generator produces '{gen_output_freq}' "
                f"but Disaggregator expects '{dis_input_freq}'"
            )

        self.logger.debug(
            f"Compatibility validated: {gen_output_freq} -> {self.disaggregator.output_frequency}"
        )

    @property
    def is_preprocessed(self) -> bool:
        """
        Check if both components are preprocessed.

        Returns
        -------
        bool
            True if both generator and disaggregator are preprocessed.
        """
        return self.generator.is_preprocessed and self.disaggregator.is_preprocessed

    @property
    def is_fitted(self) -> bool:
        """
        Check if both components are fitted.

        Returns
        -------
        bool
            True if both generator and disaggregator are fitted.
        """
        return self.generator.is_fitted and self.disaggregator.is_fitted

    @property
    def output_frequency(self) -> str:
        """
        Get the final output frequency of the pipeline.

        Returns
        -------
        str
            Pandas frequency string (e.g., 'D' for daily).
        """
        return self.disaggregator.output_frequency

    def preprocessing(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        *,
        sites: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Preprocess both generator and disaggregator.

        Passes ``Q_obs`` to both components' ``preprocessing()`` methods
        in sequence.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed historical flow data.
        sites : list of str, optional
            Sites to use. If None, uses all columns.
        **kwargs
            Additional preprocessing parameters passed to both components.
        """
        self.logger.info("Starting pipeline preprocessing...")

        # Preprocess generator
        self.logger.info("Preprocessing generator...")
        self.generator.preprocessing(Q_obs, sites=sites, **kwargs)

        # Preprocess disaggregator
        self.logger.info("Preprocessing disaggregator...")
        self.disaggregator.preprocessing(Q_obs, sites=sites, **kwargs)

        self.logger.info("Pipeline preprocessing complete")

    def fit(
        self,
        Q_obs: Optional[Union[pd.Series, pd.DataFrame]] = None,
        *,
        sites: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Fit both generator and disaggregator.

        If ``Q_obs`` is provided, ``preprocessing()`` is called automatically
        before fitting. If omitted, a prior call to ``preprocessing()`` is
        required.

        After preprocessing, validates frequency compatibility between
        generator and disaggregator before fitting.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed data. If provided, runs preprocessing automatically.
        sites : list of str, optional
            Sites to use (only when Q_obs is provided).
        **kwargs
            Additional fitting parameters passed to both components.

        Raises
        ------
        ValueError
            If preprocessing has not been completed and Q_obs is not provided.
        """
        # If Q_obs provided, run preprocessing first
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)

        if not self.is_preprocessed:
            raise ValueError(
                "Pipeline must be preprocessed before fitting. "
                "Call preprocessing(Q_obs) first or pass Q_obs to fit()."
            )

        # Validate compatibility after preprocessing (frequencies now known)
        self.validate_compatibility()

        self.logger.info("Starting pipeline fitting...")

        # Fit generator
        self.logger.info("Fitting generator...")
        self.generator.fit(**kwargs)

        # Fit disaggregator
        self.logger.info("Fitting disaggregator...")
        self.disaggregator.fit(**kwargs)

        self.logger.info("Pipeline fitting complete")

    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        *,
        realization_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> Ensemble:
        """
        Generate and disaggregate synthetic flows through the pipeline.

        This method orchestrates the complete workflow:
        1. Generate monthly (or other coarse) synthetic flows using the generator
        2. Disaggregate to finer temporal resolution using the disaggregator
        3. Return the final ensemble

        The same ``seed`` is forwarded to both stages. Generators and
        disaggregators that key their per-realization RNG streams to the global
        realization index (e.g. ``KirschGenerator`` and ``NowakDisaggregator``)
        therefore produce a given realization bit-for-bit identically regardless
        of ``n_realizations`` or how ``realization_indices`` is partitioned
        across calls or MPI ranks.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of synthetic realizations to generate. Ignored when
            ``realization_indices`` is provided.
        n_years : int, optional
            Number of years to generate.
        n_timesteps : int, optional
            Number of timesteps to generate explicitly.
        seed : int or numpy.random.SeedSequence, optional
            Master seed for reproducibility, forwarded to both the generator and
            the disaggregator.
        realization_indices : sequence of int, optional
            Explicit GLOBAL realization indices to generate. If None, uses
            ``range(n_realizations)``. Forwarded to the generator; the
            disaggregator infers the same indices from the monthly ensemble keys.
        **kwargs
            Additional parameters passed to generator and disaggregator.

        Returns
        -------
        Ensemble
            Final disaggregated ensemble at the output frequency.

        Raises
        ------
        ValueError
            If pipeline has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError(
                "Pipeline must be fitted before generating. "
                "Call preprocessing() and fit() first."
            )

        self.logger.info(
            f"Generating {n_realizations} realizations through pipeline..."
        )

        # Step 1: Generate with generator
        self.logger.info(
            f"Step 1: Generating flows with {self.generator.__class__.__name__}..."
        )
        monthly_ensemble = self.generator.generate(
            n_realizations=n_realizations,
            n_years=n_years,
            n_timesteps=n_timesteps,
            seed=seed,
            realization_indices=realization_indices,
            **kwargs,
        )
        self.logger.info(
            f"Generated {len(monthly_ensemble.data_by_realization)} realizations "
            f"at {monthly_ensemble.frequency} frequency"
        )

        # Step 2: Disaggregate with disaggregator
        self.logger.info(
            f"Step 2: Disaggregating flows with {self.disaggregator.__class__.__name__}..."
        )
        daily_ensemble = self.disaggregator.disaggregate(
            monthly_ensemble, seed=seed, **kwargs
        )
        self.logger.info(
            f"Disaggregated to {daily_ensemble.frequency} frequency, "
            f"{len(daily_ensemble.data_by_realization)} realizations"
        )

        self.logger.info("Pipeline generation complete")

        return daily_ensemble

    def summary(self) -> str:
        """
        Generate a summary of the pipeline configuration and status.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"{self.name} Summary".center(80))
        lines.append("=" * 80)
        lines.append("")

        # Pipeline Information
        lines.append("Pipeline Information")
        lines.append("-" * 80)
        lines.append(f"Generator:               {self.generator.__class__.__name__}")
        lines.append(
            f"Disaggregator:           {self.disaggregator.__class__.__name__}"
        )
        lines.append(
            f"Flow:                    {self.generator.output_frequency} -> {self.disaggregator.output_frequency}"
        )
        lines.append("")

        # Status
        lines.append("Pipeline Status")
        lines.append("-" * 80)
        lines.append(f"Preprocessed:            {self.is_preprocessed}")
        lines.append(f"Fitted:                  {self.is_fitted}")
        lines.append("")

        # Generator Status
        lines.append("Generator Status")
        lines.append("-" * 80)
        lines.append(f"Type:                    {self.generator.__class__.__name__}")
        lines.append(f"Preprocessed:            {self.generator.is_preprocessed}")
        lines.append(f"Fitted:                  {self.generator.is_fitted}")
        if self.generator.is_preprocessed and hasattr(self.generator, "_sites"):
            lines.append(f"Sites:                   {self.generator.sites}")
        lines.append("")

        # Disaggregator Status
        lines.append("Disaggregator Status")
        lines.append("-" * 80)
        lines.append(
            f"Type:                    {self.disaggregator.__class__.__name__}"
        )
        lines.append(f"Preprocessed:            {self.disaggregator.is_preprocessed}")
        lines.append(f"Fitted:                  {self.disaggregator.is_fitted}")
        if self.disaggregator.is_preprocessed and hasattr(self.disaggregator, "_sites"):
            lines.append(f"Sites:                   {self.disaggregator.sites}")
        lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        gen_status = "fitted" if self.generator.is_fitted else "not fitted"
        dis_status = "fitted" if self.disaggregator.is_fitted else "not fitted"
        return (
            f"GeneratorDisaggregatorPipeline(\n"
            f"  generator={self.generator.__class__.__name__}({gen_status}),\n"
            f"  disaggregator={self.disaggregator.__class__.__name__}({dis_status}),\n"
            f"  flow={self.generator.output_frequency}->{self.disaggregator.output_frequency}\n"
            f")"
        )

    def save(self, filepath: str) -> None:
        """
        Save the entire pipeline to file.

        Saves both the generator and disaggregator, preserving their fitted state.

        Parameters
        ----------
        filepath : str
            Path to save the pipeline.

        Raises
        ------
        ValueError
            If pipeline is not fitted.
        """
        if not self.is_fitted:
            raise ValueError(
                "Cannot save unfitted pipeline. Call preprocessing() and fit() first."
            )

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        self.logger.info(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "GeneratorDisaggregatorPipeline":
        """
        Load a pipeline from file.

        Parameters
        ----------
        filepath : str
            Path to saved pipeline file.

        Returns
        -------
        GeneratorDisaggregatorPipeline
            Loaded pipeline instance.
        """
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)

        pipeline.logger.info(f"Pipeline loaded from {filepath}")
        return pipeline
