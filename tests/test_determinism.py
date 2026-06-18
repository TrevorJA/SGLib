"""
Determinism regression tests for on-demand realization regeneration.

These tests pin the contract required by downstream projects that store only
per-realization seeds (not the daily traces) and regenerate selected
realizations on demand: realization ``k`` (a global index) must be bit-for-bit
identical regardless of how many realizations are generated, how the index
range is partitioned across calls/MPI ranks, or whether it is regenerated in
isolation.

Covers ``KirschGenerator.generate``, ``NowakDisaggregator.disaggregate``, the
``synhydro.core.seeding`` helpers, and the end-to-end ``KirschNowakPipeline``.
"""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.seeding import (
    as_seed_sequence,
    realization_rng,
    spawn_realization_seed,
)
from synhydro.methods.generation.hybrid.kirsch import KirschGenerator
from synhydro.methods.disaggregation.temporal.nowak import NowakDisaggregator
from synhydro.pipelines import KirschNowakPipeline


MASTER_SEED = 20240618


@pytest.fixture
def daily_multisite():
    """Daily two-site DataFrame spanning ten full years (no freq set)."""
    dates = pd.date_range(start="2000-01-01", end="2009-12-31", freq="D")
    rng = np.random.default_rng(0)
    data = {
        "site_1": rng.gamma(shape=2.0, scale=50.0, size=len(dates)),
        "site_2": rng.gamma(shape=2.5, scale=40.0, size=len(dates)),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.freq = None
    return df


@pytest.fixture
def fitted_kirsch(daily_multisite):
    gen = KirschGenerator()
    gen.fit(daily_multisite)
    return gen


def _realization_array(ensemble, k):
    """Return realization ``k`` of an ensemble as a float ndarray."""
    return ensemble.data_by_realization[k].to_numpy()


class TestSeedingHelpers:
    """Unit-level guarantees of the global-index seeding scheme."""

    def test_child_seed_independent_of_ensemble_size(self):
        """child_k must not depend on how many siblings are spawned."""
        master = as_seed_sequence(MASTER_SEED)
        reconstructed = spawn_realization_seed(master, 3)
        small = np.random.SeedSequence(MASTER_SEED).spawn(10)[3]
        large = np.random.SeedSequence(MASTER_SEED).spawn(100)[3]
        assert reconstructed.spawn_key == small.spawn_key == large.spawn_key
        assert reconstructed.entropy == small.entropy == large.entropy

    def test_substreams_are_distinct_but_reproducible(self):
        """The generation and disaggregation streams differ yet are reproducible."""
        master = as_seed_sequence(MASTER_SEED)
        gen_a = realization_rng(master, 7, "generation").random(16)
        gen_b = realization_rng(as_seed_sequence(MASTER_SEED), 7, "generation").random(
            16
        )
        disagg = realization_rng(master, 7, "disaggregation").random(16)
        np.testing.assert_array_equal(gen_a, gen_b)
        assert not np.array_equal(gen_a, disagg)

    def test_unknown_stage_raises(self):
        master = as_seed_sequence(MASTER_SEED)
        with pytest.raises(ValueError, match="pipeline stage"):
            realization_rng(master, 0, "hourly")


class TestKirschDeterminism:
    """Global-index determinism for the monthly Kirsch generator."""

    def test_same_seed_array_equal_across_runs(self, fitted_kirsch):
        ens_a = fitted_kirsch.generate(n_realizations=4, n_years=3, seed=MASTER_SEED)
        ens_b = fitted_kirsch.generate(n_realizations=4, n_years=3, seed=MASTER_SEED)
        for k in range(4):
            np.testing.assert_array_equal(
                _realization_array(ens_a, k), _realization_array(ens_b, k)
            )

    def test_realization_k_invariant_to_N(self, fitted_kirsch):
        """Realization 3 is identical whether N=5 or N=50."""
        small = fitted_kirsch.generate(n_realizations=5, n_years=3, seed=MASTER_SEED)
        large = fitted_kirsch.generate(n_realizations=50, n_years=3, seed=MASTER_SEED)
        np.testing.assert_array_equal(
            _realization_array(small, 3), _realization_array(large, 3)
        )

    def test_partition_independence(self, fitted_kirsch):
        """Disjoint index subsets reproduce the same realizations as one run."""
        full = fitted_kirsch.generate(n_realizations=6, n_years=3, seed=MASTER_SEED)
        part_a = fitted_kirsch.generate(
            n_years=3, seed=MASTER_SEED, realization_indices=[0, 1, 2]
        )
        part_b = fitted_kirsch.generate(
            n_years=3, seed=MASTER_SEED, realization_indices=[3, 4, 5]
        )
        merged = {**part_a.data_by_realization, **part_b.data_by_realization}
        assert set(merged) == set(full.data_by_realization)
        for k in full.data_by_realization:
            np.testing.assert_array_equal(
                merged[k].to_numpy(), _realization_array(full, k)
            )

    def test_isolated_regeneration_matches(self, fitted_kirsch):
        """A single realization regenerated alone equals its slot in a batch."""
        batch = fitted_kirsch.generate(n_realizations=10, n_years=3, seed=MASTER_SEED)
        solo = fitted_kirsch.generate(
            n_years=3, seed=MASTER_SEED, realization_indices=[7]
        )
        np.testing.assert_array_equal(
            solo.data_by_realization[7].to_numpy(), _realization_array(batch, 7)
        )

    def test_no_dependence_on_global_numpy_state(self, fitted_kirsch):
        """Perturbing the legacy global RNG must not change output."""
        np.random.seed(1)
        ens_a = fitted_kirsch.generate(n_realizations=3, n_years=3, seed=MASTER_SEED)
        np.random.seed(999)
        _ = np.random.random(123)
        ens_b = fitted_kirsch.generate(n_realizations=3, n_years=3, seed=MASTER_SEED)
        for k in range(3):
            np.testing.assert_array_equal(
                _realization_array(ens_a, k), _realization_array(ens_b, k)
            )


class TestPipelineDeterminism:
    """End-to-end (generate then disaggregate) determinism, after float cast."""

    def _pipeline(self, daily_multisite):
        pipe = KirschNowakPipeline()
        pipe.fit(daily_multisite)
        return pipe

    def test_same_seed_array_equal_across_runs(self, daily_multisite):
        pipe = self._pipeline(daily_multisite)
        ens_a = pipe.generate(n_realizations=3, n_years=2, seed=MASTER_SEED)
        ens_b = pipe.generate(n_realizations=3, n_years=2, seed=MASTER_SEED)
        for k in range(3):
            np.testing.assert_array_equal(
                _realization_array(ens_a, k), _realization_array(ens_b, k)
            )

    def test_realization_k_invariant_to_N(self, daily_multisite):
        pipe = self._pipeline(daily_multisite)
        small = pipe.generate(n_realizations=3, n_years=2, seed=MASTER_SEED)
        large = pipe.generate(n_realizations=12, n_years=2, seed=MASTER_SEED)
        np.testing.assert_array_equal(
            _realization_array(small, 2), _realization_array(large, 2)
        )

    def test_partition_independence(self, daily_multisite):
        pipe = self._pipeline(daily_multisite)
        full = pipe.generate(n_realizations=6, n_years=2, seed=MASTER_SEED)
        part_a = pipe.generate(
            n_years=2, seed=MASTER_SEED, realization_indices=[0, 1, 2]
        )
        part_b = pipe.generate(
            n_years=2, seed=MASTER_SEED, realization_indices=[3, 4, 5]
        )
        merged = {**part_a.data_by_realization, **part_b.data_by_realization}
        assert set(merged) == set(full.data_by_realization)
        for k in full.data_by_realization:
            np.testing.assert_array_equal(
                merged[k].to_numpy(), _realization_array(full, k)
            )

    def test_nowak_uses_global_index_not_loop_order(self, daily_multisite):
        """Disaggregating a single-key ensemble matches that key in a full run.

        Guards against the disaggregator keying its RNG to loop position rather
        than the global realization index carried by the ensemble key.
        """
        pipe = self._pipeline(daily_multisite)
        gen, disagg = pipe.generator, pipe.disaggregator

        full_monthly = gen.generate(n_realizations=8, n_years=2, seed=MASTER_SEED)
        full_daily = disagg.disaggregate(full_monthly, seed=MASTER_SEED)

        solo_monthly = gen.generate(
            n_years=2, seed=MASTER_SEED, realization_indices=[5]
        )
        solo_daily = disagg.disaggregate(solo_monthly, seed=MASTER_SEED)

        np.testing.assert_array_equal(
            solo_daily.data_by_realization[5].to_numpy(),
            _realization_array(full_daily, 5),
        )
