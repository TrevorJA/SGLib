"""
Deterministic per-realization seeding for reproducible ensemble generation.

This module implements a strict determinism contract: realization ``k`` (a
global index in ``[0, N)``) is fully determined by a single child RNG stream
derived from a master seed and keyed to the GLOBAL index ``k``, independent of
how many realizations are generated (``N``) or how they are partitioned across
loops, batches, or MPI ranks.

The child stream for global index ``k`` is::

    child_k = numpy.random.SeedSequence(master_seed).spawn(N)[k]

``numpy.random.SeedSequence.spawn`` assigns spawn key ``(k,)`` to the ``k``-th
child of a freshly constructed sequence regardless of how many children are
requested, so ``spawn(10)[3]`` and ``spawn(100)[3]`` are identical. This module
reconstructs ``child_k`` directly from its spawn key so a single realization can
be regenerated in isolation without materializing the other ``N - 1`` children.

Each ``child_k`` is split into labeled sub-streams (see ``SUBSTREAM_LABELS``) so
that distinct pipeline stages (generation and disaggregation) draw from
independent streams while remaining keyed to the same global index. This keeps
the generate-then-disaggregate handoff for realization ``k`` reproducible end to
end.

References
----------
NumPy SeedSequence spawning: https://numpy.org/doc/stable/reference/random/parallel.html
"""

from typing import Optional, Union

import numpy as np

# Pipeline-stage labels, one independent sub-stream per stage within child_k.
# Order defines the spawn index of each sub-stream; appending new labels is safe,
# but reordering or removing them changes the streams existing stages receive.
SUBSTREAM_LABELS = ("generation", "disaggregation")

SeedLike = Union[int, np.random.SeedSequence, None]


def as_seed_sequence(seed: SeedLike) -> np.random.SeedSequence:
    """
    Resolve a master seed to a concrete ``numpy.random.SeedSequence``.

    Parameters
    ----------
    seed : int, numpy.random.SeedSequence, or None
        Master seed. An integer or None is wrapped in a new ``SeedSequence``
        (None draws fresh OS entropy, yielding a non-reproducible master). An
        existing ``SeedSequence`` is returned unchanged.

    Returns
    -------
    numpy.random.SeedSequence
        The resolved master sequence. Resolve the master once per generation
        call and reuse it across realizations so child streams stay consistent.
    """
    if isinstance(seed, np.random.SeedSequence):
        return seed
    return np.random.SeedSequence(seed)


def spawn_realization_seed(
    master: np.random.SeedSequence, global_index: int
) -> np.random.SeedSequence:
    """
    Derive the child seed sequence for a single global realization index.

    Reconstructs ``master.spawn(N)[global_index]`` directly from its spawn key,
    so the result depends only on the master and ``global_index`` and never on
    ``N`` or the order in which realizations are produced.

    Parameters
    ----------
    master : numpy.random.SeedSequence
        Resolved master sequence (see :func:`as_seed_sequence`).
    global_index : int
        Global realization index in ``[0, N)``.

    Returns
    -------
    numpy.random.SeedSequence
        The child sequence for ``global_index``.
    """
    return np.random.SeedSequence(
        entropy=master.entropy,
        spawn_key=tuple(master.spawn_key) + (int(global_index),),
        pool_size=master.pool_size,
    )


def realization_rng(
    master: np.random.SeedSequence,
    global_index: int,
    stage: str,
) -> np.random.Generator:
    """
    Build the per-stage RNG for a single global realization.

    The child seed for ``global_index`` is split into one sub-stream per entry
    in :data:`SUBSTREAM_LABELS`; this returns the ``numpy.random.Generator`` for
    the requested pipeline ``stage``. Two stages that pass the same master seed
    and global index but different ``stage`` labels draw from independent streams
    that are each reproducible from the global index alone.

    Parameters
    ----------
    master : numpy.random.SeedSequence
        Resolved master sequence (see :func:`as_seed_sequence`).
    global_index : int
        Global realization index in ``[0, N)``.
    stage : str
        Pipeline stage label; must be a member of :data:`SUBSTREAM_LABELS`
        (``'generation'`` or ``'disaggregation'``).

    Returns
    -------
    numpy.random.Generator
        Independent generator for ``(global_index, stage)``.

    Raises
    ------
    ValueError
        If ``stage`` is not a known stage label.
    """
    if stage not in SUBSTREAM_LABELS:
        raise ValueError(
            f"Unknown pipeline stage {stage!r}; expected one of {SUBSTREAM_LABELS}"
        )
    child = spawn_realization_seed(master, global_index)
    substreams = child.spawn(len(SUBSTREAM_LABELS))
    return np.random.default_rng(substreams[SUBSTREAM_LABELS.index(stage)])
