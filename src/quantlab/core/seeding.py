"""Deterministic seeding.

apply_seed seeds Python's stdlib random module and NumPy's legacy global
RNG once, at the start of every run, so that any code that still reads
from the global RNG is reproducible. The pipeline's own randomness, the
synthetic price generator in quantlab.data.loaders and the ridge or
xgboost model fitting in quantlab.strategies.ml_signal, does not rely on
that global state: each takes the run's seed explicitly and constructs its
own numpy.random.Generator or passes random_state directly. That is the
more robust pattern for reproducibility, since it cannot be perturbed by
unrelated code elsewhere in the process that also touches the global RNG.
Calling apply_seed remains useful as a defensive baseline and keeps the
"Seed: N" line in the generated report an honest, verifiable claim.
"""

from __future__ import annotations

import random

import numpy as np

DEFAULT_SEED = 42


def apply_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed the stdlib and NumPy global RNGs."""
    random.seed(seed)
    np.random.seed(seed)
