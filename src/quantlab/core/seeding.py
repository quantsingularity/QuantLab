"""Deterministic seeding.

The PoC's current agents (a rank-based momentum signal, no model fitting)
don't actually consume any randomness, so seeding is a no-op for them today
-- the strategy is already 100% deterministic given the same price data.
It is applied anyway, globally, at the start of every run so that:

  (a) the "Seed: N" line in the generated report is an honest, verifiable
      claim rather than a hardcoded string nobody actually wires up, and
  (b) the v0.5 agents that *will* need it (e.g. an XGBoost-fitted signal --
      see `docs/04_Technology_Stack.md`) inherit reproducibility for free
      instead of requiring a second pass to bolt it on later.
"""

from __future__ import annotations

import random

import numpy as np

DEFAULT_SEED = 42


def apply_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed every stdlib/numpy RNG the pipeline currently touches or will touch."""
    random.seed(seed)
    np.random.seed(seed)
