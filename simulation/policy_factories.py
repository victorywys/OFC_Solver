"""Picklable policy factories for self-play.

The parallel `SelfPlay.run_parallel` requires policy factories to be
picklable. This module exposes module-level factory functions and a
`make_factory` helper that returns picklable callables for the standard
policies (`RandomPolicy`, `HeuristicPolicy`, `MonteCarloPolicy`).

Factories take a single `seed: int` argument (the per-player game seed)
and return a fresh `Policy` instance.

Usage
-----
    from simulation.policy_factories import (
        random_factory, heuristic_factory, mc_factory,
    )

    sp = SelfPlay(p0_factory=heuristic_factory, p1_factory=random_factory)
    sp.run_parallel(n_games=1000, ...)

For non-default policy configurations, call `make_factory(...)` which
returns a `functools.partial` that is picklable provided its arguments
are picklable (e.g. dataclasses, primitives).
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from ai.heuristic_policy import HeuristicPolicy
from ai.monte_carlo_policy import MCConfig, MonteCarloPolicy
from ai.policy import Policy
from ai.random_policy import RandomPolicy


def random_factory(seed: int) -> Policy:
    return RandomPolicy(seed=seed)


def heuristic_factory(seed: int) -> Policy:
    return HeuristicPolicy(seed=seed)


def mc_factory(seed: int) -> Policy:
    """Default MC config: small n_rollouts and top_k for fast self-play."""
    return MonteCarloPolicy(config=MCConfig(n_rollouts=8, top_k=6), seed=seed)


# ---------------------------------------------------------------------------
# Generic helper
# ---------------------------------------------------------------------------
def _build_policy(seed: int, *, klass: type, kwargs: dict[str, Any]) -> Policy:
    return klass(seed=seed, **kwargs)


def make_factory(klass: type, **kwargs: Any) -> Callable[[int], Policy]:
    """Create a picklable factory for `klass(seed=..., **kwargs)`.

    The kwargs dict and `klass` must themselves be picklable (e.g.
    frozen dataclass configs, primitives, etc.).
    """
    return functools.partial(_build_policy, klass=klass, kwargs=kwargs)


__all__ = [
    "random_factory",
    "heuristic_factory",
    "mc_factory",
    "make_factory",
]
