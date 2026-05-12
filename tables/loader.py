"""Convenience loader for a saved table-build run.

After running `python -m scripts.build_all_tables --out artifacts/run_X`,
use:

    from tables.loader import load_run_as_policy
    pol = load_run_as_policy("artifacts/run_X")
    action = pol.act(gs, player=0)

This is just a thin wrapper around `simulation.storage.load_run` that
assembles a ready-to-use `TableAwarePolicy`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ai.heuristic_policy import HeuristicPolicy
from ai.policy import Policy
from simulation.storage import load_run

from .canonical_opening import CanonicalOpeningBookTable
from .fantasy_cache import FantasyArrangementCache
from .opening_book import OpeningBookTable
from .policy_prior import PolicyPriorTable
from .table_aware_policy import TableAwareConfig, TableAwarePolicy
from .transposition import TranspositionTable


def load_tables(run_dir: Path | str) -> dict:
    """Return the dict produced by `simulation.storage.load_run`.

    Keys are the collector names: `foul_prob`, `policy_prior`,
    `opening_book`, `fantasy_ev`, `fantasy_arrangement`, plus baselines
    (`match_summary`, `foul_by_tier`, ...) and `metadata`.
    """
    return load_run(run_dir)


def load_run_as_policy(
    run_dir: Path | str,
    *,
    fallback: Optional[Policy] = None,
    config: Optional[TableAwareConfig] = None,
    transposition_max_entries: Optional[int] = 200_000,
    seed: int = 0,
) -> TableAwarePolicy:
    """Assemble a `TableAwarePolicy` from a saved run.

    Parameters
    ----------
    run_dir
        Directory written by `scripts/build_all_tables.py`.
    fallback
        Policy used when no table hit. Defaults to a fresh `HeuristicPolicy`.
        Use a `MonteCarloPolicy` here for stronger play (at the cost of speed).
    config
        Optional `TableAwareConfig` overrides (e.g. `prior_min_visits`).
    transposition_max_entries
        Cap for the runtime transposition cache. None = unbounded.
    seed
        Seed for the default fallback `HeuristicPolicy` (ignored if you
        pass your own `fallback`).
    """
    blobs = load_run(run_dir)

    # Prefer the fully-precomputed canonical book (152,646 entries,
    # 100% street-1 coverage, ~1us lookups). Fall back to the legacy
    # sampled OpeningBookTable if only that is present. Both classes
    # are duck-typed against the same `.lookup(hand, min_visits=...)`
    # API used by TableAwarePolicy.
    opening_book = blobs.get("opening_book_canonical")
    if not isinstance(opening_book, CanonicalOpeningBookTable):
        opening_book = blobs.get("opening_book")
        if not isinstance(opening_book, OpeningBookTable):
            opening_book = None

    policy_prior = blobs.get("policy_prior")
    if not isinstance(policy_prior, PolicyPriorTable):
        policy_prior = None

    fantasy_cache = blobs.get("fantasy_arrangement")
    if not isinstance(fantasy_cache, FantasyArrangementCache):
        fantasy_cache = None

    if fallback is None:
        fallback = HeuristicPolicy(seed=seed)
    if config is None:
        config = TableAwareConfig()

    return TableAwarePolicy(
        fallback=fallback,
        config=config,
        transposition=TranspositionTable(max_entries=transposition_max_entries),
        opening_book=opening_book,
        fantasy_cache=fantasy_cache,
        policy_prior=policy_prior,
    )


__all__ = ["load_tables", "load_run_as_policy"]
