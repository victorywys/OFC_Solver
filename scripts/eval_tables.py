"""Evaluate a trained `TableAwarePolicy` head-to-head vs baselines.

Usage:
    python -m scripts.eval_tables --run artifacts/run_<TS>/ --n-games 500

Pits the table-aware policy against random / heuristic / mc baselines
using `simulation.evaluation.evaluate_matchup` (seat-symmetric, so seat
bias cancels). Prints chips/hand and win rates.
"""

from __future__ import annotations

import argparse
import functools
import time

from ai.heuristic_policy import HeuristicPolicy
from ai.monte_carlo_policy import MCConfig, MonteCarloPolicy
from simulation.evaluation import evaluate_matchup
from simulation.policy_factories import (
    heuristic_factory,
    mc_factory,
    random_factory,
)

from tables import TableAwareConfig, load_run_as_policy


# ---------------------------------------------------------------------------
# Picklable factory for `TableAwarePolicy`
# ---------------------------------------------------------------------------
def _build_table_aware(
    seed: int,
    *,
    run_dir: str,
    fallback: str,
    config: TableAwareConfig,
):
    """Build a fresh TableAwarePolicy. Picklable when called via partial.

    Loads tables from disk on every call. For a single-process evaluation
    that's fine; for multiprocessing we'd want a per-worker cache.
    """
    if fallback == "heuristic":
        fb = HeuristicPolicy(seed=seed)
    elif fallback == "mc":
        fb = MonteCarloPolicy(config=MCConfig(n_rollouts=8, top_k=6), seed=seed)
    else:
        raise ValueError(f"unknown fallback {fallback!r}")
    return load_run_as_policy(
        run_dir,
        fallback=fb,
        config=config,
        seed=seed,
    )


def make_table_aware_factory(
    run_dir: str,
    fallback: str,
    config: TableAwareConfig,
):
    return functools.partial(
        _build_table_aware,
        run_dir=run_dir,
        fallback=fallback,
        config=config,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--fallback", choices=["heuristic", "mc"], default="heuristic")
    parser.add_argument("--prior-min-visits", type=int, default=8)
    parser.add_argument("--opening-min-visits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = TableAwareConfig(
        prior_min_visits=args.prior_min_visits,
        opening_min_visits=args.opening_min_visits,
    )
    ta_factory = make_table_aware_factory(args.run, args.fallback, config)

    print(f"Evaluating table-aware policy from: {args.run}")
    print(f"  fallback           : {args.fallback}")
    print(f"  prior_min_visits   : {args.prior_min_visits}")
    print(f"  opening_min_visits : {args.opening_min_visits}")
    print(f"  n_games (per side) : {args.n_games}  (×2 for seat-sym = "
          f"{2 * args.n_games} total)")
    print()

    matchups = [
        ("table-aware", "random",    ta_factory, random_factory),
        ("table-aware", "heuristic", ta_factory, heuristic_factory),
        ("table-aware", "mc",        ta_factory, mc_factory),
        ("heuristic",   "random",    heuristic_factory, random_factory),
        ("mc",          "heuristic", mc_factory, heuristic_factory),
    ]

    print(f"{'A':14s}  {'B':12s}  {'A avg':>9s}  {'A wins':>7s}  "
          f"{'B wins':>7s}  {'ties':>5s}  {'sec':>6s}")
    print("-" * 78)
    for a_name, b_name, a_fac, b_fac in matchups:
        t0 = time.perf_counter()
        result = evaluate_matchup(
            a_fac, b_fac,
            n_games=args.n_games,
            seed=args.seed,
            seat_symmetric=True,
        )
        dt = time.perf_counter() - t0
        print(f"{a_name:14s}  {b_name:12s}  "
              f"{result.a_avg_score:>+9.3f}  "
              f"{result.a_wins:>7d}  {result.b_wins:>7d}  "
              f"{result.ties:>5d}  {dt:>6.1f}")


if __name__ == "__main__":
    main()
