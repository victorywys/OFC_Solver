"""Benchmark self-play throughput: sequential vs parallel.

Run:
    python -m benchmarks.bench_self_play
"""

from __future__ import annotations

import os
import time

from simulation.collectors import (
    FantasyTransitionCollector,
    FoulByTierCollector,
    MatchSummaryCollector,
    RoyaltyByRowCollector,
)
from simulation.policy_factories import (
    heuristic_factory,
    random_factory,
)
from simulation.self_play import SelfPlay


def _baseline_collectors():
    return [
        MatchSummaryCollector,
        FoulByTierCollector,
        RoyaltyByRowCollector,
        FantasyTransitionCollector,
    ]


def _bench(label: str, fn, *args, **kwargs) -> None:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    n = out["match_summary"].n_games
    print(
        f"{label:42s}  games={n:>5d}  {dt:>6.2f} s  {n/dt:>7.1f} games/s"
    )


def main() -> None:
    print("Self-play throughput benchmarks")
    print("-" * 80)

    # Random vs random — fastest baseline
    sp = SelfPlay(p0_factory=random_factory, p1_factory=random_factory)
    _bench(
        "random vs random  (sequential)",
        sp.run, 1000, _baseline_collectors(), seed=0,
    )

    # Heuristic vs random
    sp = SelfPlay(p0_factory=heuristic_factory, p1_factory=random_factory)
    _bench(
        "heuristic vs random  (sequential)",
        sp.run, 200, _baseline_collectors(), seed=0,
    )

    # Heuristic self-play
    sp = SelfPlay(p0_factory=heuristic_factory, p1_factory=heuristic_factory)
    _bench(
        "heuristic self-play  (sequential)",
        sp.run, 200, _baseline_collectors(), seed=0,
    )

    n_workers = max(1, (os.cpu_count() or 1) - 1)
    print()
    print(f"Parallel ({n_workers} workers):")

    sp = SelfPlay(p0_factory=random_factory, p1_factory=random_factory)
    _bench(
        f"random vs random  (parallel x{n_workers})",
        sp.run_parallel, 4000, _baseline_collectors(),
        seed=0, n_workers=n_workers,
    )

    sp = SelfPlay(p0_factory=heuristic_factory, p1_factory=heuristic_factory)
    _bench(
        f"heuristic self-play  (parallel x{n_workers})",
        sp.run_parallel, 800, _baseline_collectors(),
        seed=0, n_workers=n_workers,
    )

    # Print one summary as a sanity check
    out = sp.run(50, _baseline_collectors(), seed=0)
    summary = out["match_summary"]
    print()
    print("Heuristic self-play summary (50 games):")
    print(f"  avg total_a    = {summary.avg_total_a:+.3f}")
    print(f"  P0 foul rate   = {summary.p0_foul_rate:.1%}")
    print(f"  P1 foul rate   = {summary.p1_foul_rate:.1%}")
    print(f"  P0 fantasy ent = {summary.p0_fantasy_enters} / {summary.n_games}")
    print(f"  P1 fantasy ent = {summary.p1_fantasy_enters} / {summary.n_games}")


if __name__ == "__main__":
    main()
