"""Benchmark the fantasy solver across all tiers. Run as:

    python -m benchmarks.bench_fantasy
"""

from __future__ import annotations

import random
import time

from engine.fantasy import FantasyTier
from fantasy.fantasy_solver import default_config_for, solve_fantasy
from fantasy.fantasy_search import FantasyConfig


def random_cards(rng: random.Random, n: int, include_jokers: bool = True) -> list[int]:
    pool = list(range(54)) if include_jokers else list(range(52))
    return rng.sample(pool, n)


def bench_tier(tier: FantasyTier, n_trials: int, exact: bool = False) -> None:
    rng = random.Random(0)
    n_cards = tier.n_cards
    cfg = FantasyConfig(exact=exact) if exact else default_config_for(tier)
    label = f"{tier.name} ({'exact' if exact else 'beam'})"

    total_t = 0.0
    total_leaves = 0
    total_pruned = 0
    evs = []

    for _ in range(n_trials):
        cards = random_cards(rng, n_cards)
        t0 = time.perf_counter()
        result = solve_fantasy(cards, tier, cfg)
        total_t += time.perf_counter() - t0
        total_leaves += result.stats.leaves_evaluated
        total_pruned += result.stats.pruned_by_bound
        evs.append(result.ev)

    avg_t = total_t / n_trials * 1000
    avg_leaves = total_leaves / n_trials
    avg_pruned = total_pruned / n_trials
    avg_ev = sum(evs) / len(evs)
    print(
        f"{label:20s}  trials={n_trials:>3d}  "
        f"avg={avg_t:8.1f} ms  "
        f"leaves={avg_leaves:>10,.0f}  pruned={avg_pruned:>6,.0f}  "
        f"avg EV={avg_ev:.2f}"
    )


def main() -> None:
    print("Fantasy solver benchmarks")
    print("-" * 80)
    bench_tier(FantasyTier.F14, n_trials=20, exact=True)
    bench_tier(FantasyTier.F14, n_trials=20, exact=False)
    bench_tier(FantasyTier.F15, n_trials=15, exact=False)
    bench_tier(FantasyTier.F16, n_trials=10, exact=False)
    bench_tier(FantasyTier.F17, n_trials=5, exact=False)


if __name__ == "__main__":
    main()
