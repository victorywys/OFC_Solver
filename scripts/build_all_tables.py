"""Unified table-build campaign.

Runs ONE big parallel self-play that simultaneously builds every Phase-6
table. All collectors observe the same stream of game records, so:

    * `match_summary`        — matchup metrics
    * `foul_by_tier`         — foul rate per tier
    * `royalty_by_row`       — royalty distribution by row
    * `fantasy_transitions`  — tier transition counts
    * `foul_prob`            — P(foul | state_signature)
    * `policy_prior`         — Welford(score | state, action)
    * `opening_book`         — best action per street-1 hand
    * `fantasy_ev`           — per-tier EV stats + continuation bonuses
    * `fantasy_arrangement`  — solver-output cache for fantasy hands

… all populate from a single run, with parallel workers, and are saved
under `artifacts/<run_name>/<table>.pkl`.

Usage
-----
    python -m scripts.build_all_tables --n-games 5000 --p0 mc --p1 heuristic

    python -m scripts.build_all_tables \\
        --n-games 50000 --n-workers 16 --seed 0 \\
        --p0 mc --p1 mc \\
        --fantasy-rate 0.20 \\
        --out artifacts/run_big/

Fantasy injection
-----------------
By default 0% of games start in a fantasy tier (because we self-play
from cold-start). Pass `--fantasy-rate R` to force a fraction R of games
to start P0 (and a separate R of games to start P1) in F14-F17 with
roughly the empirical entry-tier distribution. This gives the
`FantasyEVCollector` and `FantasyArrangementCacheCollector` data without
requiring a chicken-and-egg pre-existing strong policy that earns
fantasies on its own.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

from engine.fantasy import FantasyTier
from simulation.collectors import (
    FantasyTransitionCollector,
    FoulByTierCollector,
    MatchSummaryCollector,
    RoyaltyByRowCollector,
)
from simulation.policy_factories import (
    heuristic_factory,
    mc_factory,
    random_factory,
)
from simulation.self_play import SelfPlay
from simulation.storage import save_collectors

from tables import (
    FantasyArrangementCacheCollector,
    FantasyEVCollector,
    FoulProbCollector,
    OpeningBookCollector,
    PolicyPriorCollector,
)


# ---------------------------------------------------------------------------
# Policy choices
# ---------------------------------------------------------------------------
POLICY_FACTORIES = {
    "random": random_factory,
    "heuristic": heuristic_factory,
    "mc": mc_factory,
}


# ---------------------------------------------------------------------------
# Collector roster
# ---------------------------------------------------------------------------
def all_collector_factories() -> list:
    """All collector classes (used as zero-arg factories) for the campaign."""
    return [
        MatchSummaryCollector,
        FoulByTierCollector,
        RoyaltyByRowCollector,
        FantasyTransitionCollector,
        FoulProbCollector,
        PolicyPriorCollector,
        OpeningBookCollector,
        FantasyEVCollector,
        FantasyArrangementCacheCollector,
    ]


# ---------------------------------------------------------------------------
# Fantasy-injection tier factory
# ---------------------------------------------------------------------------
# Module-level so it can be pickled for multiprocessing.
_FANTASY_TIERS = (FantasyTier.F14, FantasyTier.F15, FantasyTier.F16, FantasyTier.F17)
# Empirical bias (rough): F14 most common, then F15, F16, F17 rare.
_FANTASY_WEIGHTS = (0.55, 0.25, 0.12, 0.08)


def _pick_fantasy_tier(rng_int: int) -> FantasyTier:
    """Deterministic pick of fantasy tier from a seed-derived int."""
    # Linear search over cumulative weights — only 4 categories.
    u = ((rng_int * 2654435761) % (1 << 32)) / float(1 << 32)
    acc = 0.0
    for tier, w in zip(_FANTASY_TIERS, _FANTASY_WEIGHTS):
        acc += w
        if u < acc:
            return tier
    return _FANTASY_TIERS[-1]


class FantasyInjectionFactory:
    """Picklable tier factory: each player gets fantasy with prob `rate`.

    Deterministic: depends only on (seed, player) and the configured rate.
    """

    __slots__ = ("rate",)

    def __init__(self, rate: float) -> None:
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"fantasy rate must be in [0, 1], got {rate}")
        self.rate = rate

    def __call__(self, seed: int, player: int) -> FantasyTier:
        if self.rate <= 0.0:
            return FantasyTier.NORMAL
        # Hash seed-player to a u32, use first half for the bernoulli
        # decision and the second half for the tier choice.
        h = (seed * 1_000_003 + player) & 0xFFFFFFFFFFFFFFFF
        u = ((h * 2654435761) % (1 << 32)) / float(1 << 32)
        if u >= self.rate:
            return FantasyTier.NORMAL
        return _pick_fantasy_tier(h >> 32 if h >= (1 << 32) else h ^ 0xDEADBEEF)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Phase-6 table builder.")
    parser.add_argument("--n-games", type=int, default=2_000)
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Default: cpu_count - 1.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p0", choices=POLICY_FACTORIES.keys(), default="heuristic")
    parser.add_argument("--p1", choices=POLICY_FACTORIES.keys(), default="heuristic")
    parser.add_argument("--fantasy-rate", type=float, default=0.10,
                        help="Fraction of games where each player starts in fantasy (0-1).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir. Default: artifacts/run_<timestamp>/")
    parser.add_argument("--sequential", action="store_true",
                        help="Run on a single process (debugging).")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path("artifacts") / (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out.mkdir(parents=True, exist_ok=True)

    p0_factory = POLICY_FACTORIES[args.p0]
    p1_factory = POLICY_FACTORIES[args.p1]
    tier_factory = FantasyInjectionFactory(rate=args.fantasy_rate)

    sp = SelfPlay(
        p0_factory=p0_factory,
        p1_factory=p1_factory,
        initial_tier_factory=tier_factory,
    )

    collector_factories = all_collector_factories()

    # Banner
    print("=" * 78)
    print(f"Phase-6 table-build campaign")
    print(f"  n_games        : {args.n_games}")
    print(f"  policies       : P0={args.p0}, P1={args.p1}")
    print(f"  fantasy rate   : {args.fantasy_rate:.2f}")
    print(f"  seed           : {args.seed}")
    print(f"  workers        : {args.n_workers or os.cpu_count()}")
    print(f"  collectors     : {[c.__name__ for c in collector_factories]}")
    print(f"  output         : {out}")
    print("=" * 78)

    t0 = time.perf_counter()
    if args.sequential:
        results = sp.run(
            n_games=args.n_games,
            collector_factories=collector_factories,
            seed=args.seed,
            progress_every=max(1, args.n_games // 20),
        )
    else:
        results = sp.run_parallel(
            n_games=args.n_games,
            collector_factories=collector_factories,
            seed=args.seed,
            n_workers=args.n_workers,
        )
    dt = time.perf_counter() - t0
    print(f"Self-play finished in {dt:.1f}s "
          f"({args.n_games / dt:.1f} games/s)")

    # Save artifacts
    save_collectors(
        results["_collectors"],
        out,
        metadata={
            "n_games": args.n_games,
            "seed": args.seed,
            "p0_policy": args.p0,
            "p1_policy": args.p1,
            "fantasy_rate": args.fantasy_rate,
            "n_workers": args.n_workers or os.cpu_count(),
            "duration_s": dt,
            "timestamp": datetime.now().isoformat(),
        },
    )
    print(f"Saved {len(results['_collectors'])} tables to {out}")

    # Print quick summary of every result.
    print("\nResults:")
    print("-" * 78)
    for c in results["_collectors"]:
        r = results[c.name]
        try:
            n = len(r)
        except TypeError:
            n = "n/a"
        print(f"  {c.name:24s}  type={type(r).__name__:30s}  size={n}")

    # Fantasy-EV calibration preview
    fev = results.get("fantasy_ev")
    if fev is not None:
        bonuses = fev.continue_bonuses()
        print("\nDerived fantasy continue-bonuses (per hand):")
        for tier_int, b in sorted(bonuses.items()):
            print(f"  {FantasyTier(tier_int).name:6s}  {b:+.2f}")


if __name__ == "__main__":
    main()
