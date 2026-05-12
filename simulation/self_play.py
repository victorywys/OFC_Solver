"""Self-play runner.

Drives many games through a set of `Collector`s. Supports sequential and
multiprocessing-parallel execution.

Reproducibility
---------------
The runner accepts a base `seed` and derives a per-game seed deterministically:

    game_seed = seed + game_index   # default
    or:
    game_seed = explicit_seed_list[game_index]

Each game's seed drives:
    - the deck shuffle (passed to `GameState.new(seed=...)`)
    - the policy factories (called with the same seed for both players,
      offset by 0 / 1 for P0 / P1 to avoid trivial correlation)

This means: identical (seed, factory_p0, factory_p1, n_games, collector_classes)
gives identical collector results — even across sequential vs parallel runs,
provided the parallel chunking is order-stable (we use chunk-and-merge).

Policy factories
----------------
A `PolicyFactory` is `Callable[[int], Policy]`. It must be picklable for
parallel execution (no lambdas; module-level functions or `functools.partial`
of module-level functions). The seed is the only argument.

Initial fantasy tiers
---------------------
By default both players start in NORMAL. Pass `initial_tier_factory` (a
callable `(game_seed, player) -> FantasyTier`) for richer setups
(e.g. force F14 every Nth game for fantasy-mode benchmarking).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from engine.fantasy import FantasyTier
from engine.royalties import DEFAULT_ROYALTIES, RoyaltyConfig
from state.game_state import GameState

from ai.policy import Policy

from .collectors import Collector
from .trace import GameRecord, simulate_one_game


PolicyFactory = Callable[[int], Policy]
TierFactory = Callable[[int, int], FantasyTier]
CollectorFactory = Callable[[], Collector]


def _default_tier_factory(_seed: int, _player: int) -> FantasyTier:
    return FantasyTier.NORMAL


# ---------------------------------------------------------------------------
# Single-game driver
# ---------------------------------------------------------------------------
def play_game(
    seed: int,
    p0_factory: PolicyFactory,
    p1_factory: PolicyFactory,
    *,
    initial_tier_factory: TierFactory = _default_tier_factory,
    royalty_cfg: RoyaltyConfig = DEFAULT_ROYALTIES,
    record_turns: bool = False,
) -> GameRecord:
    """Play one game with deterministic seed and return a `GameRecord`."""
    p0 = p0_factory(seed * 2)
    p1 = p1_factory(seed * 2 + 1)
    gs = GameState.new(
        seed=seed,
        royalty_cfg=royalty_cfg,
        fantasy_p0=initial_tier_factory(seed, 0),
        fantasy_p1=initial_tier_factory(seed, 1),
    )
    return simulate_one_game(
        gs,
        p0,
        p1,
        record_turns=record_turns,
        seed_for_record=seed,
    )


# ---------------------------------------------------------------------------
# Worker function for multiprocessing (must be top-level for pickling)
# ---------------------------------------------------------------------------
def _worker_run_chunk(args: tuple) -> list[bytes]:
    """Run a chunk of games and return pickled collectors.

    We pickle here (rather than returning live collectors) because some
    collector implementations may include unpicklable child state via
    `result()` outputs. Pickling at chunk-end keeps the worker boundary
    clean and ensures the merge step deals with serializable bytes.
    """
    import pickle

    (
        seeds,
        p0_factory,
        p1_factory,
        initial_tier_factory,
        royalty_cfg,
        collector_factories,
        record_turns,
    ) = args
    collectors = [f() for f in collector_factories]
    for s in seeds:
        rec = play_game(
            s,
            p0_factory,
            p1_factory,
            initial_tier_factory=initial_tier_factory,
            royalty_cfg=royalty_cfg,
            record_turns=record_turns,
        )
        for c in collectors:
            c.observe(rec)
    return [pickle.dumps(c, protocol=4) for c in collectors]


# ---------------------------------------------------------------------------
# SelfPlay driver
# ---------------------------------------------------------------------------
@dataclass
class SelfPlay:
    """Self-play runner. Use `run` (sequential) or `run_parallel` (mp).

    Both methods take a list of `CollectorFactory` callables (zero-arg) so
    that workers can construct fresh collector instances. The result is a
    dict mapping `collector.name` -> `collector.result()`, plus the merged
    collector list under `'_collectors'` for further inspection.
    """

    p0_factory: PolicyFactory
    p1_factory: PolicyFactory
    initial_tier_factory: TierFactory = _default_tier_factory
    royalty_cfg: RoyaltyConfig = DEFAULT_ROYALTIES

    # ------- sequential -------
    def run(
        self,
        n_games: int,
        collector_factories: Sequence[CollectorFactory],
        *,
        seed: int = 0,
        seeds: Optional[Sequence[int]] = None,
        progress_every: Optional[int] = None,
    ) -> dict:
        """Run `n_games` sequentially. Returns dict of collector results."""
        collectors = [f() for f in collector_factories]
        record_turns = any(c.needs_full_trace for c in collectors)

        if seeds is None:
            seeds = range(seed, seed + n_games)
        else:
            if len(seeds) != n_games:
                raise ValueError(
                    f"seeds length {len(seeds)} != n_games {n_games}"
                )

        for i, s in enumerate(seeds):
            rec = play_game(
                s,
                self.p0_factory,
                self.p1_factory,
                initial_tier_factory=self.initial_tier_factory,
                royalty_cfg=self.royalty_cfg,
                record_turns=record_turns,
            )
            for c in collectors:
                c.observe(rec)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  [self-play] {i + 1}/{n_games} games")

        return self._build_results(collectors)

    # ------- parallel (multiprocessing) -------
    def run_parallel(
        self,
        n_games: int,
        collector_factories: Sequence[CollectorFactory],
        *,
        seed: int = 0,
        n_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> dict:
        """Run `n_games` across `n_workers` processes.

        Each worker constructs its own collectors, plays its share of
        games, and pickles the result. The main process unpickles and
        merges them. Result is identical to `run()` (since `Collector.merge`
        is required to be commutative and associative).
        """
        import pickle

        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 1) - 1)
        if chunk_size is None:
            # ~16 chunks per worker for decent load balancing without
            # excessive pickling overhead.
            chunk_size = max(1, n_games // (n_workers * 16))

        # Need at least one collector to know if we should record turns.
        proto_collectors = [f() for f in collector_factories]
        record_turns = any(c.needs_full_trace for c in proto_collectors)

        # Build chunked seed lists.
        seeds = list(range(seed, seed + n_games))
        chunks = [seeds[i : i + chunk_size] for i in range(0, n_games, chunk_size)]

        worker_args = [
            (
                ch,
                self.p0_factory,
                self.p1_factory,
                self.initial_tier_factory,
                self.royalty_cfg,
                tuple(collector_factories),
                record_turns,
            )
            for ch in chunks
        ]

        # Sort chunk results by chunk-index after collection so that the
        # merge order is deterministic. Pool.imap preserves order, so this
        # is automatic — but we also accept Pool.imap_unordered for speed
        # by sorting results explicitly. Use imap (ordered) for determinism.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            chunk_results = list(pool.imap(_worker_run_chunk, worker_args))

        # Initialize a clean set of merged collectors.
        merged = [f() for f in collector_factories]
        for chunk_pickled in chunk_results:
            for slot, blob in enumerate(chunk_pickled):
                worker_collector = pickle.loads(blob)
                merged[slot].merge(worker_collector)

        return self._build_results(merged)

    # ------- output formatting -------
    @staticmethod
    def _build_results(collectors: list[Collector]) -> dict:
        out: dict = {"_collectors": collectors}
        for c in collectors:
            out[c.name] = c.result()
        return out


__all__ = [
    "PolicyFactory",
    "TierFactory",
    "CollectorFactory",
    "SelfPlay",
    "play_game",
]
