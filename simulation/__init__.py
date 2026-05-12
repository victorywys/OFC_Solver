"""Self-play framework for OFC.

Provides the *framework* layer for large-scale data collection:

    * `simulation.trace`      — `Turn`, `GameRecord`, `simulate_one_game()`
    * `simulation.collectors` — `Collector` ABC + baseline collectors
    * `simulation.self_play`  — `SelfPlay` runner (sequential + parallel)
    * `simulation.storage`    — `save_collectors()` / `load_table()`
    * `simulation.evaluation` — head-to-head matchup harness

This module is intentionally framework-only. Phase 6 plugs in precomputed
tables (fantasy EV, foul probability, rollout priors, opening book,
fantasy arrangement cache, transposition table, ...) by adding new
`Collector` subclasses and table builders that consume `GameRecord`s or
direct sampling.
"""

from .trace import GameRecord, Turn, simulate_one_game
from .collectors import (
    Collector,
    FantasyTransitionCollector,
    FoulByTierCollector,
    MatchSummaryCollector,
    RoyaltyByRowCollector,
    TraceCollector,
)
from .self_play import (
    PolicyFactory,
    SelfPlay,
    play_game,
)
from .storage import save_collectors, load_table

__all__ = [
    "GameRecord",
    "Turn",
    "simulate_one_game",
    "Collector",
    "MatchSummaryCollector",
    "FoulByTierCollector",
    "RoyaltyByRowCollector",
    "FantasyTransitionCollector",
    "TraceCollector",
    "PolicyFactory",
    "SelfPlay",
    "play_game",
    "save_collectors",
    "load_table",
]
