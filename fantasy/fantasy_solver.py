"""Public entry point for the fantasy solver.

Exposes:
    - `solve_fantasy(cards, tier, config)` — the search function
    - `FantasyConfig`, `FantasyResult`, `SearchStats` — types
    - `default_config_for(tier)` — pre-tuned configs per tier
    - `FantasySolverPolicy` — wraps the solver as a `Policy` so any AI
      component can use it on fantasy hands transparently.

Default beam tuning (interactive latency targets):
    F14: exact mode (~1M leaves)
    F15: beam_bottom=400  beam_middle=120
    F16: beam_bottom=200  beam_middle=80
    F17: beam_bottom=120  beam_middle=60
"""

from __future__ import annotations

from engine.fantasy import FantasyTier
from state.action import Action
from state.board import SLOT_BOTTOM, SLOT_DISCARD, SLOT_MIDDLE, SLOT_TOP
from state.game_state import GameState

from ai.policy import Policy

from .fantasy_search import (
    FantasyConfig,
    FantasyResult,
    SearchStats,
    solve as _solve,
)


def default_config_for(tier: FantasyTier) -> FantasyConfig:
    """Reasonable defaults that target sub-second solves on F14-F17."""
    if tier == FantasyTier.F14:
        return FantasyConfig(exact=True)
    if tier == FantasyTier.F15:
        return FantasyConfig(bottom_beam=400, middle_beam=120)
    if tier == FantasyTier.F16:
        return FantasyConfig(bottom_beam=200, middle_beam=80)
    if tier == FantasyTier.F17:
        return FantasyConfig(bottom_beam=120, middle_beam=60)
    return FantasyConfig()


def solve_fantasy(
    cards,
    tier: FantasyTier,
    config: FantasyConfig | None = None,
) -> FantasyResult:
    """Solve a fantasy layout for `cards` under `tier`.

    If `config` is None, uses `default_config_for(tier)`.
    Returns the best found `FantasyResult`. Never returns a fouled layout.
    """
    if config is None:
        config = default_config_for(tier)
    return _solve(list(cards), tier, config)


# ---------------------------------------------------------------------------
# Policy integration: drop-in replacement for fantasy decisions
# ---------------------------------------------------------------------------
def fantasy_result_to_action(result: FantasyResult) -> Action:
    """Convert a `FantasyResult` into the engine's `Action` format."""
    placements: list[tuple[int, int]] = []
    for c in result.top:
        placements.append((c, SLOT_TOP))
    for c in result.middle:
        placements.append((c, SLOT_MIDDLE))
    for c in result.bottom:
        placements.append((c, SLOT_BOTTOM))
    for c in result.discards:
        placements.append((c, SLOT_DISCARD))
    return Action(tuple(placements))


class FantasySolverPolicy(Policy):
    """A `Policy` that uses the fantasy solver when in a fantasy tier and
    delegates to a fallback policy on normal streets.

    Composability: any `Policy` (heuristic, rollout, MCTS, ...) can be
    passed as `fallback`, so this works transparently inside larger AIs.
    """

    name = "fantasy_solver"

    def __init__(
        self,
        fallback: Policy,
        config_by_tier=None,
    ) -> None:
        self.fallback = fallback
        # Allow override of per-tier config maps
        self.config_by_tier = config_by_tier or {
            FantasyTier.F14: default_config_for(FantasyTier.F14),
            FantasyTier.F15: default_config_for(FantasyTier.F15),
            FantasyTier.F16: default_config_for(FantasyTier.F16),
            FantasyTier.F17: default_config_for(FantasyTier.F17),
        }

    def act(self, gs: GameState, player: int) -> Action:
        hs = gs.hands[player]
        if hs.fantasy_tier == FantasyTier.NORMAL:
            return self.fallback.act(gs, player)
        cfg = self.config_by_tier.get(hs.fantasy_tier)
        result = solve_fantasy(hs.pending, hs.fantasy_tier, cfg)
        return fantasy_result_to_action(result)


__all__ = [
    "FantasyConfig",
    "FantasyResult",
    "SearchStats",
    "default_config_for",
    "solve_fantasy",
    "fantasy_result_to_action",
    "FantasySolverPolicy",
]
