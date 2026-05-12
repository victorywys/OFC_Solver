"""Uniform-random policy.

Useful as:
    - a sanity baseline,
    - a fast opponent for stress-testing the engine,
    - a control in evaluator vs heuristic comparisons.

Uses its own seeded `random.Random` so simulations are reproducible.
"""

from __future__ import annotations

import random

from engine.fantasy import FantasyTier
from state.action import (
    Action,
    enumerate_initial_actions,
    enumerate_pineapple_actions,
    iter_fantasy_actions,
)
from state.game_state import GameState

from .policy import Policy


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int | None = None, fantasy_budget: int = 256) -> None:
        self._rng = random.Random(seed)
        self.fantasy_budget = fantasy_budget

    def act(self, gs: GameState, player: int) -> Action:
        hs = gs.hands[player]
        cards = list(hs.pending)

        if hs.fantasy_tier != FantasyTier.NORMAL:
            # Fantasy: huge action space; sample uniformly from a reservoir.
            reservoir: list[Action] = []
            for i, a in enumerate(
                iter_fantasy_actions(cards, hs.board, budget=self.fantasy_budget)
            ):
                if i < self.fantasy_budget:
                    reservoir.append(a)
            return self._rng.choice(reservoir)

        if gs.current_street == 1:
            acts = enumerate_initial_actions(cards)
        else:
            acts = enumerate_pineapple_actions(cards, hs.board)
        return self._rng.choice(acts)


__all__ = ["RandomPolicy"]
