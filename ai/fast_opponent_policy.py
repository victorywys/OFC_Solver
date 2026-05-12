"""Cheap opponent policy used during rollouts in the fast analyzer.

Trades opponent move quality for speed. ~100x faster than `HeuristicPolicy`
because it skips per-action feature scoring entirely and decides by raw
card rank.

Strategy
--------
* Street 1 (5 cards in hand): 2 strongest go to bottom, next 2 to middle,
  the weakest to top. Roughly preserves Bottom >= Middle >= Top ordering
  on average, which keeps opponent foul rate from blowing up.
* Streets 2-5 (3-card pineapple): discard the weakest, then place the two
  remaining cards in the deepest free row first (bottom -> middle -> top).
* Fantasy hands: defer to the existing exact `FantasySolverPolicy`.
  Fantasy is rare in rollouts and matters too much to crippie with a
  rank-only heuristic.

Used only for the *non-acting* seat. The acting seat keeps using the
full heuristic, because that seat's decisions directly determine the EV
we are estimating.
"""

from __future__ import annotations

from typing import Sequence

from engine.cards import card_rank, is_joker
from engine.fantasy import FantasyTier
from state.action import Action
from state.board import (
    PlayerBoard,
    SLOT_BOTTOM,
    SLOT_DISCARD,
    SLOT_MIDDLE,
    SLOT_TOP,
)
from state.game_state import GameState

from .policy import Policy


def _card_strength(c: int) -> int:
    """Sort key (higher = stronger). Jokers rank above any natural card."""
    if is_joker(c):
        return 100
    return card_rank(c)


class FastOpponentPolicy(Policy):
    """Cheap rank-based completion policy.

    Stateless and deterministic for a given input. Cheaper than
    `HeuristicPolicy` by ~2 orders of magnitude because it avoids
    enumerating + scoring every legal action.
    """

    name = "fast_opponent"

    def __init__(self, fantasy_fallback: Policy | None = None) -> None:
        # Fantasy hands need a real solver — they're rare during rollouts
        # but worth a lot of points when they happen. Inject a fallback so
        # callers can share one FantasySolverPolicy instance across seats.
        self._fantasy_fallback = fantasy_fallback

    def act(self, gs: GameState, player: int) -> Action:
        hs = gs.hands[player]
        if hs.fantasy_tier != FantasyTier.NORMAL:
            if self._fantasy_fallback is None:
                raise RuntimeError(
                    "FastOpponentPolicy got a fantasy hand but no fallback"
                )
            return self._fantasy_fallback.act(gs, player)
        if gs.current_street == 1:
            return self._act_initial(hs.pending)
        return self._act_pineapple(hs.pending, hs.board)

    # ------------------------------------------------------------------
    @staticmethod
    def _act_initial(cards: Sequence[int]) -> Action:
        if len(cards) != 5:
            raise ValueError(f"street 1 expects 5 cards, got {len(cards)}")
        s = sorted(cards, key=_card_strength, reverse=True)  # strong → weak
        placements = (
            (s[0], SLOT_BOTTOM),
            (s[1], SLOT_BOTTOM),
            (s[2], SLOT_MIDDLE),
            (s[3], SLOT_MIDDLE),
            (s[4], SLOT_TOP),
        )
        return Action(placements)

    # ------------------------------------------------------------------
    @staticmethod
    def _act_pineapple(cards: Sequence[int], board: PlayerBoard) -> Action:
        if len(cards) != 3:
            raise ValueError(f"pineapple expects 3 cards, got {len(cards)}")
        s = sorted(cards, key=_card_strength)  # weak → strong
        discard = s[0]
        keep = (s[2], s[1])  # strongest first
        # free[slot] for slot in (TOP, MIDDLE, BOTTOM)
        free = [board.free_top(), board.free_middle(), board.free_bottom()]
        placements: list[tuple[int, int]] = []
        for c in keep:
            for slot in (SLOT_BOTTOM, SLOT_MIDDLE, SLOT_TOP):
                if free[slot] > 0:
                    placements.append((c, slot))
                    free[slot] -= 1
                    break
            else:
                # Shouldn't happen: there are always at least 2 free slots
                # before a pineapple street completes.
                raise RuntimeError("no free slot for opponent placement")
        placements.append((discard, SLOT_DISCARD))
        return Action(tuple(placements))


__all__ = ["FastOpponentPolicy"]
