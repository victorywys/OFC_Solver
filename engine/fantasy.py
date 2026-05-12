"""Fantasyland mechanics.

Per the user spec:

    Entry conditions (top row at end of normal round):
        QQ  -> F14   (deal 14 cards next round)
        KK  -> F15
        AA  -> F16
        trips -> F17

    Fantasy type NEVER upgrades. Once entered as F14, future maintenance
    can only renew the SAME tier or drop to Normal.

    Maintain Fantasy:
        - top row still satisfies the original entry condition, OR
        - bottom row is quads or stronger.

So a player in F14 (entered with QQ) maintains F14 if the new arrangement has
top QQ+ pair OR bottom quads+. Note that even AA on top maintains as F14, not
upgrades to F16.
"""

from __future__ import annotations

from enum import IntEnum

from .cards import RANK_A, RANK_K, RANK_Q
from .evaluator import (
    HandRank,
    PAIR,
    QUADS,
    ROYAL_FLUSH,
    STRAIGHT_FLUSH,
    TRIPS,
)


class FantasyTier(IntEnum):
    NORMAL = 0
    F14 = 14
    F15 = 15
    F16 = 16
    F17 = 17

    @property
    def n_cards(self) -> int:
        """Number of cards dealt at the start of a fantasy round (or 5 for normal)."""
        return 5 if self == FantasyTier.NORMAL else int(self)


def fantasy_tier_from_top(top_rank: HandRank) -> FantasyTier:
    """Tier earned by entering Fantasy from a normal round's top row."""
    cat, kickers = top_rank
    if cat == TRIPS:
        return FantasyTier.F17
    if cat == PAIR:
        r = kickers[0]
        if r == RANK_A:
            return FantasyTier.F16
        if r == RANK_K:
            return FantasyTier.F15
        if r == RANK_Q:
            return FantasyTier.F14
    return FantasyTier.NORMAL


def _top_meets_entry(top_rank: HandRank, current_tier: FantasyTier) -> bool:
    """Does the top row meet the original entry condition for `current_tier`?"""
    if current_tier == FantasyTier.NORMAL:
        return False
    cat, kickers = top_rank
    if current_tier == FantasyTier.F17:
        return cat == TRIPS
    # F14/F15/F16 all entered via a pair on top.
    if cat != PAIR:
        return False
    r = kickers[0]
    if current_tier == FantasyTier.F14:
        return r >= RANK_Q
    if current_tier == FantasyTier.F15:
        return r >= RANK_K
    if current_tier == FantasyTier.F16:
        return r == RANK_A
    return False


def maintains_fantasy(
    current_tier: FantasyTier,
    top_rank: HandRank,
    bottom_rank: HandRank,
) -> bool:
    """Does the new placement keep the player in their current fantasy tier?

    Maintenance rules per spec:
        - top row still satisfies the original entry condition, OR
        - bottom row is quads or stronger
    """
    if current_tier == FantasyTier.NORMAL:
        return False
    if _top_meets_entry(top_rank, current_tier):
        return True
    return bottom_rank[0] >= QUADS


def next_fantasy_tier(
    current_tier: FantasyTier,
    top_rank: HandRank,
    bottom_rank: HandRank,
) -> FantasyTier:
    """Compute the player's next-round fantasy tier after a (non-foul) showdown.

    From Normal: tier earned via top row entry condition.
    From F*: maintain same tier (per maintenance rules) or drop to Normal.
        IMPORTANT: never upgrade.
    """
    if current_tier == FantasyTier.NORMAL:
        return fantasy_tier_from_top(top_rank)
    if maintains_fantasy(current_tier, top_rank, bottom_rank):
        return current_tier
    return FantasyTier.NORMAL


__all__ = [
    "FantasyTier",
    "fantasy_tier_from_top",
    "maintains_fantasy",
    "next_fantasy_tier",
]
