"""Royalty tables.

User spec (default config):
    Top:
        - high card / pair below 6  : 1
        - pair 66+                  : 2
        - trips                     : 4
    Middle:
        - high_card .. two_pair     : 1
        - trips / straight          : 2
        - flush                     : 4
        - full_house                : 8
        - quads                     : 12
        - straight_flush / royal    : 20
    Bottom:
        - high_card .. trips        : 0
        - straight                  : 1
        - flush                     : 2
        - full_house                : 4
        - quads                     : 8
        - straight_flush            : 12
        - royal_flush               : 25

A `STANDARD_PINEAPPLE` preset is also exposed for comparison/research.

All numbers live in `RoyaltyConfig`. Code never hardcodes them; pass a config
into `royalty_top/middle/bottom` (or use the module-level default).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import RANK_2, RANK_6, RANK_A
from .evaluator import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    ROYAL_FLUSH,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    HandRank,
)


@dataclass(frozen=True)
class RoyaltyConfig:
    """Configurable royalty values. All keys present so lookups are total."""

    # top: pair royalties indexed by pair rank (rank 0..12 = 2..A); high-card too
    top_high_card: int = 1
    top_pair_by_rank: tuple[int, ...] = field(
        default_factory=lambda: tuple(
            # pairs 22..55 = 1, 66..AA = 2 (per spec). Index is pair rank.
            (1 if r < RANK_6 else 2) for r in range(13)
        )
    )
    # top trips royalties indexed by trips rank
    top_trips_by_rank: tuple[int, ...] = field(
        default_factory=lambda: tuple(4 for _ in range(13))
    )

    # middle: indexed by category id (0..9)
    middle_by_category: tuple[int, ...] = (
        1,   # HIGH_CARD
        1,   # PAIR
        1,   # TWO_PAIR
        2,   # TRIPS
        2,   # STRAIGHT
        4,   # FLUSH
        8,   # FULL_HOUSE
        12,  # QUADS
        20,  # STRAIGHT_FLUSH
        20,  # ROYAL_FLUSH
    )

    # bottom: indexed by category id
    bottom_by_category: tuple[int, ...] = (
        0,   # HIGH_CARD
        0,   # PAIR
        0,   # TWO_PAIR
        0,   # TRIPS
        1,   # STRAIGHT
        2,   # FLUSH
        4,   # FULL_HOUSE
        8,   # QUADS
        12,  # STRAIGHT_FLUSH
        25,  # ROYAL_FLUSH
    )


DEFAULT_ROYALTIES = RoyaltyConfig()


def _standard_pineapple() -> RoyaltyConfig:
    """Standard real-world Pineapple OFC royalty table (for research/comparison)."""
    # top pair royalties: 66=1, 77=2, 88=3, 99=4, TT=5, JJ=6, QQ=7, KK=8, AA=9
    pair_top = list(0 for _ in range(13))
    for r in range(RANK_6, RANK_A + 1):
        pair_top[r] = (r - RANK_6) + 1
    # top trips: 222=10 ... AAA=22
    trips_top = list((r - RANK_2) + 10 for r in range(13))
    return RoyaltyConfig(
        top_high_card=0,
        top_pair_by_rank=tuple(pair_top),
        top_trips_by_rank=tuple(trips_top),
        middle_by_category=(0, 0, 0, 2, 4, 8, 12, 20, 30, 50),
        bottom_by_category=(0, 0, 0, 0, 2, 4, 6, 10, 15, 25),
    )


STANDARD_PINEAPPLE = _standard_pineapple()


def royalty_top(rank: HandRank, cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> int:
    cat, kickers = rank
    if cat == HIGH_CARD:
        return cfg.top_high_card
    if cat == PAIR:
        return cfg.top_pair_by_rank[kickers[0]]
    if cat == TRIPS:
        return cfg.top_trips_by_rank[kickers[0]]
    # Top can't actually contain >TRIPS in valid play, but be defensive.
    return 0


def royalty_middle(rank: HandRank, cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> int:
    cat = rank[0]
    return cfg.middle_by_category[cat]


def royalty_bottom(rank: HandRank, cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> int:
    cat = rank[0]
    return cfg.bottom_by_category[cat]


__all__ = [
    "RoyaltyConfig",
    "DEFAULT_ROYALTIES",
    "STANDARD_PINEAPPLE",
    "royalty_top",
    "royalty_middle",
    "royalty_bottom",
]
