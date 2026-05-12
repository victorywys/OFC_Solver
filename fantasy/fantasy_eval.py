"""Cached evaluator + royalty + continuation utilities for the fantasy solver.

The existing `engine.evaluator` already memoizes the no-joker 5-card path on
sorted tuples. Here we add:
    - mask-keyed wrappers so the search loop can pass card masks directly
      (avoids re-decoding card ids per call),
    - cheap "max possible royalty over a card pool" upper bounds for
      branch-and-bound pruning,
    - exact tier-aware continuation checks.

All public functions here are pure and deterministic.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from engine.cards import RANK_A, RANK_K, RANK_Q
from engine.evaluator import (
    HandRank,
    PAIR,
    QUADS,
    TRIPS,
    evaluate_3,
    evaluate_5,
)
from engine.fantasy import FantasyTier
from engine.royalties import (
    DEFAULT_ROYALTIES,
    RoyaltyConfig,
    royalty_bottom,
    royalty_middle,
    royalty_top,
)

from .cards_mask import cards_of


# ---------------------------------------------------------------------------
# Mask-keyed evaluators (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1 << 18)
def eval_5_by_mask(mask: int) -> HandRank:
    cards = cards_of(mask)
    return evaluate_5(cards)


@lru_cache(maxsize=1 << 16)
def eval_3_by_mask(mask: int) -> HandRank:
    cards = cards_of(mask)
    return evaluate_3(cards)


# ---------------------------------------------------------------------------
# Continuation
# ---------------------------------------------------------------------------
def is_continuation(
    tier: FantasyTier,
    top_rank: HandRank,
    bottom_rank: HandRank,
) -> bool:
    """Does this final layout maintain `tier` for the next round?

    Per spec:
        F14 (entered QQ): top >= QQ pair OR bottom quads+
        F15 (entered KK): top >= KK pair OR bottom quads+
        F16 (entered AA): top == AA pair OR bottom quads+
        F17 (entered trips): top trips OR bottom quads+

    `tier` is the *original* entry tier; tier never upgrades.
    """
    if tier == FantasyTier.NORMAL:
        return False
    # bottom quads+ always maintains
    if bottom_rank[0] >= QUADS:
        return True
    cat, kickers = top_rank
    if tier == FantasyTier.F17:
        return cat == TRIPS
    if cat != PAIR:
        return False
    r = kickers[0]
    if tier == FantasyTier.F14:
        return r >= RANK_Q
    if tier == FantasyTier.F15:
        return r >= RANK_K
    if tier == FantasyTier.F16:
        return r == RANK_A
    return False


def bottom_guarantees_continuation(bottom_rank: HandRank) -> bool:
    """Bottom alone guarantees continuation iff it is quads or stronger."""
    return bottom_rank[0] >= QUADS


def top_guarantees_continuation(tier: FantasyTier, top_rank: HandRank) -> bool:
    """Top alone (regardless of bottom) suffices for `tier`'s entry condition."""
    cat, kickers = top_rank
    if tier == FantasyTier.F17:
        return cat == TRIPS
    if cat != PAIR:
        return False
    r = kickers[0]
    if tier == FantasyTier.F14:
        return r >= RANK_Q
    if tier == FantasyTier.F15:
        return r >= RANK_K
    if tier == FantasyTier.F16:
        return r == RANK_A
    return False


# ---------------------------------------------------------------------------
# Royalty wrappers
# ---------------------------------------------------------------------------
def bottom_royalty(rank: HandRank, cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> int:
    return royalty_bottom(rank, cfg)


def middle_royalty(rank: HandRank, cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> int:
    return royalty_middle(rank, cfg)


def top_royalty_value(rank: HandRank, cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> int:
    return royalty_top(rank, cfg)


# ---------------------------------------------------------------------------
# Cheap upper-bound estimates for branch-and-bound
# ---------------------------------------------------------------------------
def upper_bound_middle_royalty(cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> int:
    """Highest possible middle royalty for any 5-card hand."""
    return max(cfg.middle_by_category)


def upper_bound_bottom_royalty(cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> int:
    return max(cfg.bottom_by_category)


def upper_bound_top_royalty(cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> int:
    return max(
        cfg.top_high_card,
        max(cfg.top_pair_by_rank),
        max(cfg.top_trips_by_rank),
    )


__all__ = [
    "eval_5_by_mask",
    "eval_3_by_mask",
    "is_continuation",
    "bottom_guarantees_continuation",
    "top_guarantees_continuation",
    "bottom_royalty",
    "middle_royalty",
    "top_royalty_value",
    "upper_bound_middle_royalty",
    "upper_bound_bottom_royalty",
    "upper_bound_top_royalty",
]
