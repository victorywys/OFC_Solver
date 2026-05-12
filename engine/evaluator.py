"""Hand evaluator for 3-card (top) and 5-card (middle/bottom) hands.

Design choices:
    - Each hand evaluates to a `HandRank` = (category: int, kickers: tuple[int, ...]).
      Direct Python tuple comparison gives correct ordering, including across
      rows (top trips = TRIPS=3 < FULL_HOUSE=6 on bottom).
    - Categories are unified between 3-card and 5-card so `bottom >= middle >= top`
      reduces to a tuple compare. (3-card hands only produce HIGH_CARD, PAIR, TRIPS.)
    - Joker support: a joker may substitute for any non-joker card not already in
      the hand. We enumerate replacements and pick the maximum rank. The no-joker
      fast path is memoized.

Performance notes:
    - The no-joker 5-card path is hot. We sort ranks once, build a small
      multiplicity profile, and avoid per-call allocations in the common case.
    - We expose the canonical sorted-tuple form as cache key, so callers can
      pre-sort if they evaluate the same hand many times.
    - `evaluate_5`/`evaluate_3` use a `max(cards) < NUM_STD_CARDS` joker fast
      path so the common (no-joker) case avoids the per-card joker scan.
    - Future: replace this with a 32-bit packed lookup using bit tricks
      (see Cactus Kev / TwoPlusTwo evaluators). The interface returns
      hashable comparable values so swap-in is straightforward.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Sequence

from .cards import (
    NUM_STD_CARDS,
    RANK_5,
    RANK_A,
    card_rank,
    card_suit,
    is_joker,
)

# ---------------------------------------------------------------------------
# Category constants. Ordered so direct integer comparison gives poker order.
# ---------------------------------------------------------------------------
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8
ROYAL_FLUSH = 9

CATEGORY_NAMES = {
    HIGH_CARD: "high_card",
    PAIR: "pair",
    TWO_PAIR: "two_pair",
    TRIPS: "trips",
    STRAIGHT: "straight",
    FLUSH: "flush",
    FULL_HOUSE: "full_house",
    QUADS: "quads",
    STRAIGHT_FLUSH: "straight_flush",
    ROYAL_FLUSH: "royal_flush",
}

HandRank = tuple  # alias; (category:int, kickers:tuple[int,...])


# ---------------------------------------------------------------------------
# Internal: 5-card no-joker evaluation, memoized on canonical sorted tuple.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1 << 16)
def _eval_5_no_joker_sorted(cards: tuple) -> HandRank:
    """Evaluate exactly 5 distinct standard cards. `cards` need not be sorted."""
    c0, c1, c2, c3, c4 = cards
    r0, r1, r2, r3, r4 = c0 >> 2, c1 >> 2, c2 >> 2, c3 >> 2, c4 >> 2
    s0, s1, s2, s3, s4 = c0 & 3, c1 & 3, c2 & 3, c3 & 3, c4 & 3
    is_flush = s0 == s1 == s2 == s3 == s4

    # 5-element descending sort (manual unrolled is fastest in pure Python)
    rs = [r0, r1, r2, r3, r4]
    rs.sort(reverse=True)
    a, b, c, d, e = rs

    # Manual rank-count multiplicity (avoids Counter overhead).
    # Walk descending so quad/trip/pair/single ranks come out high-to-low.
    quad_r = -1
    trip_r = -1
    pair1 = -1
    pair2 = -1
    s1_ = -1
    s2_ = -1
    s3_ = -1
    s4_ = -1

    # Group identical ranks in the already-sorted list:
    # `rs` is descending, so equal ranks are adjacent.
    i = 0
    while i < 5:
        r = rs[i]
        j = i + 1
        while j < 5 and rs[j] == r:
            j += 1
        n = j - i
        if n == 4:
            quad_r = r
        elif n == 3:
            trip_r = r
        elif n == 2:
            if pair1 < 0:
                pair1 = r
            else:
                pair2 = r
        else:  # n == 1
            if s1_ < 0:
                s1_ = r
            elif s2_ < 0:
                s2_ = r
            elif s3_ < 0:
                s3_ = r
            else:
                s4_ = r
        i = j

    # Straight detection (only meaningful when all ranks distinct)
    straight_high = -1
    if quad_r < 0 and trip_r < 0 and pair1 < 0:
        if a - e == 4:
            straight_high = a
        elif a == RANK_A and b == 3 and c == 2 and d == 1 and e == 0:
            straight_high = RANK_5

    if is_flush and straight_high >= 0:
        if straight_high == RANK_A:
            return (ROYAL_FLUSH, (RANK_A,))
        return (STRAIGHT_FLUSH, (straight_high,))
    if quad_r >= 0:
        return (QUADS, (quad_r, s1_))
    if trip_r >= 0 and pair1 >= 0:
        return (FULL_HOUSE, (trip_r, pair1))
    if is_flush:
        return (FLUSH, (a, b, c, d, e))
    if straight_high >= 0:
        return (STRAIGHT, (straight_high,))
    if trip_r >= 0:
        return (TRIPS, (trip_r, s1_, s2_))
    if pair2 >= 0:
        return (TWO_PAIR, (pair1, pair2, s1_))
    if pair1 >= 0:
        return (PAIR, (pair1, s1_, s2_, s3_))
    return (HIGH_CARD, (a, b, c, d, e))


@lru_cache(maxsize=1 << 14)
def _eval_3_no_joker_sorted(cards: tuple) -> HandRank:
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    if ranks[0] == ranks[1] == ranks[2]:
        return (TRIPS, (ranks[0],))
    if ranks[0] == ranks[1]:
        return (PAIR, (ranks[0], ranks[2]))
    if ranks[1] == ranks[2]:
        return (PAIR, (ranks[1], ranks[0]))
    return (HIGH_CARD, tuple(ranks))


# ---------------------------------------------------------------------------
# Public evaluators (joker-aware).
# ---------------------------------------------------------------------------
def evaluate_5(cards: Sequence[int]) -> HandRank:
    """Best 5-card hand rank. Handles 0/1/2 jokers (wildcards)."""
    # Convert to tuple once so the cache key is hashable and stable.
    # Calling with a tuple (the common heuristic path) skips this entirely.
    t = cards if type(cards) is tuple else tuple(cards)
    return _evaluate_5_cached(t)


@lru_cache(maxsize=1 << 16)
def _evaluate_5_cached(cards: tuple) -> HandRank:
    if len(cards) != 5:
        raise ValueError(f"evaluate_5 requires 5 cards, got {len(cards)}")
    # Fast path: no jokers (the common case for fantasy/heuristic policies).
    # Joker ids are >= NUM_STD_CARDS, so a single max() suffices.
    if max(cards) < NUM_STD_CARDS:
        return _eval_5_no_joker_sorted(cards)

    jokers = 0
    nonj: list[int] = []
    for c in cards:
        if is_joker(c):
            jokers += 1
        else:
            nonj.append(c)
    if jokers == 0:
        return _eval_5_no_joker_sorted(tuple(nonj))

    used = set(nonj)
    pool = [c for c in range(NUM_STD_CARDS) if c not in used]
    if jokers == 1:
        best = None
        for r in pool:
            v = _eval_5_no_joker_sorted(tuple(nonj + [r]))
            if best is None or v > best:
                best = v
        return best  # type: ignore[return-value]

    # 2 jokers
    best = None
    for r1, r2 in combinations(pool, 2):
        v = _eval_5_no_joker_sorted(tuple(nonj + [r1, r2]))
        if best is None or v > best:
            best = v
    return best  # type: ignore[return-value]


def evaluate_3(cards: Sequence[int]) -> HandRank:
    """Best 3-card top-row hand rank. Handles 0/1/2 jokers."""
    t = cards if type(cards) is tuple else tuple(cards)
    return _evaluate_3_cached(t)


@lru_cache(maxsize=1 << 14)
def _evaluate_3_cached(cards: tuple) -> HandRank:
    if len(cards) != 3:
        raise ValueError(f"evaluate_3 requires 3 cards, got {len(cards)}")
    # Fast path: no jokers.
    if max(cards) < NUM_STD_CARDS:
        return _eval_3_no_joker_sorted(cards)

    jokers = 0
    nonj: list[int] = []
    for c in cards:
        if is_joker(c):
            jokers += 1
        else:
            nonj.append(c)
    if jokers == 0:
        return _eval_3_no_joker_sorted(tuple(nonj))

    # With jokers + 3-card: trips of the highest non-joker rank is always optimal
    # if any non-joker exists (jokers fill to make trips). With 0 non-jokers
    # (all 3 jokers, impossible since only 2 jokers), trips of A.
    if jokers == 3:  # impossible deck-wise but defensive
        return (TRIPS, (RANK_A,))
    if jokers == 2:
        # only 1 real card: jokers become two of that rank -> trips
        r = card_rank(nonj[0])
        return (TRIPS, (r,))
    # jokers == 1 with 2 real cards
    r0 = card_rank(nonj[0])
    r1 = card_rank(nonj[1])
    if r0 == r1:
        return (TRIPS, (r0,))
    hi, lo = (r0, r1) if r0 > r1 else (r1, r0)
    return (PAIR, (hi, lo))


def hand_rank_str(rank: HandRank) -> str:
    cat, kickers = rank
    return f"{CATEGORY_NAMES[cat]}{tuple(kickers)}"


__all__ = [
    "HIGH_CARD", "PAIR", "TWO_PAIR", "TRIPS", "STRAIGHT", "FLUSH",
    "FULL_HOUSE", "QUADS", "STRAIGHT_FLUSH", "ROYAL_FLUSH",
    "CATEGORY_NAMES", "HandRank",
    "evaluate_5", "evaluate_3", "hand_rank_str",
]
