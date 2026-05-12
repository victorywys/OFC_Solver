"""Heuristic completion policy for OFC Pineapple.

Spec priorities (in order):
    1. Avoid foul.
    2. Preserve row ordering flexibility.
    3. Maximize royalties.
    4. Pursue Fantasy EV.
    5. Reduce scoop risk.

Design
------
For each candidate `Action`, we score the resulting partial board with a
weighted sum of features:

    score(action)
        = sum_per_row( row_strength(prof, cards) * row_position_weight )
        + completed_row_royalty_bonus
        + fantasy_top_bonus
        - ordering_penalty
        - discard_penalty

`row_strength` is split into two regimes:
    * full row (n == capacity): use the *actual* evaluator category (exact)
    * partial row: feature-based (pair/two-pair/trips/full-house/quads
      detection, flush count, longest distinct-rank run, high-card lift),
      plus a small constant per remaining empty slot (potential).

`ordering_penalty` uses both the committed-vs-possible category bounds
(certain-foul detection) and a fine-grained (max_mult, max_mult_rank,
second_mult) comparison to detect "top will likely dominate middle" early.

This module is deliberately allocation-light. It can comfortably drive
Phase 4 rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from engine.cards import (
    NUM_RANKS,
    RANK_A,
    RANK_K,
    RANK_Q,
    card_rank,
    card_suit,
    is_joker,
)
from engine.evaluator import (
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
from state.action import (
    Action,
    enumerate_initial_actions,
    enumerate_pineapple_actions,
    iter_fantasy_actions,
)
from state.board import (
    PlayerBoard,
    ROW_CAPACITY,
    SLOT_BOTTOM,
    SLOT_DISCARD,
    SLOT_MIDDLE,
    SLOT_TOP,
)
from state.game_state import GameState

from .policy import Policy


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeuristicWeights:
    """Feature weights for the heuristic policy. All values configurable."""

    # row position multipliers (bottom > middle > top)
    row_top: float = 1.0
    row_middle: float = 1.2
    row_bottom: float = 1.5

    # partial-row feature weights
    w_pair: float = 6.0
    w_two_pair_extra: float = 6.0   # added on top of w_pair when 2 pairs detected
    w_trips: float = 22.0
    w_full_house_extra: float = 12.0  # on top of w_trips when paired-with
    w_quads: float = 60.0
    w_flush_each: float = 3.0       # per same-suit card beyond first
    w_straight_each: float = 2.5    # per card in longest run beyond first
    w_high_card: float = 0.15
    w_potential_per_slot: float = 1.5  # bonus per empty slot

    # full-row category strength (using exact evaluator)
    full_row_strength: tuple = (
        0.5,    # HIGH_CARD
        8.0,    # PAIR
        14.0,   # TWO_PAIR
        24.0,   # TRIPS
        30.0,   # STRAIGHT
        40.0,   # FLUSH
        60.0,   # FULL_HOUSE
        90.0,   # QUADS
        130.0,  # STRAIGHT_FLUSH
        200.0,  # ROYAL_FLUSH
    )

    # royalty multiplier for completed rows
    w_complete_royalty: float = 1.0
    # Discount factor applied to ``full_row_strength`` when a partial row
    # has already committed at TRIPS or stronger. Slightly < 1 so that a
    # fully-realised row is still preferred over a committed-but-partial
    # row at the same category, all else equal.
    w_partial_committed_factor: float = 0.92

    # foul / ordering penalties
    w_foul: float = 1500.0          # certain foul (committed > possible)
    w_order_violation: float = 22.0 # per (gap+1) * filled_count_other_row unit
    w_expected_order: float = 16.0  # per category-unit gap of expected-cat violation

    # fantasy
    w_fantasy_top_qq: float = 6.0
    w_fantasy_top_kk: float = 10.0
    w_fantasy_top_aa: float = 14.0
    w_fantasy_top_trips: float = 28.0

    # pineapple discard
    w_discard_value: float = 0.6   # base: prefer discarding low-rank cards
    w_discard_match: float = 4.0   # extra penalty if card pairs with board material

    # Per-joker penalty when a joker on a row contributes nothing to the
    # row's committed category — i.e. dropping it would leave the row at
    # the same committed strength. Such jokers are "dead weight" because
    # they could have been placed elsewhere as flexible material.
    w_wasted_joker: float = 12.0


DEFAULT_WEIGHTS = HeuristicWeights()


# ---------------------------------------------------------------------------
# RowProfile: cheap structural summary of a (partial) row
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RowProfile:
    n: int
    capacity: int
    max_mult: int           # max rank multiplicity (jokers boost)
    max_mult_rank: int
    second_mult: int        # second-highest multiplicity (excluding max_mult_rank)
    max_suit_count: int     # max same-suit count (jokers count as wild)
    longest_run: int        # longest connected run length (1..5), wheel-aware
    top_rank: int           # highest non-joker rank seen (-1 if none)
    n_jokers: int


def _profile_row(cards: Sequence[int], capacity: int) -> RowProfile:
    """Compute the structural profile of a (partial) row.

    Pure: depends only on the multiset of cards + capacity. Sorted-tuple
    key dispatch lets us memoize, since the same row state recurs across
    many candidates within a single `score_action` sweep (most actions
    only change 1-2 rows, leaving the others identical).
    """
    if not cards:
        # Skip cache: zero allocation, hottest possible path.
        return _EMPTY_PROFILES.get(capacity) or _empty_profile(capacity)
    # Hashable key. tuple(sorted(...)) is ~1us for n<=5; allocation
    # dominates only when the cache misses, which is exactly when we
    # would have done the full work anyway.
    key = tuple(sorted(cards))
    return _profile_row_cached(key, capacity)


_EMPTY_PROFILES: dict[int, RowProfile] = {}


def _empty_profile(capacity: int) -> RowProfile:
    prof = RowProfile(0, capacity, 0, -1, 0, 0, 0, -1, 0)
    _EMPTY_PROFILES[capacity] = prof
    return prof


@lru_cache(maxsize=1 << 16)
def _profile_row_cached(cards: tuple[int, ...], capacity: int) -> RowProfile:
    """Compute RowProfile for a sorted card tuple. LRU-cached.

    The hot path through `score_action` calls this 3 times per candidate.
    Many candidates share at least one identical row (they only touch
    1-2 of the 3 rows), so a single `score_action` sweep over ~200 legal
    actions hits the cache hard. Across an entire rollout chunk the cache
    stays warm for the most common row states encountered.
    """
    rank_count = [0] * NUM_RANKS
    suit_count = [0, 0, 0, 0]
    n_jokers = 0
    top_rank = -1
    for c in cards:
        if is_joker(c):
            n_jokers += 1
            continue
        r = card_rank(c)
        s = card_suit(c)
        rank_count[r] += 1
        suit_count[s] += 1
        if r > top_rank:
            top_rank = r

    # Manual max/max_index/second_max — avoids three separate `max(...)`
    # calls plus a generator expression. Saves ~30% over the previous
    # implementation, which dominated `_profile_row` self-time.
    base_max = 0
    max_mult_rank = -1
    second_mult = 0
    for r in range(NUM_RANKS):
        v = rank_count[r]
        if v > base_max:
            second_mult = base_max
            base_max = v
            max_mult_rank = r
        elif v == base_max and v > 0:
            # Tie at top: the old code preferred the HIGHEST rank as
            # max_mult_rank, and counted the "demoted" rank's count
            # toward second_mult. Reproduce both.
            second_mult = base_max
            max_mult_rank = r
        elif v > second_mult:
            second_mult = v

    max_mult = base_max + n_jokers
    suit_max = suit_count[0]
    if suit_count[1] > suit_max:
        suit_max = suit_count[1]
    if suit_count[2] > suit_max:
        suit_max = suit_count[2]
    if suit_count[3] > suit_max:
        suit_max = suit_count[3]
    max_suit_count = suit_max + n_jokers

    has_rank = [rc > 0 for rc in rank_count]
    longest_run = _longest_straight_run(has_rank, n_jokers)

    return RowProfile(
        n=len(cards),
        capacity=capacity,
        max_mult=max_mult,
        max_mult_rank=max_mult_rank,
        second_mult=second_mult,
        max_suit_count=max_suit_count,
        longest_run=longest_run,
        top_rank=top_rank,
        n_jokers=n_jokers,
    )


def _longest_straight_run(has_rank: list[bool], n_jokers: int) -> int:
    """Best 5-card straight progress, allowing wheel and joker fills."""
    extended = [has_rank[12]] + has_rank  # A also at index 0
    best = 0
    for start in range(0, 14 - 5 + 1):
        present = sum(extended[start:start + 5])
        gaps = 5 - present
        score = present + min(gaps, n_jokers)
        if score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Row strength: partial features OR exact evaluator if full
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1 << 13)
def _committed_category(prof: RowProfile, is_top: bool) -> int:
    """Lowest poker category guaranteed by the cards already in the row,
    irrespective of any cards yet to be dealt. Used so that partial rows
    that have already locked in a strong category (e.g. ``AAA + 1 joker``
    on the bottom is unavoidably QUADS) are scored at the strength of
    that category — not at the much lower "partial features" sum.

    Joker contributions are baked into ``prof.max_mult`` /
    ``prof.max_suit_count`` / ``prof.longest_run`` already, so this
    function only needs to read those.
    """
    if is_top:
        if prof.max_mult >= 3:
            return TRIPS
        if prof.max_mult >= 2:
            return PAIR
        return HIGH_CARD
    if prof.max_mult >= 4:
        return QUADS
    if prof.max_mult >= 3 and prof.second_mult >= 2:
        return FULL_HOUSE
    if prof.max_mult >= 3:
        return TRIPS
    if prof.max_mult >= 2 and prof.second_mult >= 2:
        return TWO_PAIR
    if prof.max_mult >= 2:
        return PAIR
    return HIGH_CARD


def _row_strength(
    prof: RowProfile,
    cards: Sequence[int],
    w: HeuristicWeights,
    is_top: bool,
) -> float:
    if prof.n == prof.capacity:
        # full row: use the actual evaluator category for exact strength
        if is_top:
            cat, _ = evaluate_3(cards)
        else:
            cat, _ = evaluate_5(cards)
        s = w.full_row_strength[cat]
        if prof.top_rank >= 0:
            s += prof.top_rank * w.w_high_card
        return s
    # Partial row — fully a function of (prof, w, is_top). Fast-path the
    # common case where ``w is DEFAULT_WEIGHTS`` to avoid hashing 20+
    # weight floats on every call.
    if w is DEFAULT_WEIGHTS:
        return _row_strength_partial_default(prof, is_top)
    return _row_strength_partial_compute(prof, w, is_top)


@lru_cache(maxsize=1 << 13)
def _row_strength_partial_default(prof: RowProfile, is_top: bool) -> float:
    """Cached default-weights partial-row strength. (prof, is_top) key."""
    return _row_strength_partial_compute(prof, DEFAULT_WEIGHTS, is_top)


def _row_strength_partial_compute(
    prof: RowProfile,
    w: HeuristicWeights,
    is_top: bool,
) -> float:
    """Uncached partial-row strength computation."""
    # Partial row.
    #
    # When the cards already locked into the row commit a category at
    # TRIPS or stronger, the row is essentially "as strong as a full
    # row of that category" — the only thing the missing cards change
    # is the kicker. Using ``full_row_strength`` for that category here
    # closes the gap that previously made the heuristic prefer
    # "complete the bottom NOW" plays (which can waste jokers).
    committed = _committed_category(prof, is_top)
    if committed >= TRIPS:
        # Tiny uncertainty discount so a partial-but-committed row is
        # still very slightly less attractive than a fully-realised one
        # at the same category (the kicker isn't fixed yet).
        s = w.full_row_strength[committed] * w.w_partial_committed_factor
    else:
        # Below TRIPS the row's eventual category is genuinely unsettled;
        # fall back to the additive partial-feature score that captures
        # incremental progress (pair / two-pair / suit / run).
        s = 0.0
        if prof.max_mult == 2:
            s += w.w_pair
        if not is_top and prof.max_mult >= 2 and prof.second_mult >= 2:
            s += w.w_two_pair_extra

    # Suit / straight progression are still informative regardless of
    # the rank-based committed category, so add them on top (they
    # describe potential, not committed strength).
    if not is_top:
        if prof.max_suit_count >= 2:
            s += w.w_flush_each * (prof.max_suit_count - 1)
        if prof.longest_run >= 2:
            s += w.w_straight_each * (prof.longest_run - 1)

    if prof.top_rank >= 0:
        s += prof.top_rank * w.w_high_card

    # potential: empty slots are valuable (preserves flexibility)
    s += (prof.capacity - prof.n) * w.w_potential_per_slot
    return s


# ---------------------------------------------------------------------------
# Cross-row ordering / foul penalties
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1 << 15)
def _max_possible_category(prof: RowProfile) -> int:
    free = prof.capacity - prof.n
    can = HIGH_CARD
    if prof.capacity >= 5:
        if prof.max_suit_count + free >= 5 and prof.longest_run + free >= 5:
            can = max(can, STRAIGHT_FLUSH)
        if prof.max_suit_count + free >= 5:
            can = max(can, FLUSH)
        if prof.longest_run + free >= 5:
            can = max(can, STRAIGHT)
        if prof.max_mult + free >= 4:
            can = max(can, QUADS)
        if (
            prof.max_mult + free >= 3
            and prof.second_mult + max(0, free - max(0, 3 - prof.max_mult)) >= 2
        ):
            can = max(can, FULL_HOUSE)
    if prof.max_mult + free >= 3:
        can = max(can, TRIPS)
    if prof.max_mult >= 2 and prof.second_mult + free >= 2:
        can = max(can, TWO_PAIR)
    if prof.max_mult + free >= 2:
        can = max(can, PAIR)
    return can


@lru_cache(maxsize=1 << 14)
def _min_committed_category(prof: RowProfile) -> int:
    if prof.max_mult >= 4:
        return QUADS
    if prof.max_mult == 3 and prof.second_mult >= 2:
        return FULL_HOUSE
    if prof.max_mult == 3:
        return TRIPS
    if prof.max_mult == 2 and prof.second_mult >= 2:
        return TWO_PAIR
    if prof.max_mult == 2:
        return PAIR
    return HIGH_CARD


def _partial_strength_tuple(prof: RowProfile) -> tuple[int, int, int]:
    """Lexicographic strength signature for ordering-risk comparisons.

    Layout: ``(max_mult, second_mult, max_mult_rank)``.

    The order of fields matters. An earlier version put
    ``max_mult_rank`` ahead of ``second_mult``, which made

        pair-of-aces (2, A=12, 0)  >  two-pair-of-lows (2, 5, 2)

    even though two-pair is strictly the stronger *poker* hand. With
    ``second_mult`` second, the comparison correctly distinguishes
    one-pair from two-pair before the kicker rank is considered. The
    rank is still meaningful as a tiebreaker (AA dominates KK at the
    same shape) and remains last.
    """
    return (
        prof.max_mult,
        prof.second_mult,
        prof.max_mult_rank if prof.max_mult_rank >= 0 else -1,
    )


@lru_cache(maxsize=1 << 14)
def _expected_category(prof: RowProfile) -> float:
    """Smooth interpolation between min_committed and max_possible category.

    Uses β = 0.5 + 0.5*(n/capacity) so that:
        - empty row -> 0.5*min + 0.5*max  (acknowledges potential)
        - full row  -> min == max == actual category
    Returns a float (category indices live in [0, 9]).
    """
    if prof.capacity == 0:
        return float(HIGH_CARD)
    a = prof.n / prof.capacity
    beta = 0.5 + 0.5 * a
    return beta * _min_committed_category(prof) + (1.0 - beta) * _max_possible_category(prof)


def _ordering_penalty(
    top: RowProfile,
    mid: RowProfile,
    bot: RowProfile,
    w: HeuristicWeights,
) -> float:
    # Fast-path the common default-weights case. Avoids hashing the
    # 20+ HeuristicWeights floats on every cache lookup.
    if w is DEFAULT_WEIGHTS:
        return _ordering_penalty_default(top, mid, bot)
    return _ordering_penalty_compute(top, mid, bot, w)


@lru_cache(maxsize=1 << 15)
def _ordering_penalty_default(
    top: RowProfile,
    mid: RowProfile,
    bot: RowProfile,
) -> float:
    """Cached default-weights ordering penalty."""
    return _ordering_penalty_compute(top, mid, bot, DEFAULT_WEIGHTS)


def _ordering_penalty_compute(
    top: RowProfile,
    mid: RowProfile,
    bot: RowProfile,
    w: HeuristicWeights,
) -> float:
    pen = 0.0

    # certain fouls: committed > opponent's max possible
    top_min = _min_committed_category(top)
    mid_min = _min_committed_category(mid)
    mid_max = _max_possible_category(mid)
    bot_max = _max_possible_category(bot)
    if top_min > mid_max:
        pen += w.w_foul
    if top_min > bot_max:
        pen += w.w_foul
    if mid_min > bot_max:
        pen += w.w_foul

    # likely fouls / ordering risk via partial-strength tuple (catches
    # rank-level dominance when max_mult is the same)
    s_top = _partial_strength_tuple(top)
    s_mid = _partial_strength_tuple(mid)
    s_bot = _partial_strength_tuple(bot)

    if mid.n >= 2 and top.max_mult >= 2 and s_top > s_mid:
        gap = top.max_mult - max(mid.max_mult, 1)
        pen += w.w_order_violation * (gap + 1) * mid.n
    if bot.n >= 2 and mid.max_mult >= 2 and s_mid > s_bot:
        gap = mid.max_mult - max(bot.max_mult, 1)
        pen += w.w_order_violation * (gap + 1) * bot.n
    if bot.n >= 2 and top.max_mult >= 2 and s_top > s_bot:
        gap = top.max_mult - max(bot.max_mult, 1)
        pen += w.w_order_violation * (gap + 1) * bot.n

    # expected-category ordering risk: catches "middle on track for straight
    # while bottom committed to a pair" patterns that the partial-strength
    # tuple misses. Weight by the *upper* row's commitment so an empty row
    # above does not falsely penalize commitments below.
    e_top = _expected_category(top)
    e_mid = _expected_category(mid)
    e_bot = _expected_category(bot)
    if e_top > e_mid:
        pen += w.w_expected_order * (e_top - e_mid) * (1 + top.n)
    if e_mid > e_bot:
        pen += w.w_expected_order * (e_mid - e_bot) * (1 + mid.n)
    if e_top > e_bot:
        pen += w.w_expected_order * (e_top - e_bot) * (1 + top.n)
    return pen


# Backwards-compatible alias for tests
_foul_penalty = _ordering_penalty


# ---------------------------------------------------------------------------
# Fantasy bias and discard penalty
# ---------------------------------------------------------------------------
def _fantasy_top_bonus(top: RowProfile, w: HeuristicWeights) -> float:
    if top.max_mult >= 3:
        return w.w_fantasy_top_trips
    if top.max_mult == 2:
        r = top.max_mult_rank
        if r == RANK_A:
            return w.w_fantasy_top_aa
        if r == RANK_K:
            return w.w_fantasy_top_kk
        if r == RANK_Q:
            return w.w_fantasy_top_qq
    return 0.0


def _discard_penalty(
    discards: tuple[int, ...],
    board: PlayerBoard,
    w: HeuristicWeights,
) -> float:
    if not discards:
        return 0.0
    # The (on_ranks, on_suits) pair is purely a function of the board's
    # placed cards and is *invariant* across all candidate actions for a
    # given `act()` call (~200 candidates). Cache it on the board's
    # row tuples so we pay the O(board_size) scan once per act().
    on_ranks, on_suits = _board_rank_suit_counts(
        tuple(board.top), tuple(board.middle), tuple(board.bottom)
    )

    pen = 0.0
    for c in discards:
        if is_joker(c):
            pen += 18 * w.w_discard_value
            continue
        r = card_rank(c)
        s = card_suit(c)
        v = r * w.w_discard_value
        if on_ranks.get(r, 0) >= 2:
            v += w.w_discard_match * 2
        elif on_ranks.get(r, 0) == 1:
            v += w.w_discard_match
        if on_suits.get(s, 0) >= 3:
            v += w.w_discard_match * 0.5
        pen += v
    return pen


@lru_cache(maxsize=1 << 12)
def _board_rank_suit_counts(
    top_t: tuple[int, ...],
    mid_t: tuple[int, ...],
    bot_t: tuple[int, ...],
) -> tuple[dict[int, int], dict[int, int]]:
    """Compute on-board rank and suit counts. Cached on the row tuples
    so all 200 candidates in an `act()` sweep share one computation."""
    on_ranks: dict[int, int] = {}
    on_suits: dict[int, int] = {}
    for row in (top_t, mid_t, bot_t):
        for c in row:
            if is_joker(c):
                continue
            r = card_rank(c)
            s = card_suit(c)
            on_ranks[r] = on_ranks.get(r, 0) + 1
            on_suits[s] = on_suits.get(s, 0) + 1
    return on_ranks, on_suits


# ---------------------------------------------------------------------------
# Action scoring
# ---------------------------------------------------------------------------
@dataclass
class ActionScore:
    total: float
    row_strength: float
    royalty_bonus: float
    fantasy_bonus: float
    foul_penalty: float
    discard_penalty: float


def _row_committed_royalty(
    prof: RowProfile,
    cards: list[int],
    slot: int,
    cfg: RoyaltyConfig,
) -> float:
    """Royalty value the row is *guaranteed* to score, regardless of any
    cards still to be drawn.

    For a fully-placed row this matches the actual royalty (we use the
    joker-aware evaluator). For a partial row we use ``_committed_category``
    so that an already-locked category (e.g. AAA + 1 joker = QUADS-A on
    bottom) gets its royalty credited up-front, instead of waiting for
    the row to be filled and rewarding "stuff every card in now" plays.
    """
    is_top = slot == SLOT_TOP
    full_cap = 3 if is_top else 5
    # Full row: use the actual evaluator (joker-aware via Board.evaluate
    # is not needed here because we only ever have a single row's cards;
    # the per-row evaluators give the same answer in isolation).
    if len(cards) == full_cap:
        if is_top:
            return royalty_top(evaluate_3(cards), cfg)
        if slot == SLOT_MIDDLE:
            return royalty_middle(evaluate_5(cards), cfg)
        return royalty_bottom(evaluate_5(cards), cfg)
    # Partial row: function of (prof, slot, cfg). Fast-path the common
    # case where ``cfg is DEFAULT_ROYALTIES`` to avoid hashing the cfg
    # tuples on every call.
    if cfg is DEFAULT_ROYALTIES:
        return _row_committed_royalty_partial_default(prof, slot)
    return _row_committed_royalty_partial_compute(prof, slot, cfg)


@lru_cache(maxsize=1 << 13)
def _row_committed_royalty_partial_default(prof: RowProfile, slot: int) -> float:
    """Cached default-cfg partial royalty. (prof, slot) key."""
    return _row_committed_royalty_partial_compute(prof, slot, DEFAULT_ROYALTIES)


def _row_committed_royalty_partial_compute(
    prof: RowProfile,
    slot: int,
    cfg: RoyaltyConfig,
) -> float:
    """Uncached partial-row committed royalty."""
    is_top = slot == SLOT_TOP
    cat = _committed_category(prof, is_top)
    if is_top:
        # Top royalties depend on the rank, which is `max_mult_rank` for
        # pair / trips. With pure-joker rows the rank is unknown and we
        # conservatively assume rank 0 (no royalty) to avoid pretending
        # we know the kicker.
        r = prof.max_mult_rank
        if cat == TRIPS and r >= 0:
            return float(cfg.top_trips_by_rank[r])
        if cat == PAIR and r >= 0:
            return float(cfg.top_pair_by_rank[r])
        return 0.0
    if slot == SLOT_MIDDLE:
        return float(cfg.middle_by_category[cat])
    return float(cfg.bottom_by_category[cat])


# Backwards-compatible alias for any external callers.
def _row_complete_royalty(
    cards: list[int], slot: int, cfg: RoyaltyConfig
) -> float:
    full_cap = 3 if slot == SLOT_TOP else 5
    if len(cards) != full_cap:
        return 0.0
    if slot == SLOT_TOP:
        return royalty_top(evaluate_3(cards), cfg)
    if slot == SLOT_MIDDLE:
        return royalty_middle(evaluate_5(cards), cfg)
    return royalty_bottom(evaluate_5(cards), cfg)


@lru_cache(maxsize=1 << 13)
def _wasted_jokers_in_row(prof: RowProfile, is_top: bool) -> int:
    """Count jokers on this row whose removal would NOT lower the row's
    committed category.

    A joker is "wasted" in the strategic sense the user pointed out: e.g.
    with ``A A A * *`` on the bottom, the *second* joker doesn't change
    the committed category (still QUADS-A whether one or two jokers are
    alongside the trip aces); the freed-up joker would be more valuable
    on another row where it could move the category needle.
    """
    if prof.n_jokers == 0:
        return 0
    cat_full = _committed_category(prof, is_top)
    base_max = max(0, prof.max_mult - prof.n_jokers)
    wasted = 0
    for k in range(1, prof.n_jokers + 1):
        hypo = RowProfile(
            n=prof.n - k,
            capacity=prof.capacity,
            max_mult=base_max + (prof.n_jokers - k),
            max_mult_rank=prof.max_mult_rank,
            second_mult=prof.second_mult,
            max_suit_count=max(0, prof.max_suit_count - k),
            longest_run=max(0, prof.longest_run - k),
            top_rank=prof.top_rank,
            n_jokers=prof.n_jokers - k,
        )
        if _committed_category(hypo, is_top) >= cat_full:
            wasted = k
        else:
            break
    return wasted


def _wasted_joker_penalty(
    top_p: RowProfile,
    mid_p: RowProfile,
    bot_p: RowProfile,
    w: HeuristicWeights,
) -> float:
    if w.w_wasted_joker <= 0.0:
        return 0.0
    # Fast-path the default-weights case to avoid hashing the full
    # HeuristicWeights on every call. The result is just
    # `wasted_count * w.w_wasted_joker`, so we cache the count and
    # multiply by the live weight.
    if w is DEFAULT_WEIGHTS:
        n = _wasted_joker_count_default(top_p, mid_p, bot_p)
    else:
        n = (
            _wasted_jokers_in_row(top_p, is_top=True)
            + _wasted_jokers_in_row(mid_p, is_top=False)
            + _wasted_jokers_in_row(bot_p, is_top=False)
        )
    return n * w.w_wasted_joker


@lru_cache(maxsize=1 << 14)
def _wasted_joker_count_default(
    top_p: RowProfile,
    mid_p: RowProfile,
    bot_p: RowProfile,
) -> int:
    """Cached wasted-joker count across all 3 rows."""
    return (
        _wasted_jokers_in_row(top_p, is_top=True)
        + _wasted_jokers_in_row(mid_p, is_top=False)
        + _wasted_jokers_in_row(bot_p, is_top=False)
    )


def score_action(
    action: Action,
    board: PlayerBoard,
    cfg: RoyaltyConfig = DEFAULT_ROYALTIES,
    w: HeuristicWeights = DEFAULT_WEIGHTS,
) -> ActionScore:
    """Score an action by simulating its application and analyzing the
    resulting partial state. Pure: does not mutate `board`."""
    nb = action.apply(board)
    top_p = _profile_row(nb.top, ROW_CAPACITY[SLOT_TOP])
    mid_p = _profile_row(nb.middle, ROW_CAPACITY[SLOT_MIDDLE])
    bot_p = _profile_row(nb.bottom, ROW_CAPACITY[SLOT_BOTTOM])

    row_total = (
        _row_strength(top_p, nb.top, w, is_top=True) * w.row_top
        + _row_strength(mid_p, nb.middle, w, is_top=False) * w.row_middle
        + _row_strength(bot_p, nb.bottom, w, is_top=False) * w.row_bottom
    )

    royalty_bonus = w.w_complete_royalty * (
        _row_committed_royalty(top_p, nb.top, SLOT_TOP, cfg)
        + _row_committed_royalty(mid_p, nb.middle, SLOT_MIDDLE, cfg)
        + _row_committed_royalty(bot_p, nb.bottom, SLOT_BOTTOM, cfg)
    )

    fantasy_bonus = _fantasy_top_bonus(top_p, w)
    foul_pen = _ordering_penalty(top_p, mid_p, bot_p, w)
    disc_pen = _discard_penalty(action.discards(), board, w)
    waste_pen = _wasted_joker_penalty(top_p, mid_p, bot_p, w)

    total = row_total + royalty_bonus + fantasy_bonus - foul_pen - disc_pen - waste_pen
    return ActionScore(
        total=total,
        row_strength=row_total,
        royalty_bonus=royalty_bonus,
        fantasy_bonus=fantasy_bonus,
        foul_penalty=foul_pen + waste_pen,
        discard_penalty=disc_pen,
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class HeuristicPolicy(Policy):
    """Greedy argmax over `score_action`. Random tie-break via seeded RNG."""

    name = "heuristic"

    def __init__(
        self,
        weights: HeuristicWeights = DEFAULT_WEIGHTS,
        royalty_cfg: RoyaltyConfig = DEFAULT_ROYALTIES,
        seed: int | None = None,
        fantasy_budget: int = 4096,
    ) -> None:
        import random

        self.weights = weights
        self.royalty_cfg = royalty_cfg
        self._rng = random.Random(seed)
        self.fantasy_budget = fantasy_budget

    def _candidate_actions(self, gs: GameState, player: int):
        hs = gs.hands[player]
        cards = list(hs.pending)
        if hs.fantasy_tier != FantasyTier.NORMAL:
            return list(
                iter_fantasy_actions(cards, hs.board, budget=self.fantasy_budget)
            )
        if gs.current_street == 1:
            return enumerate_initial_actions(cards)
        return enumerate_pineapple_actions(cards, hs.board)

    def act(self, gs: GameState, player: int) -> Action:
        hs = gs.hands[player]
        candidates = self._candidate_actions(gs, player)
        if not candidates:
            raise RuntimeError("no legal actions")

        best_score = float("-inf")
        ties: list[Action] = []
        for a in candidates:
            sc = score_action(a, hs.board, self.royalty_cfg, self.weights).total
            if sc > best_score:
                best_score = sc
                ties = [a]
            elif sc == best_score:
                ties.append(a)
        if len(ties) > 1:
            return self._rng.choice(ties)
        return ties[0]


__all__ = [
    "HeuristicPolicy",
    "HeuristicWeights",
    "DEFAULT_WEIGHTS",
    "ActionScore",
    "score_action",
    "RowProfile",
]
