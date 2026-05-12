"""Board validity, foul detection, and joker-aware joint resolution.

A board is foul if `bottom_rank >= middle_rank >= top_rank` is violated.
Because all `HandRank` tuples share a single category space, these
comparisons are simple Python tuple compares.

When the board contains jokers (wildcards), evaluating each row in
isolation is wrong: each row's evaluator will pick the substitution that
maximizes that row alone, even if the resulting layout fouls. Real OFC
rules let the player choose joker substitutions at scoring time, so
`resolve_board` enumerates joint joker assignments and returns the
non-fouling assignment that maximizes total royalties (falling back to
the lex-max ranks when no non-fouling assignment exists).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .cards import NUM_STD_CARDS, is_joker
from .evaluator import HandRank, evaluate_3, evaluate_5
from .royalties import (
    DEFAULT_ROYALTIES,
    RoyaltyConfig,
    royalty_bottom,
    royalty_middle,
    royalty_top,
)


def evaluate_rows(
    top: Sequence[int],
    middle: Sequence[int],
    bottom: Sequence[int],
) -> tuple[HandRank, HandRank, HandRank]:
    """Joker-naive per-row evaluation. Each row's joker is substituted to
    maximize THAT row only, ignoring the foul constraint. Use
    `resolve_board` for the joint, foul-aware resolution.
    """
    return evaluate_3(top), evaluate_5(middle), evaluate_5(bottom)


@lru_cache(maxsize=1 << 13)
def resolve_board(
    top: tuple,
    middle: tuple,
    bottom: tuple,
) -> tuple[HandRank, HandRank, HandRank, bool]:
    """Resolve any jokers jointly across (top, middle, bottom).

    Returns ``(top_rank, middle_rank, bottom_rank, fouled)`` where the
    ranks reflect the chosen joker assignment.

    Resolution rules (mirror real-game player choice at scoring time):

      1. Enumerate joint joker substitutions. Each joker independently
         maps to a standard card not already on the board, and the two
         jokers (when both present) must map to distinct cards.
      2. Among non-fouling assignments, pick the one with maximum total
         royalties. Tie-break by the lex max of (top, middle, bottom).
      3. If no non-fouling assignment exists, the board is fouled; we
         return the lex-max per-row ranks for diagnostic display.

    Caches on the (top, middle, bottom) tuple so repeated boards are O(1).
    """
    has_joker = (
        any(is_joker(c) for c in top)
        or any(is_joker(c) for c in middle)
        or any(is_joker(c) for c in bottom)
    )
    if not has_joker:
        t = evaluate_3(top)
        m = evaluate_5(middle)
        b = evaluate_5(bottom)
        return (t, m, b, not (b >= m >= t))

    # Locate joker positions (row index 0/1/2, slot index within row).
    joker_positions: list[tuple[int, int]] = []
    for i, c in enumerate(top):
        if is_joker(c):
            joker_positions.append((0, i))
    for i, c in enumerate(middle):
        if is_joker(c):
            joker_positions.append((1, i))
    for i, c in enumerate(bottom):
        if is_joker(c):
            joker_positions.append((2, i))
    n_jokers = len(joker_positions)

    # Build pool of standard cards not already placed on the board.
    used: set[int] = set()
    for c in top:
        if not is_joker(c):
            used.add(c)
    for c in middle:
        if not is_joker(c):
            used.add(c)
    for c in bottom:
        if not is_joker(c):
            used.add(c)
    pool = [c for c in range(NUM_STD_CARDS) if c not in used]

    # Enumerate joker assignments.
    if n_jokers == 1:
        substs: list[tuple[int, ...]] = [(c,) for c in pool]
    else:  # n_jokers == 2
        if joker_positions[0][0] == joker_positions[1][0]:
            # Both jokers in same row — evaluator is order-agnostic, so
            # combinations suffice (each unordered pair tested once).
            substs = []
            for i in range(len(pool)):
                ai = pool[i]
                for j in range(i + 1, len(pool)):
                    substs.append((ai, pool[j]))
        else:
            # Different rows — order matters; test all ordered distinct pairs.
            substs = [(a, b) for a in pool for b in pool if a != b]

    best_valid: tuple[HandRank, HandRank, HandRank] | None = None
    best_valid_royalty = -(1 << 30)
    best_invalid: tuple[HandRank, HandRank, HandRank] | None = None

    top_l = list(top)
    mid_l = list(middle)
    bot_l = list(bottom)

    for assignment in substs:
        # Apply substitution.
        for (row, idx), card in zip(joker_positions, assignment):
            if row == 0:
                top_l[idx] = card
            elif row == 1:
                mid_l[idx] = card
            else:
                bot_l[idx] = card
        t = evaluate_3(tuple(top_l))
        m = evaluate_5(tuple(mid_l))
        b = evaluate_5(tuple(bot_l))
        if b >= m >= t:
            roy = (
                royalty_top(t, DEFAULT_ROYALTIES)
                + royalty_middle(m, DEFAULT_ROYALTIES)
                + royalty_bottom(b, DEFAULT_ROYALTIES)
            )
            if (
                best_valid is None
                or roy > best_valid_royalty
                or (roy == best_valid_royalty and (t, m, b) > best_valid)
            ):
                best_valid = (t, m, b)
                best_valid_royalty = roy
        elif best_valid is None:
            if best_invalid is None or (t, m, b) > best_invalid:
                best_invalid = (t, m, b)

    if best_valid is not None:
        return (best_valid[0], best_valid[1], best_valid[2], False)
    assert best_invalid is not None
    return (best_invalid[0], best_invalid[1], best_invalid[2], True)


def is_valid(
    top: Sequence[int],
    middle: Sequence[int],
    bottom: Sequence[int],
) -> bool:
    """True iff some joker assignment yields ``bottom >= middle >= top``.

    For boards without jokers this is equivalent to the row-by-row
    inequality. For boards with jokers, it asks whether *any* joker
    substitution produces a non-fouled layout — which is what the player
    is allowed to choose at scoring time.
    """
    if len(top) != 3 or len(middle) != 5 or len(bottom) != 5:
        raise ValueError("rows must be 3/5/5")
    return not resolve_board(tuple(top), tuple(middle), tuple(bottom))[3]


def is_foul(
    top: Sequence[int],
    middle: Sequence[int],
    bottom: Sequence[int],
) -> bool:
    return not is_valid(top, middle, bottom)


__all__ = ["evaluate_rows", "resolve_board", "is_valid", "is_foul"]
