"""Action representation and legal-action enumeration.

An `Action` is an immutable assignment of a hand of cards to slots
(top/middle/bottom/discard).

Action shapes:
    - InitialAction: 5 cards, all placed (no discard).
    - PineappleAction: 3 cards in hand, 2 placed + 1 discarded.
    - FantasyAction: N cards in hand (14..17), 13 placed + (N-13) discarded.

All three are represented by the same frozen `Action` dataclass. The
generator functions enforce the appropriate shape.

Performance:
    - Round 1 has up to 232 enumerations (capacity-constrained 3^5).
    - Streets 2..5 have at most 27 enumerations (3 discard choices x 9
      placement assignments). The board's row capacities prune further.
    - Fantasy actions are enormous; we expose a generator and a budget cap.
      The heuristic policy (Phase 3) is responsible for sampling/pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Iterator

from .board import (
    PlayerBoard,
    ROW_CAPACITY,
    SLOT_BOTTOM,
    SLOT_DISCARD,
    SLOT_MIDDLE,
    SLOT_TOP,
    TOTAL_PLACED,
)


@dataclass(frozen=True)
class Action:
    """Immutable assignment of (card -> slot) decisions.

    `placements` is a tuple of (card_id, slot_id) pairs covering all cards
    that were dealt this street. For pineapple/fantasy, exactly the discarded
    cards are paired with SLOT_DISCARD.
    """

    placements: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        # Precompute the discarded-cards tuple once at construction so
        # `discards()` is a free attribute read. `score_action` is called
        # ~200x per `act()` and used to be the #1 hotspot. `frozen=True`
        # blocks normal assignment, so we use `object.__setattr__`.
        d: list[int] = []
        for c, s in self.placements:
            if s == SLOT_DISCARD:
                d.append(c)
        object.__setattr__(self, "_discards", tuple(d))

    def discards(self) -> tuple[int, ...]:
        return self._discards  # type: ignore[attr-defined]

    def placed_only(self) -> tuple[tuple[int, int], ...]:
        return tuple((c, s) for c, s in self.placements if s != SLOT_DISCARD)

    def apply(self, board: PlayerBoard) -> PlayerBoard:
        """Return a new board with this action applied (does not mutate)."""
        nb = board.clone()
        for card, slot in self.placements:
            nb.place(card, slot)
        return nb

    def apply_inplace(self, board: PlayerBoard) -> None:
        for card, slot in self.placements:
            board.place(card, slot)


# ---------------------------------------------------------------------------
# Initial street (round 1): place ALL 5 cards into rows.
# ---------------------------------------------------------------------------
def enumerate_initial_actions(cards: Iterable[int]) -> list[Action]:
    """All capacity-respecting assignments of 5 cards across the 3 rows.

    No discards. Caller may further prune (e.g. via heuristic).
    """
    cs = list(cards)
    if len(cs) != 5:
        raise ValueError("initial street requires exactly 5 cards")

    out: list[Action] = []
    cap = ROW_CAPACITY  # (3,5,5)
    # 3^5 = 243 assignments; capacity filter prunes to 232.
    for assign in product((SLOT_TOP, SLOT_MIDDLE, SLOT_BOTTOM), repeat=5):
        # capacity check
        a = b = c = 0
        for s in assign:
            if s == SLOT_TOP:
                a += 1
            elif s == SLOT_MIDDLE:
                b += 1
            else:
                c += 1
        if a > cap[SLOT_TOP] or b > cap[SLOT_MIDDLE] or c > cap[SLOT_BOTTOM]:
            continue
        out.append(Action(tuple(zip(cs, assign))))
    return out


# ---------------------------------------------------------------------------
# Pineapple street (rounds 2..5): receive 3 cards, place 2, discard 1.
# ---------------------------------------------------------------------------
def enumerate_pineapple_actions(
    cards: Iterable[int],
    board: PlayerBoard,
) -> list[Action]:
    """All legal (discard 1, place 2) actions for a 3-card pineapple street.

    Respects current row capacities on `board`.
    """
    cs = list(cards)
    if len(cs) != 3:
        raise ValueError("pineapple street requires exactly 3 cards")

    free = (board.free_top(), board.free_middle(), board.free_bottom())
    out: list[Action] = []

    # choose which of the 3 cards is discarded
    for i_disc in range(3):
        kept = [cs[j] for j in range(3) if j != i_disc]
        disc_card = cs[i_disc]
        # place the 2 kept cards into rows; check capacity per row across both
        for s0, s1 in product(
            (SLOT_TOP, SLOT_MIDDLE, SLOT_BOTTOM),
            repeat=2,
        ):
            need = [0, 0, 0]
            need[s0] += 1
            need[s1] += 1
            if need[0] > free[0] or need[1] > free[1] or need[2] > free[2]:
                continue
            placements = (
                (kept[0], s0),
                (kept[1], s1),
                (disc_card, SLOT_DISCARD),
            )
            out.append(Action(placements))
    return out


# ---------------------------------------------------------------------------
# Fantasyland: place 13 of N cards (N = 14..17). Vast action space.
# Provide a generator; callers cap the work via budget.
# ---------------------------------------------------------------------------
def iter_fantasy_actions(
    cards: Iterable[int],
    board: PlayerBoard | None = None,
    *,
    budget: int | None = None,
) -> Iterator[Action]:
    """Yield legal fantasy placements.

    Fantasy convention here: the player solves a fresh hand from a single big
    deal. The board passed in should be empty (we'll default-construct one).
    Each action lays out exactly 13 cards (3+5+5) and discards the rest.

    The cardinality is huge: for 14 cards we choose 1 to discard (14 options),
    then choose which 3 go to top (C(13,3)=286), then which 5 of remaining 10
    go to middle (C(10,5)=252). Total = 14 * 286 * 252 = ~1M assignments,
    most equivalent up to row order. Use `budget` to cap.

    NOTE: within a row, the order of cards does not matter for evaluation,
    so we generate canonical placements (cards sorted within each row).
    """
    cs = list(cards)
    n = len(cs)
    if n < TOTAL_PLACED:
        raise ValueError(f"fantasy requires >= {TOTAL_PLACED} cards, got {n}")
    if board is not None and board.total_placed() != 0:
        raise ValueError("fantasy assumes an empty starting board")
    discards_needed = n - TOTAL_PLACED

    # We iterate over (top set, middle set, bottom set) partitions of cs into
    # sizes (3, 5, 5) plus a discard set of size discards_needed.
    # To stay enumerable we iterate by choosing positions deterministically.
    from itertools import combinations

    indices = range(n)
    yielded = 0
    for top_idx in combinations(indices, 3):
        rest1 = [i for i in indices if i not in top_idx]
        for mid_idx in combinations(rest1, 5):
            rest2 = [i for i in rest1 if i not in mid_idx]
            for bot_idx in combinations(rest2, 5):
                disc_idx = [i for i in rest2 if i not in bot_idx]
                placements: list[tuple[int, int]] = []
                for i in top_idx:
                    placements.append((cs[i], SLOT_TOP))
                for i in mid_idx:
                    placements.append((cs[i], SLOT_MIDDLE))
                for i in bot_idx:
                    placements.append((cs[i], SLOT_BOTTOM))
                for i in disc_idx:
                    placements.append((cs[i], SLOT_DISCARD))
                yield Action(tuple(placements))
                yielded += 1
                if budget is not None and yielded >= budget:
                    return


__all__ = [
    "Action",
    "enumerate_initial_actions",
    "enumerate_pineapple_actions",
    "iter_fantasy_actions",
]
