"""Per-player in-progress board.

Capacity:
    TOP:    3 cards
    MIDDLE: 5 cards
    BOTTOM: 5 cards
Total 13. Plus discards (separate, not part of the final board).

`PlayerBoard` is the mutable working state during a hand. When all rows are
full, call `to_final_board()` to get the immutable `engine.scoring.Board`
used for evaluation/scoring.

Slots are encoded as small ints for speed:
    SLOT_TOP    = 0
    SLOT_MIDDLE = 1
    SLOT_BOTTOM = 2
    SLOT_DISCARD = 3
"""

from __future__ import annotations

from typing import Iterable

from engine.scoring import Board

SLOT_TOP = 0
SLOT_MIDDLE = 1
SLOT_BOTTOM = 2
SLOT_DISCARD = 3
N_PLACE_SLOTS = 3  # TOP, MIDDLE, BOTTOM are the placement slots

SLOT_NAMES = {SLOT_TOP: "T", SLOT_MIDDLE: "M", SLOT_BOTTOM: "B", SLOT_DISCARD: "X"}
SLOT_FROM_CHAR = {"T": SLOT_TOP, "M": SLOT_MIDDLE, "B": SLOT_BOTTOM, "X": SLOT_DISCARD}

ROW_CAPACITY = (3, 5, 5)
TOTAL_PLACED = 13


class PlayerBoard:
    """Mutable per-player board.

    Stores cards as plain lists (cheap to copy/clone). All operations are O(1)
    or O(row size). `clone()` deep-copies so simulation/MCTS can branch safely.
    """

    __slots__ = ("rows", "discards")

    def __init__(self) -> None:
        # rows[0]=top, rows[1]=middle, rows[2]=bottom
        self.rows: tuple[list[int], list[int], list[int]] = ([], [], [])
        self.discards: list[int] = []

    # ----- introspection -----
    @property
    def top(self) -> list[int]:
        return self.rows[SLOT_TOP]

    @property
    def middle(self) -> list[int]:
        return self.rows[SLOT_MIDDLE]

    @property
    def bottom(self) -> list[int]:
        return self.rows[SLOT_BOTTOM]

    def free_slots(self, row: int) -> int:
        return ROW_CAPACITY[row] - len(self.rows[row])

    def free_top(self) -> int:
        return ROW_CAPACITY[SLOT_TOP] - len(self.rows[SLOT_TOP])

    def free_middle(self) -> int:
        return ROW_CAPACITY[SLOT_MIDDLE] - len(self.rows[SLOT_MIDDLE])

    def free_bottom(self) -> int:
        return ROW_CAPACITY[SLOT_BOTTOM] - len(self.rows[SLOT_BOTTOM])

    def total_placed(self) -> int:
        return len(self.top) + len(self.middle) + len(self.bottom)

    def is_full(self) -> bool:
        return self.total_placed() == TOTAL_PLACED

    def all_cards(self) -> list[int]:
        """All cards on the board (placed + discarded)."""
        out: list[int] = []
        out.extend(self.rows[0])
        out.extend(self.rows[1])
        out.extend(self.rows[2])
        out.extend(self.discards)
        return out

    # ----- mutation -----
    def place(self, card: int, slot: int) -> None:
        if slot == SLOT_DISCARD:
            self.discards.append(card)
            return
        if slot < 0 or slot > SLOT_BOTTOM:
            raise ValueError(f"invalid placement slot: {slot}")
        if len(self.rows[slot]) >= ROW_CAPACITY[slot]:
            raise ValueError(f"row {SLOT_NAMES[slot]} is full")
        self.rows[slot].append(card)

    def place_many(self, items: Iterable[tuple[int, int]]) -> None:
        for card, slot in items:
            self.place(card, slot)

    def clone(self) -> "PlayerBoard":
        new = PlayerBoard.__new__(PlayerBoard)
        new.rows = ([*self.rows[0]], [*self.rows[1]], [*self.rows[2]])
        new.discards = [*self.discards]
        return new

    # ----- finalization -----
    def to_final_board(self) -> Board:
        if not self.is_full():
            raise ValueError(
                f"board not full: {self.total_placed()}/{TOTAL_PLACED} placed"
            )
        return Board(tuple(self.top), tuple(self.middle), tuple(self.bottom))

    # ----- pretty -----
    def __repr__(self) -> str:
        from engine.cards import cards_str

        return (
            f"PlayerBoard(T=[{cards_str(self.top)}], "
            f"M=[{cards_str(self.middle)}], "
            f"B=[{cards_str(self.bottom)}], "
            f"X=[{cards_str(self.discards)}])"
        )


__all__ = [
    "PlayerBoard",
    "SLOT_TOP", "SLOT_MIDDLE", "SLOT_BOTTOM", "SLOT_DISCARD",
    "SLOT_NAMES", "SLOT_FROM_CHAR",
    "ROW_CAPACITY", "TOTAL_PLACED", "N_PLACE_SLOTS",
]
