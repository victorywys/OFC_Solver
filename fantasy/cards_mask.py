"""Bitmask helpers for the fantasy solver.

A card mask is a Python int where bit `card_id` (0..53) indicates presence.
Python ints handle 54-bit masks natively at full speed and we get cheap
`&`, `|`, `~`, `popcount` operations.
"""

from __future__ import annotations

from typing import Iterable

from engine.cards import NUM_CARDS


def mask_of(cards: Iterable[int]) -> int:
    """Build a card mask from a card-id iterable."""
    m = 0
    for c in cards:
        m |= 1 << c
    return m


def cards_of(mask: int) -> list[int]:
    """Decompose a mask into card ids (ascending)."""
    out: list[int] = []
    while mask:
        b = mask & -mask
        out.append(b.bit_length() - 1)
        mask ^= b
    return out


def popcount(mask: int) -> int:
    return mask.bit_count()


def disjoint(a: int, b: int) -> bool:
    return (a & b) == 0


# All cards (54-bit). Useful when computing complements.
ALL_CARDS_MASK: int = (1 << NUM_CARDS) - 1


__all__ = ["mask_of", "cards_of", "popcount", "disjoint", "ALL_CARDS_MASK"]
