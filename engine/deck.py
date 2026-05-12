"""Deck: deterministic, seedable shuffler with cheap deal/return primitives."""

from __future__ import annotations

import random
from typing import Iterable

from .cards import NUM_CARDS, full_deck


class Deck:
    """Mutable deck modeled as a stack. `deal(n)` pops n cards from the top.

    Designed for fast simulation:
        - O(1) deal of one card (pop from end)
        - O(n) deal of n cards
        - reset() reuses the same list (no realloc)
    """

    __slots__ = ("_cards", "_rng")

    def __init__(self, seed: int | None = None, include_jokers: bool = True) -> None:
        self._rng = random.Random(seed)
        if include_jokers:
            self._cards = full_deck()
        else:
            self._cards = list(range(NUM_CARDS - 2))
        self._rng.shuffle(self._cards)

    def __len__(self) -> int:
        return len(self._cards)

    def remaining(self) -> int:
        return len(self._cards)

    def deal(self, n: int = 1) -> list[int]:
        if n > len(self._cards):
            raise ValueError(f"deck has {len(self._cards)} cards, asked {n}")
        out = self._cards[-n:]
        del self._cards[-n:]
        out.reverse()  # so the "first dealt" card is index 0
        return out

    def deal_one(self) -> int:
        return self._cards.pop()

    def remove(self, cards: Iterable[int]) -> None:
        """Remove the given cards from the deck (e.g. cards already dealt)."""
        s = set(cards)
        self._cards = [c for c in self._cards if c not in s]

    def reset(self, seed: int | None = None, include_jokers: bool = True) -> None:
        if seed is not None:
            self._rng.seed(seed)
        if include_jokers:
            self._cards = full_deck()
        else:
            self._cards = list(range(NUM_CARDS - 2))
        self._rng.shuffle(self._cards)

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)

    def cards(self) -> list[int]:
        """Read-only view of remaining cards (not a defensive copy)."""
        return self._cards


def remaining_after(known: Iterable[int], include_jokers: bool = True) -> list[int]:
    """Helper: cards remaining if `known` are removed from a fresh deck.

    Useful for rollouts that sample from a known game state.
    """
    seen = set(known)
    base = list(range(NUM_CARDS)) if include_jokers else list(range(NUM_CARDS - 2))
    return [c for c in base if c not in seen]


__all__ = ["Deck", "remaining_after"]
