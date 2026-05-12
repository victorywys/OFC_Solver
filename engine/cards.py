"""Card representation.

Cards are encoded as small ints in [0, 54):
    standard cards: card_id = rank * 4 + suit
        rank in [0, 13)   (0=2, 1=3, ..., 12=A)
        suit in [0, 4)    (0=c, 1=d, 2=h, 3=s)
    jokers/wildcards:
        JOKER_1 = 52
        JOKER_2 = 53

Why integer encoding?
    - O(1) extraction of rank/suit
    - cheap to put into sets / numpy arrays
    - easy bitmask construction in the evaluator
    - friendly for future C / numba acceleration
"""

from __future__ import annotations

from typing import Iterable

NUM_RANKS = 13
NUM_SUITS = 4
NUM_STD_CARDS = NUM_RANKS * NUM_SUITS  # 52
NUM_JOKERS = 2
NUM_CARDS = NUM_STD_CARDS + NUM_JOKERS  # 54

JOKER_1 = 52
JOKER_2 = 53
JOKERS = (JOKER_1, JOKER_2)

# Rank constants (0=2, ..., 12=A)
RANK_2, RANK_3, RANK_4, RANK_5, RANK_6 = 0, 1, 2, 3, 4
RANK_7, RANK_8, RANK_9, RANK_T, RANK_J = 5, 6, 7, 8, 9
RANK_Q, RANK_K, RANK_A = 10, 11, 12

# Suit constants
SUIT_C, SUIT_D, SUIT_H, SUIT_S = 0, 1, 2, 3

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

_RANK_FROM_CHAR = {c: i for i, c in enumerate(RANK_CHARS)}
_SUIT_FROM_CHAR = {c: i for i, c in enumerate(SUIT_CHARS)}


def make_card(rank: int, suit: int) -> int:
    """Pack a rank/suit pair into a card id."""
    if not (0 <= rank < NUM_RANKS):
        raise ValueError(f"rank out of range: {rank}")
    if not (0 <= suit < NUM_SUITS):
        raise ValueError(f"suit out of range: {suit}")
    return rank * NUM_SUITS + suit


def card_rank(card: int) -> int:
    """Rank of a non-joker card. Returns -1 for jokers."""
    if card >= NUM_STD_CARDS:
        return -1
    return card >> 2  # card // 4


def card_suit(card: int) -> int:
    """Suit of a non-joker card. Returns -1 for jokers."""
    if card >= NUM_STD_CARDS:
        return -1
    return card & 3  # card % 4


def is_joker(card: int) -> bool:
    return card >= NUM_STD_CARDS


def parse_card(s: str) -> int:
    """Parse "As", "Td", "*1", "*2" (jokers) into a card id.

    Joker forms accepted: "*1", "*2", "Jk1", "Jk2", "*", "Xx".
    """
    s = s.strip()
    if s in ("*1", "Jk1", "JK1"):
        return JOKER_1
    if s in ("*2", "Jk2", "JK2"):
        return JOKER_2
    if s in ("*", "Xx", "??"):  # generic wildcard; map to first free joker
        return JOKER_1
    if len(s) != 2:
        raise ValueError(f"invalid card string: {s!r}")
    r, u = s[0].upper(), s[1].lower()
    if r not in _RANK_FROM_CHAR:
        raise ValueError(f"invalid rank in {s!r}")
    if u not in _SUIT_FROM_CHAR:
        raise ValueError(f"invalid suit in {s!r}")
    return make_card(_RANK_FROM_CHAR[r], _SUIT_FROM_CHAR[u])


def parse_cards(seq: str | Iterable[str]) -> list[int]:
    """Parse a whitespace-separated string or iterable of card strings."""
    if isinstance(seq, str):
        seq = seq.split()
    return [parse_card(tok) for tok in seq]


def card_str(card: int) -> str:
    """String representation of a card."""
    if card == JOKER_1:
        return "*1"
    if card == JOKER_2:
        return "*2"
    return RANK_CHARS[card_rank(card)] + SUIT_CHARS[card_suit(card)]


def cards_str(cards: Iterable[int]) -> str:
    return " ".join(card_str(c) for c in cards)


def full_deck() -> list[int]:
    """All 54 cards (including 2 jokers), unshuffled, deterministic order."""
    return list(range(NUM_CARDS))


__all__ = [
    "NUM_RANKS", "NUM_SUITS", "NUM_STD_CARDS", "NUM_JOKERS", "NUM_CARDS",
    "JOKER_1", "JOKER_2", "JOKERS",
    "RANK_2", "RANK_3", "RANK_4", "RANK_5", "RANK_6", "RANK_7", "RANK_8",
    "RANK_9", "RANK_T", "RANK_J", "RANK_Q", "RANK_K", "RANK_A",
    "SUIT_C", "SUIT_D", "SUIT_H", "SUIT_S",
    "RANK_CHARS", "SUIT_CHARS",
    "make_card", "card_rank", "card_suit", "is_joker",
    "parse_card", "parse_cards", "card_str", "cards_str", "full_deck",
]
