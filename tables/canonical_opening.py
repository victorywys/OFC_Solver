"""Suit-symmetric canonical form for street-1 opening hands.

Quotient symmetries
-------------------
* ``S_4`` acting on the 4 suits (24 permutations). Acts trivially on jokers.
* ``S_2`` swapping the two jokers (they are game-equivalent).

A canonical form for a 5-card hand is the lexicographically smallest
``(card_id, card_id, ...)`` tuple obtained over all 24×2 = 48 group
elements. We track the permutation chosen so we can re-suit the
canonical *action* back to the player's real hand.

Key invariants
--------------
* ``canonicalize(hand)`` is stable: the same hand always maps to the
  same canonical form.
* Round-trip: for every ``(card, slot)`` in the canonical action,
  ``apply_suit_perm`` re-emits a placement whose card multiset matches
  the player's real ``hand``.
* No standard card ever becomes a joker (or vice-versa) under any group
  element — the orbits respect the standard/joker split. Joker count is
  preserved.

The public API is two functions and one table class:

    canonicalize(hand)               -> (canonical_hand_key, ctx)
    apply_inverse(action, ctx)       -> ActionSignature with real card-ids
    CanonicalOpeningBookTable        -> drop-in replacement for the duck-
                                        typed ``opening_book.lookup`` API

``ctx`` is a small opaque struct (``CanonContext``) the table carries
between canonicalize and inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Optional, Sequence

from engine.cards import JOKER_1, JOKER_2, NUM_STD_CARDS

# All 24 suit permutations, precomputed once.
_SUIT_PERMS: tuple[tuple[int, int, int, int], ...] = tuple(
    permutations(range(4))  # type: ignore[arg-type]
)


@dataclass(frozen=True)
class CanonContext:
    """Bookkeeping needed to invert canonicalization.

    Attributes
    ----------
    suit_perm
        The chosen S_4 permutation ``p`` such that for every standard
        card with real ``(rank, suit)``, the canonical card has
        ``(rank, p[suit])``.
    joker_remap
        Mapping from canonical joker id to real joker id. Always a
        permutation of ``(JOKER_1, JOKER_2)`` (or partial if only one
        joker is present). Empty when no jokers.
    """

    suit_perm: tuple[int, int, int, int]
    joker_remap: tuple[tuple[int, int], ...]  # ((canon_jok, real_jok), ...)


# ---------------------------------------------------------------------------
# canonicalize
# ---------------------------------------------------------------------------
def _apply_perm(card_ids: Sequence[int], perm: Sequence[int]) -> tuple[int, ...]:
    """Apply suit permutation ``perm`` to a set of card ids.

    Standard cards: ``new_id = rank * 4 + perm[old_suit]``.
    Jokers: pass through unchanged (suit perm doesn't touch them).
    """
    out: list[int] = []
    for c in card_ids:
        if c >= NUM_STD_CARDS:
            out.append(c)
        else:
            rank = c >> 2
            suit = c & 3
            out.append(rank * 4 + perm[suit])
    return tuple(out)


def canonicalize(hand: Sequence[int]) -> tuple[tuple[int, ...], CanonContext]:
    """Return (canonical_sorted_hand, ctx) for a 5-card street-1 hand.

    Both jokers in the input are canonicalized to JOKER_1 first
    (then JOKER_2 if a second is present). Any "JOKER_2 only" hand
    becomes a "JOKER_1 only" hand under the S_2 joker swap. The chosen
    joker remap is captured in ``ctx.joker_remap`` so inversion can put
    the real joker id back into the action.

    Parameters
    ----------
    hand
        Iterable of 5 card ids (any order).

    Returns
    -------
    canonical_sorted_hand
        ``tuple(sorted(canonical_form))`` — the lookup key.
    ctx
        ``CanonContext`` carrying the chosen suit perm + joker remap.
    """
    # 1. Pull real jokers aside.
    std: list[int] = []
    real_jokers: list[int] = []
    for c in hand:
        (real_jokers if c >= NUM_STD_CARDS else std).append(c)
    real_jokers.sort()  # deterministic order

    # 2. Map real jokers -> canonical (JOKER_1, JOKER_2 in that order).
    canon_jokers_template = [JOKER_1, JOKER_2][: len(real_jokers)]
    joker_remap = tuple(
        (canon, real) for canon, real in zip(canon_jokers_template, real_jokers)
    )

    # 3. Try every suit perm on the standard portion; pick lex-min full hand.
    best_sorted: Optional[tuple[int, ...]] = None
    best_perm: Optional[tuple[int, int, int, int]] = None
    for p in _SUIT_PERMS:
        relabeled_std = _apply_perm(std, p)
        full = tuple(sorted(relabeled_std + tuple(canon_jokers_template)))
        if best_sorted is None or full < best_sorted:
            best_sorted = full
            best_perm = p
    assert best_sorted is not None and best_perm is not None

    return best_sorted, CanonContext(suit_perm=best_perm, joker_remap=joker_remap)


# ---------------------------------------------------------------------------
# inverse
# ---------------------------------------------------------------------------
def _inverse_perm(p: Sequence[int]) -> tuple[int, int, int, int]:
    inv = [0, 0, 0, 0]
    for i, v in enumerate(p):
        inv[v] = i
    return tuple(inv)  # type: ignore[return-value]


def apply_inverse(
    action_sig: tuple[tuple[int, int], ...],
    ctx: CanonContext,
) -> tuple[tuple[int, int], ...]:
    """Re-suit a canonical action back to a real-hand action.

    Parameters
    ----------
    action_sig
        A canonical ``ActionSignature`` (sorted tuple of ``(card_id, slot)``).
    ctx
        Context returned by :func:`canonicalize`.

    Returns
    -------
    A new ``ActionSignature`` whose card ids match the player's actual
    hand. The result is canonical-sorted to remain a valid
    ``ActionSignature``.
    """
    inv = _inverse_perm(ctx.suit_perm)
    canon_to_real = dict(ctx.joker_remap)
    out: list[tuple[int, int]] = []
    for c, s in action_sig:
        if c >= NUM_STD_CARDS:
            real = canon_to_real.get(c, c)
            out.append((real, s))
        else:
            rank = c >> 2
            suit = c & 3
            out.append((rank * 4 + inv[suit], s))
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Lookup table
# ---------------------------------------------------------------------------
class CanonicalOpeningBookTable:
    """A fully-precomputed opening book keyed by canonical hands.

    The on-disk representation is a plain ``dict``:

        canonical_sorted_hand : tuple[int, ...]   ->   action_sig

    where ``action_sig`` is a sorted tuple of ``(card_id, slot)`` pairs
    using the *canonical* card ids. Lookup canonicalizes the live hand,
    fetches the canonical action, and re-suits it back.

    The class is API-compatible with :class:`tables.opening_book.OpeningBookTable`
    so ``TableAwarePolicy.opening_book`` can be swapped in directly. The
    ``min_visits`` argument is accepted and ignored — each canonical
    entry is authoritative by construction.
    """

    def __init__(
        self,
        entries: dict[tuple[int, ...], tuple[tuple[int, int], ...]],
    ) -> None:
        self.entries = entries

    def lookup(
        self,
        hand: Sequence[int],
        min_visits: int = 0,  # accepted, ignored
    ) -> Optional[tuple[tuple[int, int], ...]]:
        if len(hand) != 5:
            return None
        canon_key, ctx = canonicalize(hand)
        canon_action = self.entries.get(canon_key)
        if canon_action is None:
            return None
        return apply_inverse(canon_action, ctx)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, hand: Sequence[int]) -> bool:
        canon_key, _ = canonicalize(hand)
        return canon_key in self.entries

    def __repr__(self) -> str:
        return f"CanonicalOpeningBookTable(entries={len(self.entries):,})"


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def enumerate_canonical_hands() -> list[tuple[int, ...]]:
    """All 152,646 canonical 5-card street-1 hands (jokers included).

    Each entry is the sorted tuple of canonical card ids — i.e., a key
    suitable for :attr:`CanonicalOpeningBookTable.entries`.
    """
    from itertools import combinations

    seen: set[tuple[int, ...]] = set()
    # 0, 1, or 2 jokers; remaining cards drawn from the 52 standards.
    for n_jok in (0, 1, 2):
        canon_jokers = tuple([JOKER_1, JOKER_2][:n_jok])
        for std in combinations(range(NUM_STD_CARDS), 5 - n_jok):
            key, _ = canonicalize(tuple(std) + canon_jokers)
            seen.add(key)
    return sorted(seen)


__all__ = [
    "CanonContext",
    "canonicalize",
    "apply_inverse",
    "CanonicalOpeningBookTable",
    "enumerate_canonical_hands",
]
