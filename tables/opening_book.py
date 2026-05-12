"""Opening book: street-1 dealt-hand → recommended action.

For every observed street-1 5-card deal, accumulate Welford EV per
canonical action. Lookup returns the best action by mean EV (with min
visit threshold).

Built from full-trace `GameRecord`s.

Note: street-1 is symmetric across players (5 cards in, 5 cards placed,
no discard, no prior board). So we aggregate observations from BOTH
players in the same hand — both sides are equally valid samples for the
same hand-type.
"""

from __future__ import annotations

from typing import Optional

from simulation.collectors import Collector
from simulation.trace import GameRecord

from .signatures import (
    ActionSignature,
    canonical_action,
    street1_hand_signature,
)
from .welford import Welford


HandKey = tuple[int, ...]


class OpeningBookTable:
    """Lookup table: street-1 hand -> action histogram with EV stats."""

    def __init__(
        self,
        entries: Optional[dict[HandKey, dict[ActionSignature, Welford]]] = None,
    ) -> None:
        self.entries: dict[HandKey, dict[ActionSignature, Welford]] = (
            entries or {}
        )

    def lookup(
        self,
        hand: HandKey,
        min_visits: int = 4,
    ) -> Optional[ActionSignature]:
        """Return best action by mean EV, or None if not enough data."""
        per_hand = self.entries.get(hand)
        if per_hand is None:
            return None
        best_a: Optional[ActionSignature] = None
        best_mean = float("-inf")
        for a, w in per_hand.items():
            if w.n < min_visits:
                continue
            if w.mean > best_mean:
                best_mean = w.mean
                best_a = a
        return best_a

    def stats(self, hand: HandKey) -> dict[ActionSignature, Welford]:
        return self.entries.get(hand, {})

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"OpeningBookTable(hands={len(self.entries)})"


class OpeningBookCollector(Collector):
    name = "opening_book"
    needs_full_trace = True

    def __init__(self) -> None:
        self.entries: dict[HandKey, dict[ActionSignature, Welford]] = {}

    def observe(self, rec: GameRecord) -> None:
        for turn in rec.turns:
            if turn.street != 1:
                continue
            if len(turn.pending) != 5:
                # Skip fantasy hands (which are also "street 1" but big).
                continue
            hand = street1_hand_signature(turn.pending)
            asig = canonical_action(turn.placements)
            outcome = float(
                rec.score.total_a if turn.player == 0 else -rec.score.total_a
            )
            per_hand = self.entries.get(hand)
            if per_hand is None:
                per_hand = {}
                self.entries[hand] = per_hand
            w = per_hand.get(asig)
            if w is None:
                w = Welford()
                per_hand[asig] = w
            w.push(outcome)

    def merge(self, other: "OpeningBookCollector") -> None:
        if type(other) is not OpeningBookCollector:
            raise TypeError(f"cannot merge with {type(other).__name__}")
        for hand, omap in other.entries.items():
            per_hand = self.entries.get(hand)
            if per_hand is None:
                self.entries[hand] = dict(omap)
                continue
            for asig, ow in omap.items():
                w = per_hand.get(asig)
                if w is None:
                    per_hand[asig] = ow
                else:
                    w.merge(ow)

    def result(self) -> OpeningBookTable:
        return OpeningBookTable(
            entries={k: dict(v) for k, v in self.entries.items()}
        )


__all__ = ["OpeningBookTable", "OpeningBookCollector"]
