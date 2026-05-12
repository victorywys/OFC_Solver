"""Fantasy arrangement cache.

Caches solver outputs for specific fantasy hands. Two complementary build
paths:

1. **Online (during self-play)**: `RecordingFantasyPolicy` wraps
   `FantasySolverPolicy`, intercepts every fantasy `act()` call, and stores
   the chosen `(top, middle, bottom, discards)` keyed by
   `(sorted_dealt_cards, tier)`. Plug it in as a player factory and a big
   self-play campaign builds the cache for free.

2. **Offline (from full traces)**: `FantasyArrangementCacheCollector` walks
   `Turn`s with `fantasy_tier != NORMAL` and records the final placements
   the policy chose. Use this if you self-played without the recording
   wrapper.

Lookup at runtime: convert dealt cards to the canonical signature, fetch
cached placements, return as an `Action`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.fantasy import FantasyTier
from simulation.collectors import Collector
from simulation.trace import GameRecord
from state.action import Action
from state.board import (
    SLOT_BOTTOM,
    SLOT_DISCARD,
    SLOT_MIDDLE,
    SLOT_TOP,
)
from state.game_state import GameState

from ai.policy import Policy

from .signatures import fantasy_hand_signature


@dataclass
class FantasyEntry:
    """One cached arrangement."""

    top: tuple[int, ...]
    middle: tuple[int, ...]
    bottom: tuple[int, ...]
    discards: tuple[int, ...]
    n_hits: int = 0       # how many times the cache served this entry

    def to_action(self, dealt_cards) -> Action:
        """Build the `Action` that lays this arrangement on an empty board."""
        placements = []
        for c in self.top:
            placements.append((c, SLOT_TOP))
        for c in self.middle:
            placements.append((c, SLOT_MIDDLE))
        for c in self.bottom:
            placements.append((c, SLOT_BOTTOM))
        for c in self.discards:
            placements.append((c, SLOT_DISCARD))
        # Validate that the action covers exactly the dealt cards.
        action_cs = sorted(c for c, _ in placements)
        dealt_cs = sorted(dealt_cards)
        if action_cs != dealt_cs:
            raise ValueError(
                "FantasyEntry.to_action: cached cards do not match dealt"
            )
        return Action(tuple(placements))


class FantasyArrangementCache:
    """Lookup table: (sorted_dealt_cards, tier) -> FantasyEntry."""

    def __init__(self, entries: Optional[dict[tuple, FantasyEntry]] = None) -> None:
        self.entries: dict[tuple, FantasyEntry] = entries or {}

    def lookup(self, cards, tier: FantasyTier) -> Optional[FantasyEntry]:
        sig = fantasy_hand_signature(cards, int(tier))
        entry = self.entries.get(sig)
        if entry is not None:
            entry.n_hits += 1
        return entry

    def insert(
        self,
        cards,
        tier: FantasyTier,
        top,
        middle,
        bottom,
        discards,
    ) -> None:
        sig = fantasy_hand_signature(cards, int(tier))
        if sig in self.entries:
            return
        self.entries[sig] = FantasyEntry(
            top=tuple(top),
            middle=tuple(middle),
            bottom=tuple(bottom),
            discards=tuple(discards),
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        hits = sum(e.n_hits for e in self.entries.values())
        return f"FantasyArrangementCache(entries={len(self)}, hits={hits})"


# ---------------------------------------------------------------------------
# Offline collector — extract entries from recorded trace
# ---------------------------------------------------------------------------
class FantasyArrangementCacheCollector(Collector):
    name = "fantasy_arrangement"
    needs_full_trace = True

    def __init__(self) -> None:
        self.entries: dict[tuple, FantasyEntry] = {}

    def observe(self, rec: GameRecord) -> None:
        for turn in rec.turns:
            if turn.fantasy_tier == int(FantasyTier.NORMAL):
                continue
            sig = fantasy_hand_signature(turn.pending, turn.fantasy_tier)
            if sig in self.entries:
                continue
            top: list[int] = []
            middle: list[int] = []
            bottom: list[int] = []
            discards: list[int] = []
            for c, s in turn.placements:
                if s == SLOT_TOP:
                    top.append(c)
                elif s == SLOT_MIDDLE:
                    middle.append(c)
                elif s == SLOT_BOTTOM:
                    bottom.append(c)
                elif s == SLOT_DISCARD:
                    discards.append(c)
            self.entries[sig] = FantasyEntry(
                top=tuple(top),
                middle=tuple(middle),
                bottom=tuple(bottom),
                discards=tuple(discards),
            )

    def merge(self, other: "FantasyArrangementCacheCollector") -> None:
        if type(other) is not FantasyArrangementCacheCollector:
            raise TypeError(f"cannot merge with {type(other).__name__}")
        for sig, entry in other.entries.items():
            if sig not in self.entries:
                self.entries[sig] = entry

    def result(self) -> FantasyArrangementCache:
        return FantasyArrangementCache(
            entries={
                k: FantasyEntry(
                    top=v.top, middle=v.middle, bottom=v.bottom, discards=v.discards
                )
                for k, v in self.entries.items()
            }
        )


# ---------------------------------------------------------------------------
# Online builder — wraps any fantasy-capable policy and records
# ---------------------------------------------------------------------------
class RecordingFantasyPolicy(Policy):
    """Drop-in replacement for any `Policy` that records its fantasy moves.

    On each `act()` call, delegates to `inner.act(...)`. If the player is
    in a fantasy tier, parses the chosen action's placements and inserts
    them into `cache`. This builds the arrangement cache as a side effect
    of any self-play campaign.

    Only useful within a single process (cache must be shared by reference).
    For multiprocessing builds, use `FantasyArrangementCacheCollector`.
    """

    name = "recording_fantasy"

    def __init__(self, inner: Policy, cache: FantasyArrangementCache) -> None:
        self.inner = inner
        self.cache = cache
        # propagate name for traceability
        self.name = getattr(inner, "name", "policy")

    def act(self, gs: GameState, player: int) -> Action:
        action = self.inner.act(gs, player)
        hs = gs.hands[player]
        if hs.fantasy_tier != FantasyTier.NORMAL:
            top: list[int] = []
            middle: list[int] = []
            bottom: list[int] = []
            discards: list[int] = []
            for c, s in action.placements:
                if s == SLOT_TOP:
                    top.append(c)
                elif s == SLOT_MIDDLE:
                    middle.append(c)
                elif s == SLOT_BOTTOM:
                    bottom.append(c)
                elif s == SLOT_DISCARD:
                    discards.append(c)
            self.cache.insert(
                hs.pending, hs.fantasy_tier, top, middle, bottom, discards
            )
        return action


__all__ = [
    "FantasyEntry",
    "FantasyArrangementCache",
    "FantasyArrangementCacheCollector",
    "RecordingFantasyPolicy",
]
