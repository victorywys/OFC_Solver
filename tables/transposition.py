"""Transposition table — process-local runtime cache.

A small LRU-flavored dict mapping `state_signature -> chosen Action`.

Two ways to populate:
    * **Runtime memoization**: `TableAwarePolicy` looks up before
      decision-making and stores after. Same identical state never costs
      a re-search.
    * **Offline derivation**: `from_policy_prior(...)` projects a
      `PolicyPriorTable` into a transposition table by taking the best
      action per state with sufficient support.

Persistence: `save(path)` / `load(path)` use pickle.

Capacity: optional `max_entries`. When full, eviction is FIFO via a
doubly-linked-list-style approach (we just use `dict` insertion order,
which Python 3.7+ guarantees; eviction pops the oldest key).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from state.action import Action

from .signatures import StateSignature


class TranspositionTable:
    """state_signature -> Action."""

    def __init__(
        self,
        max_entries: Optional[int] = 200_000,
        entries: Optional[dict[StateSignature, Action]] = None,
    ) -> None:
        self.max_entries = max_entries
        self.entries: dict[StateSignature, Action] = entries or {}
        self._hits = 0
        self._misses = 0

    # ---------- core ----------
    def lookup(self, sig: StateSignature) -> Optional[Action]:
        a = self.entries.get(sig)
        if a is not None:
            self._hits += 1
        else:
            self._misses += 1
        return a

    def store(self, sig: StateSignature, action: Action) -> None:
        if sig in self.entries:
            return
        self.entries[sig] = action
        if self.max_entries is not None and len(self.entries) > self.max_entries:
            # evict oldest
            try:
                first = next(iter(self.entries))
                del self.entries[first]
            except StopIteration:
                pass

    # ---------- stats ----------
    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        denom = self._hits + self._misses
        return self._hits / denom if denom else 0.0

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (
            f"TranspositionTable(entries={len(self)}, "
            f"hits={self._hits}, misses={self._misses})"
        )

    # ---------- persistence ----------
    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(
                {
                    "max_entries": self.max_entries,
                    "entries": self.entries,
                },
                f,
                protocol=4,
            )

    @classmethod
    def load(cls, path: Path | str) -> "TranspositionTable":
        with Path(path).open("rb") as f:
            blob = pickle.load(f)
        return cls(
            max_entries=blob["max_entries"],
            entries=blob["entries"],
        )

    # ---------- derivation ----------
    @classmethod
    def from_policy_prior(
        cls,
        prior_table,
        min_visits: int = 8,
        max_entries: Optional[int] = 200_000,
    ) -> "TranspositionTable":
        """Project `PolicyPriorTable` -> `TranspositionTable` (best action / state).

        Only includes states with enough support (`min_visits`).
        """
        entries: dict[StateSignature, Action] = {}
        for sig, action_map in prior_table.cells.items():
            best_a = None
            best_mean = float("-inf")
            for asig, w in action_map.items():
                if w.n < min_visits:
                    continue
                if w.mean > best_mean:
                    best_mean = w.mean
                    best_a = asig
            if best_a is None:
                continue
            entries[sig] = Action(best_a)
            if max_entries is not None and len(entries) >= max_entries:
                break
        return cls(max_entries=max_entries, entries=entries)


__all__ = ["TranspositionTable"]
