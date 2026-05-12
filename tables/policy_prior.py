"""Policy prior / state-action EV table.

For every observed `(state_signature, action_signature)` pair from the
self-play trace, records a Welford accumulator over the eventual signed
score from that player's perspective.

Use cases:
    * MCTS prior / UCT bias: `mean(s, a)` favors actions historically
      successful in this state.
    * Greedy table-lookup policy: argmax over `mean(s, a)` for known
      states; fall back to base policy otherwise.

Built from full-trace `GameRecord`s.
"""

from __future__ import annotations

from typing import Optional

from simulation.collectors import Collector
from simulation.trace import GameRecord

from .signatures import (
    ActionSignature,
    StateSignature,
    canonical_action,
    turn_state_signature,
)
from .welford import Welford


class PolicyPriorTable:
    """Lookup table: (state_sig, action_sig) -> Welford."""

    def __init__(
        self,
        cells: Optional[dict[StateSignature, dict[ActionSignature, Welford]]] = None,
    ) -> None:
        self.cells: dict[StateSignature, dict[ActionSignature, Welford]] = (
            cells or {}
        )

    def actions(self, state_sig: StateSignature) -> dict[ActionSignature, Welford]:
        """Return the action -> Welford map for a state, empty if unseen."""
        return self.cells.get(state_sig, {})

    def lookup(
        self,
        state_sig: StateSignature,
        action_sig: ActionSignature,
    ) -> Optional[Welford]:
        per_state = self.cells.get(state_sig)
        if per_state is None:
            return None
        return per_state.get(action_sig)

    def best_action(
        self,
        state_sig: StateSignature,
        min_visits: int = 4,
    ) -> Optional[ActionSignature]:
        """Return the most-visited / highest-mean action sig.

        Selection criterion: among actions with `n >= min_visits`, return
        the one with the highest mean. If none qualify, return None so the
        caller can fall back to a base policy.
        """
        per_state = self.cells.get(state_sig)
        if per_state is None:
            return None
        best_a: Optional[ActionSignature] = None
        best_mean = float("-inf")
        for a, w in per_state.items():
            if w.n < min_visits:
                continue
            if w.mean > best_mean:
                best_mean = w.mean
                best_a = a
        return best_a

    def total_visits(self) -> int:
        return sum(w.n for st in self.cells.values() for w in st.values())

    def __len__(self) -> int:
        return len(self.cells)

    def __repr__(self) -> str:
        return (
            f"PolicyPriorTable(states={len(self.cells)}, "
            f"visits={self.total_visits()})"
        )


class PolicyPriorCollector(Collector):
    name = "policy_prior"
    needs_full_trace = True

    def __init__(self) -> None:
        self.cells: dict[StateSignature, dict[ActionSignature, Welford]] = {}

    def observe(self, rec: GameRecord) -> None:
        for turn in rec.turns:
            sig = turn_state_signature(turn)
            asig = canonical_action(turn.placements)
            outcome = float(
                rec.score.total_a if turn.player == 0 else -rec.score.total_a
            )
            per_state = self.cells.get(sig)
            if per_state is None:
                per_state = {}
                self.cells[sig] = per_state
            w = per_state.get(asig)
            if w is None:
                w = Welford()
                per_state[asig] = w
            w.push(outcome)

    def merge(self, other: "PolicyPriorCollector") -> None:
        if type(other) is not PolicyPriorCollector:
            raise TypeError(f"cannot merge with {type(other).__name__}")
        for sig, omap in other.cells.items():
            per_state = self.cells.get(sig)
            if per_state is None:
                # Copy by reference to avoid duplicating Welford instances;
                # safe because `other` is discarded by the caller after merge.
                self.cells[sig] = dict(omap)
                continue
            for asig, ow in omap.items():
                w = per_state.get(asig)
                if w is None:
                    per_state[asig] = ow
                else:
                    w.merge(ow)

    def result(self) -> PolicyPriorTable:
        return PolicyPriorTable(cells={k: dict(v) for k, v in self.cells.items()})


__all__ = ["PolicyPriorTable", "PolicyPriorCollector"]
