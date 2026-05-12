"""Foul-probability table.

For every observed `(state_signature)` at a decision point, records:
    n_visits   : how many times we saw this state
    n_eventual_fouls : how many of those games ended with that player fouling

Lookup `P(foul | state)` is `n_eventual_fouls / n_visits`. A high value
warns the heuristic / MC policy to play conservatively. A low value
greenlights aggressive royalty hunting.

Practical caveat
----------------
State signatures are exact (sorted row tuples + discards + tier + street
+ pending). Hits on a previously unseen state will return the prior
default (0.0). Bucketing / generalization (e.g. via row-feature hashing)
is left for a future iteration.

Built from full-trace `GameRecord`s, so the collector requires
`needs_full_trace = True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from simulation.collectors import Collector
from simulation.trace import GameRecord

from .signatures import StateSignature, turn_state_signature


@dataclass
class _Cell:
    n: int = 0
    fouls: int = 0


class FoulProbTable:
    """Lookup table: state signature -> P(foul | state)."""

    def __init__(self, cells: Optional[dict[StateSignature, _Cell]] = None) -> None:
        self.cells: dict[StateSignature, _Cell] = cells or {}

    def lookup(self, sig: StateSignature, default: float = 0.0) -> float:
        cell = self.cells.get(sig)
        if cell is None or cell.n == 0:
            return default
        return cell.fouls / cell.n

    def support(self, sig: StateSignature) -> int:
        cell = self.cells.get(sig)
        return cell.n if cell else 0

    def __len__(self) -> int:
        return len(self.cells)

    def __repr__(self) -> str:
        return f"FoulProbTable(states={len(self.cells)})"


class FoulProbCollector(Collector):
    """Collector for `FoulProbTable`.

    For each turn in each game, increments the cell for that turn's
    `(state_signature)` and records whether the player who acted ended
    up fouling.
    """

    name = "foul_prob"
    needs_full_trace = True

    def __init__(self) -> None:
        self.cells: dict[StateSignature, _Cell] = {}

    def observe(self, rec: GameRecord) -> None:
        for turn in rec.turns:
            sig = turn_state_signature(turn)
            cell = self.cells.get(sig)
            if cell is None:
                cell = _Cell()
                self.cells[sig] = cell
            cell.n += 1
            if rec.is_foul(turn.player):
                cell.fouls += 1

    def merge(self, other: "FoulProbCollector") -> None:
        if type(other) is not FoulProbCollector:
            raise TypeError(f"cannot merge with {type(other).__name__}")
        for sig, ocell in other.cells.items():
            cell = self.cells.get(sig)
            if cell is None:
                self.cells[sig] = _Cell(ocell.n, ocell.fouls)
            else:
                cell.n += ocell.n
                cell.fouls += ocell.fouls

    def result(self) -> FoulProbTable:
        return FoulProbTable(cells=dict(self.cells))


__all__ = ["FoulProbTable", "FoulProbCollector"]
