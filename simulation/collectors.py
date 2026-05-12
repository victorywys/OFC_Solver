"""Collectors: pluggable aggregators over `GameRecord` streams.

A `Collector` consumes `GameRecord`s one at a time via `observe()`, and
aggregates state internally. After all games are seen, `result()` returns
the produced table (typically a dict). For parallel self-play, each worker
constructs its own collectors, runs games, and then the main process
calls `merge()` to combine them.

Design contract
---------------
* Every collector MUST implement `observe`, `merge`, `result`.
* Collectors MUST be picklable (no lambdas, no open files in instance state).
* `merge(other)` MUST be commutative and produce a state equivalent to
  observing the union of game streams.
* `result()` SHOULD return a fresh dict / dataclass / mapping; do not
  return internal mutable state.
* `needs_full_trace` (class attribute) tells `SelfPlay` whether to record
  per-turn data. If any collector has `needs_full_trace=True`, the runner
  flips trace recording on for every game. Default False (cheap).

This module provides four baseline collectors that exercise both the
summary-only and full-trace paths. Phase 6 will add many more.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from engine.evaluator import (
    CATEGORY_NAMES,
    HIGH_CARD,
    PAIR,
    ROYAL_FLUSH,
    TRIPS,
    evaluate_3,
    evaluate_5,
)
from engine.fantasy import FantasyTier
from engine.royalties import (
    DEFAULT_ROYALTIES,
    RoyaltyConfig,
    royalty_bottom,
    royalty_middle,
    royalty_top,
)

from .trace import GameRecord


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Collector(ABC):
    """Pluggable aggregator over a stream of `GameRecord`s."""

    name: str = "collector"
    needs_full_trace: bool = False  # if True, per-turn data is recorded

    @abstractmethod
    def observe(self, rec: GameRecord) -> None:
        """Process one game record."""

    @abstractmethod
    def merge(self, other: "Collector") -> None:
        """Merge another collector of the same class into self (commutative)."""

    @abstractmethod
    def result(self) -> Any:
        """Return the produced table. Should be a fresh, picklable object."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def _check_same_class(self: Collector, other: Collector) -> None:
    if type(self) is not type(other):
        raise TypeError(
            f"cannot merge {type(self).__name__} with {type(other).__name__}"
        )


# ---------------------------------------------------------------------------
# 1. MatchSummaryCollector — always-on cheap summary
# ---------------------------------------------------------------------------
@dataclass
class MatchSummary:
    """Aggregate match metrics from P0's perspective."""

    n_games: int
    sum_total_a: int
    p0_wins: int
    p1_wins: int
    ties: int
    p0_fouls: int
    p1_fouls: int
    p0_scoops: int          # # of games P0 scooped (won all 3 rows, no foul)
    p1_scoops: int
    p0_royalties_total: int
    p1_royalties_total: int
    p0_fantasy_starts: int  # games where P0 started in a fantasy tier
    p1_fantasy_starts: int
    p0_fantasy_enters: int  # games where P0 ended with next-tier > NORMAL
    p1_fantasy_enters: int
    p0_fantasy_continues: int  # started fantasy AND ends with next-tier > NORMAL
    p1_fantasy_continues: int

    @property
    def avg_total_a(self) -> float:
        return self.sum_total_a / self.n_games if self.n_games else 0.0

    @property
    def p0_foul_rate(self) -> float:
        return self.p0_fouls / self.n_games if self.n_games else 0.0

    @property
    def p1_foul_rate(self) -> float:
        return self.p1_fouls / self.n_games if self.n_games else 0.0


class MatchSummaryCollector(Collector):
    name = "match_summary"
    needs_full_trace = False

    def __init__(self) -> None:
        self.n_games = 0
        self.sum_total_a = 0
        self.p0_wins = 0
        self.p1_wins = 0
        self.ties = 0
        self.p0_fouls = 0
        self.p1_fouls = 0
        self.p0_scoops = 0
        self.p1_scoops = 0
        self.p0_royalties_total = 0
        self.p1_royalties_total = 0
        self.p0_fantasy_starts = 0
        self.p1_fantasy_starts = 0
        self.p0_fantasy_enters = 0
        self.p1_fantasy_enters = 0
        self.p0_fantasy_continues = 0
        self.p1_fantasy_continues = 0

    def observe(self, rec: GameRecord) -> None:
        self.n_games += 1
        ta = rec.total_a
        self.sum_total_a += ta
        if ta > 0:
            self.p0_wins += 1
        elif ta < 0:
            self.p1_wins += 1
        else:
            self.ties += 1

        sb = rec.score
        if sb.a_foul:
            self.p0_fouls += 1
        if sb.b_foul:
            self.p1_fouls += 1

        # Scoop = +3 line + +3 scoop bonus (per spec, treated as 3+3 = 6 swing).
        if sb.scoop_bonus_a > 0 and not sb.a_foul:
            self.p0_scoops += 1
        elif sb.scoop_bonus_a < 0 and not sb.b_foul:
            self.p1_scoops += 1

        self.p0_royalties_total += sb.a_royalties
        self.p1_royalties_total += sb.b_royalties

        if rec.initial_tiers[0] != int(FantasyTier.NORMAL):
            self.p0_fantasy_starts += 1
        if rec.initial_tiers[1] != int(FantasyTier.NORMAL):
            self.p1_fantasy_starts += 1

        if rec.final_tiers[0] != int(FantasyTier.NORMAL):
            self.p0_fantasy_enters += 1
            if rec.initial_tiers[0] != int(FantasyTier.NORMAL):
                self.p0_fantasy_continues += 1
        if rec.final_tiers[1] != int(FantasyTier.NORMAL):
            self.p1_fantasy_enters += 1
            if rec.initial_tiers[1] != int(FantasyTier.NORMAL):
                self.p1_fantasy_continues += 1

    def merge(self, other: "MatchSummaryCollector") -> None:
        _check_same_class(self, other)
        self.n_games += other.n_games
        self.sum_total_a += other.sum_total_a
        self.p0_wins += other.p0_wins
        self.p1_wins += other.p1_wins
        self.ties += other.ties
        self.p0_fouls += other.p0_fouls
        self.p1_fouls += other.p1_fouls
        self.p0_scoops += other.p0_scoops
        self.p1_scoops += other.p1_scoops
        self.p0_royalties_total += other.p0_royalties_total
        self.p1_royalties_total += other.p1_royalties_total
        self.p0_fantasy_starts += other.p0_fantasy_starts
        self.p1_fantasy_starts += other.p1_fantasy_starts
        self.p0_fantasy_enters += other.p0_fantasy_enters
        self.p1_fantasy_enters += other.p1_fantasy_enters
        self.p0_fantasy_continues += other.p0_fantasy_continues
        self.p1_fantasy_continues += other.p1_fantasy_continues

    def result(self) -> MatchSummary:
        return MatchSummary(
            n_games=self.n_games,
            sum_total_a=self.sum_total_a,
            p0_wins=self.p0_wins,
            p1_wins=self.p1_wins,
            ties=self.ties,
            p0_fouls=self.p0_fouls,
            p1_fouls=self.p1_fouls,
            p0_scoops=self.p0_scoops,
            p1_scoops=self.p1_scoops,
            p0_royalties_total=self.p0_royalties_total,
            p1_royalties_total=self.p1_royalties_total,
            p0_fantasy_starts=self.p0_fantasy_starts,
            p1_fantasy_starts=self.p1_fantasy_starts,
            p0_fantasy_enters=self.p0_fantasy_enters,
            p1_fantasy_enters=self.p1_fantasy_enters,
            p0_fantasy_continues=self.p0_fantasy_continues,
            p1_fantasy_continues=self.p1_fantasy_continues,
        )


# ---------------------------------------------------------------------------
# 2. FoulByTierCollector — foul rate per starting fantasy tier
# ---------------------------------------------------------------------------
class FoulByTierCollector(Collector):
    name = "foul_by_tier"
    needs_full_trace = False

    def __init__(self) -> None:
        # tier_int -> [n_games, n_fouls]. Plain dict (defaultdict-with-lambda
        # is not picklable, which would break parallel self-play).
        self.counts: dict[int, list[int]] = {}

    def _cell(self, tier: int) -> list[int]:
        cell = self.counts.get(tier)
        if cell is None:
            cell = [0, 0]
            self.counts[tier] = cell
        return cell

    def observe(self, rec: GameRecord) -> None:
        for p in (0, 1):
            tier = rec.initial_tiers[p]
            cell = self._cell(tier)
            cell[0] += 1
            if rec.is_foul(p):
                cell[1] += 1

    def merge(self, other: "FoulByTierCollector") -> None:
        _check_same_class(self, other)
        for tier, (n, f) in other.counts.items():
            cell = self._cell(tier)
            cell[0] += n
            cell[1] += f

    def result(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for tier_int, (n, f) in self.counts.items():
            tier = FantasyTier(tier_int).name
            out[tier] = {
                "n_games": n,
                "n_fouls": f,
                "foul_rate": (f / n) if n else 0.0,
            }
        return out


# ---------------------------------------------------------------------------
# 3. RoyaltyByRowCollector — per-row royalty distribution
# ---------------------------------------------------------------------------
class RoyaltyByRowCollector(Collector):
    """Histogram of royalty values per row, only for non-foul boards.

    Useful for understanding: which rows contribute most royalties under a
    given policy, and what the empirical distribution looks like.
    """

    name = "royalty_by_row"
    needs_full_trace = False

    def __init__(self, royalty_cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> None:
        self.royalty_cfg = royalty_cfg
        self.top: Counter[int] = Counter()
        self.middle: Counter[int] = Counter()
        self.bottom: Counter[int] = Counter()
        self.n_boards = 0

    def _accumulate(self, board, foul: bool) -> None:
        if foul:
            return
        self.n_boards += 1
        t, m, b = board.evaluate()
        self.top[royalty_top(t, self.royalty_cfg)] += 1
        self.middle[royalty_middle(m, self.royalty_cfg)] += 1
        self.bottom[royalty_bottom(b, self.royalty_cfg)] += 1

    def observe(self, rec: GameRecord) -> None:
        self._accumulate(rec.final_a, rec.is_foul(0))
        self._accumulate(rec.final_b, rec.is_foul(1))

    def merge(self, other: "RoyaltyByRowCollector") -> None:
        _check_same_class(self, other)
        self.top.update(other.top)
        self.middle.update(other.middle)
        self.bottom.update(other.bottom)
        self.n_boards += other.n_boards

    def result(self) -> dict:
        return {
            "n_boards": self.n_boards,
            "top": dict(self.top),
            "middle": dict(self.middle),
            "bottom": dict(self.bottom),
        }


# ---------------------------------------------------------------------------
# 4. FantasyTransitionCollector — start-tier x end-tier transitions
# ---------------------------------------------------------------------------
class FantasyTransitionCollector(Collector):
    """Counts (starting tier) -> (next-hand tier) transitions.

    From this you can derive:
        * P(enter F14 | NORMAL)
        * P(continue Fxx | started Fxx)   [maintenance rate]
        * which end-tier is most common per start-tier
    """

    name = "fantasy_transitions"
    needs_full_trace = False

    def __init__(self) -> None:
        # (start_tier_int, end_tier_int) -> count
        self.counts: Counter[tuple[int, int]] = Counter()

    def observe(self, rec: GameRecord) -> None:
        for p in (0, 1):
            self.counts[(rec.initial_tiers[p], rec.final_tiers[p])] += 1

    def merge(self, other: "FantasyTransitionCollector") -> None:
        _check_same_class(self, other)
        self.counts.update(other.counts)

    def result(self) -> dict:
        # nested dict: {start_tier_name: {end_tier_name: count}}
        out: dict[str, dict[str, int]] = defaultdict(dict)
        for (s, e), n in self.counts.items():
            out[FantasyTier(s).name][FantasyTier(e).name] = n
        return {k: dict(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 5. TraceCollector — keeps every GameRecord in memory (heavy; trace-on)
# ---------------------------------------------------------------------------
class TraceCollector(Collector):
    """Retains every `GameRecord` for downstream analysis.

    Heavy: O(n_games * trace_size) memory. Recommended only for small
    samples (1k-10k games) used to bootstrap Phase-6 tables.
    """

    name = "trace"
    needs_full_trace = True

    def __init__(self) -> None:
        self.records: list[GameRecord] = []

    def observe(self, rec: GameRecord) -> None:
        self.records.append(rec)

    def merge(self, other: "TraceCollector") -> None:
        _check_same_class(self, other)
        self.records.extend(other.records)

    def result(self) -> list[GameRecord]:
        return list(self.records)


__all__ = [
    "Collector",
    "MatchSummary",
    "MatchSummaryCollector",
    "FoulByTierCollector",
    "RoyaltyByRowCollector",
    "FantasyTransitionCollector",
    "TraceCollector",
]
