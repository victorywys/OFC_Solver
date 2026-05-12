"""Fantasy EV table — per-tier value & continuation calibration.

Records, for each starting fantasy tier, statistics needed to set the
`continue_f14/15/16/17` values in `FantasyConfig`:

    * sum of immediate score (this hand) when player started in tier T
    * count of games started in tier T
    * count of games where player ENTERED any fantasy next hand
    * count of games where player ENDED in same-or-higher tier
      (continuation in the strict sense)

From these we derive:

    p_continue(T)   = P(next-hand tier > NORMAL | start tier = T)
    immediate_R(T)  = mean total_a from tier T
    continue_bonus(T) ≈ V(T) - V(NORMAL)

where V(T) is the long-run value of being in tier T (solved as the
fixed point of a small Markov chain over tiers).

The `result()` returns a `FantasyEVTable` with both the raw counts and
the derived continuation bonuses, so callers can either feed them into
`fantasy.fantasy_search.FantasyConfig` or further analyze.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine.fantasy import FantasyTier
from simulation.collectors import Collector
from simulation.trace import GameRecord


# All tiers we care about (NORMAL + every fantasy tier).
ALL_TIERS = (
    FantasyTier.NORMAL,
    FantasyTier.F14,
    FantasyTier.F15,
    FantasyTier.F16,
    FantasyTier.F17,
)


@dataclass
class TierStats:
    """Per-tier accumulators."""

    n_games: int = 0
    sum_score: float = 0.0          # signed score, this player's perspective
    n_to_normal: int = 0            # next-hand tier == NORMAL
    n_to_fantasy: int = 0           # next-hand tier > NORMAL
    # transition counts: dest_tier -> count
    transitions: dict[int, int] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        return self.sum_score / self.n_games if self.n_games else 0.0

    @property
    def p_continue(self) -> float:
        return self.n_to_fantasy / self.n_games if self.n_games else 0.0


@dataclass
class FantasyEVTable:
    """Per-tier EV statistics + derived continuation bonuses."""

    stats: dict[int, TierStats]   # tier_int -> TierStats

    # ---------- queries ----------
    def for_tier(self, tier: FantasyTier) -> TierStats:
        return self.stats.get(int(tier), TierStats())

    # ---------- derivations ----------
    def transition_matrix(self) -> dict[int, dict[int, float]]:
        """P(start_tier -> end_tier). Empty rows fall back to identity."""
        out: dict[int, dict[int, float]] = {}
        for src, st in self.stats.items():
            if st.n_games == 0:
                out[src] = {src: 1.0}
                continue
            out[src] = {
                dst: cnt / st.n_games for dst, cnt in st.transitions.items()
            }
        return out

    def value_function(self, tol: float = 1e-6, max_iter: int = 1000) -> dict[int, float]:
        """Solve V(T) = R(T) + sum_T' P(T -> T') * V(T') by iteration.

        With T == NORMAL serving as the discount anchor (V(NORMAL) is
        purely the expected per-hand score from NORMAL; from there a
        player can re-enter fantasy with some probability). All values
        are *per-hand* expected scores, so no discount factor is needed.

        Notes
        -----
        Without a discount factor, V is well-defined only because the
        Markov chain is recurrent and each hand has finite reward. We
        anchor by subtracting V(NORMAL) from every V at the end so that
        V(NORMAL) == 0 and the differences `V(T) - V(NORMAL)` give the
        per-hand 'fantasy bonus' for being in tier T.
        """
        tiers = list(self.stats.keys())
        if not tiers:
            return {}
        # rewards
        R = {t: self.stats[t].mean_score for t in tiers}
        P = self.transition_matrix()
        V = {t: 0.0 for t in tiers}
        for _ in range(max_iter):
            new_V: dict[int, float] = {}
            for t in tiers:
                v = R[t]
                for dst, p in P[t].items():
                    if dst in V:
                        v += p * V[dst]
                new_V[t] = v
            # anchor: subtract V(NORMAL) so it doesn't blow up
            anchor = new_V.get(int(FantasyTier.NORMAL), 0.0)
            for t in tiers:
                new_V[t] -= anchor
            # convergence
            diff = max(abs(new_V[t] - V[t]) for t in tiers)
            V = new_V
            if diff < tol:
                break
        return V

    def continue_bonuses(self) -> dict[int, float]:
        """`continue_bonus(T) = V(T) - V(NORMAL)` for each fantasy tier.

        Use these as the `continue_fXX` values in `FantasyConfig`.
        Falls back to 0.0 for tiers with no observations.
        """
        V = self.value_function()
        out: dict[int, float] = {}
        for t in (FantasyTier.F14, FantasyTier.F15, FantasyTier.F16, FantasyTier.F17):
            out[int(t)] = V.get(int(t), 0.0)
        return out

    def horizon_value_relative(self, horizon: int) -> dict[int, float]:
        """Per-tier `V_H(T) - V_H(NORMAL)` over `horizon` future hands.

        `V_H(T)` is the expected total signed score over the next `H`
        hands when the player STARTS the next hand in tier `T`. Anchored
        by subtracting `V_H(NORMAL)` so the returned numbers represent
        the *bonus* of carrying tier `T` into the next hand vs being
        NORMAL. `horizon == 0` yields all-zero (no future hands).

        `horizon == -1` means "infinite horizon" and returns the
        converged fixed-point bonuses from `value_function()` directly
        (per-hand bonuses, anchored to V(NORMAL) == 0).

        For finite `horizon` large enough the iteration converges to
        the same limit.
        """
        tiers = list(self.stats.keys())
        if not tiers or horizon == 0:
            return {t: 0.0 for t in tiers}
        if horizon < 0:
            # Infinite-horizon limit: per-hand value function bonuses.
            return self.value_function()
        R = {t: self.stats[t].mean_score for t in tiers}
        P = self.transition_matrix()
        V = {t: 0.0 for t in tiers}
        for _ in range(horizon):
            new_V: dict[int, float] = {}
            for t in tiers:
                v = R[t]
                for dst, p in P[t].items():
                    if dst in V:
                        v += p * V[dst]
                new_V[t] = v
            V = new_V
        anchor = V.get(int(FantasyTier.NORMAL), 0.0)
        return {t: V[t] - anchor for t in tiers}

    def __repr__(self) -> str:
        head = ", ".join(
            f"{FantasyTier(t).name}: n={self.stats[t].n_games}"
            for t in sorted(self.stats)
        )
        return f"FantasyEVTable({head})"


class FantasyEVCollector(Collector):
    """Per-tier accumulator. Observes summary GameRecords (no trace needed)."""

    name = "fantasy_ev"
    needs_full_trace = False

    def __init__(self) -> None:
        self.stats: dict[int, TierStats] = {int(t): TierStats() for t in ALL_TIERS}

    def observe(self, rec: GameRecord) -> None:
        for player in (0, 1):
            src = rec.initial_tiers[player]
            dst = rec.final_tiers[player]
            score = float(
                rec.score.total_a if player == 0 else -rec.score.total_a
            )
            st = self.stats.get(src)
            if st is None:
                st = TierStats()
                self.stats[src] = st
            st.n_games += 1
            st.sum_score += score
            if dst == int(FantasyTier.NORMAL):
                st.n_to_normal += 1
            else:
                st.n_to_fantasy += 1
            st.transitions[dst] = st.transitions.get(dst, 0) + 1

    def merge(self, other: "FantasyEVCollector") -> None:
        if type(other) is not FantasyEVCollector:
            raise TypeError(f"cannot merge with {type(other).__name__}")
        for tier, ost in other.stats.items():
            st = self.stats.get(tier)
            if st is None:
                st = TierStats()
                self.stats[tier] = st
            st.n_games += ost.n_games
            st.sum_score += ost.sum_score
            st.n_to_normal += ost.n_to_normal
            st.n_to_fantasy += ost.n_to_fantasy
            for dst, cnt in ost.transitions.items():
                st.transitions[dst] = st.transitions.get(dst, 0) + cnt

    def result(self) -> FantasyEVTable:
        # deep copy to detach from collector mutability
        copied = {
            tier: TierStats(
                n_games=st.n_games,
                sum_score=st.sum_score,
                n_to_normal=st.n_to_normal,
                n_to_fantasy=st.n_to_fantasy,
                transitions=dict(st.transitions),
            )
            for tier, st in self.stats.items()
        }
        return FantasyEVTable(stats=copied)


__all__ = [
    "FantasyEVTable",
    "FantasyEVCollector",
    "TierStats",
    "ALL_TIERS",
]
