"""Bottom-first DFS for the fantasy solver.

Algorithm
---------
1. Precompute every C(N,5) and C(N,3) hand evaluation once. Both `bottom`
   and `middle` candidate enumeration share the 5-subset cache, eliminating
   ~99% of `evaluate_5` calls.
2. Enumerate all C(N, 5) bottom candidates from the dealt cards, ranked by
   `(continuation_guarantee, royalty, hand_strength)`. Keep top
   `bottom_beam` (or all in exact mode).
3. For each kept bottom, enumerate compatible middles (5 of remaining N-5),
   filter by `middle <= bottom`, sort by royalty (cheap int-key sort),
   keep top `middle_beam`.
4. For each (bottom, middle), enumerate all C(N-10, 3) tops; filter
   `top <= middle`. This is small (4..35) so we keep all.
5. Compute exact EV; track best layout.

Branch-and-bound
----------------
- After fixing a bottom, derive an admissible upper bound on the remaining
  EV (best possible middle + top royalty + max continuation bonus). If
  less than current best, skip this bottom.
- After fixing (bottom, middle), do the same for top.

The solver is correct in `exact=True` mode (no beams) and near-optimal in
beam mode; calibration is the user's job via `FantasyConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from engine.cards import cards_str
from engine.evaluator import HandRank, evaluate_3, evaluate_5
from engine.fantasy import FantasyTier
from engine.royalties import RoyaltyConfig

from .fantasy_eval import (
    bottom_guarantees_continuation,
    bottom_royalty,
    is_continuation,
    middle_royalty,
    top_royalty_value,
    upper_bound_middle_royalty,
    upper_bound_top_royalty,
)


# ---------------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FantasyConfig:
    """Configurable knobs for the fantasy solver."""

    royalty_cfg: RoyaltyConfig = field(default_factory=RoyaltyConfig)

    # weights (per-row royalty multiplier)
    w_top: float = 1.0
    w_middle: float = 1.0
    w_bottom: float = 1.0

    # continuation bonus per tier (placeholder values; calibrate via Phase 6)
    continue_f14: float = 20.0
    continue_f15: float = 30.0
    continue_f16: float = 45.0
    continue_f17: float = 70.0

    # search / pruning
    bottom_beam: int = 0    # 0 -> keep all (exact-on-bottom)
    middle_beam: int = 0    # 0 -> keep all
    use_branch_and_bound: bool = True

    # if True, override beams: search every layout. Slow for F17.
    exact: bool = False

    def continue_bonus(self, tier: FantasyTier) -> float:
        if tier == FantasyTier.F14:
            return self.continue_f14
        if tier == FantasyTier.F15:
            return self.continue_f15
        if tier == FantasyTier.F16:
            return self.continue_f16
        if tier == FantasyTier.F17:
            return self.continue_f17
        return 0.0


@dataclass
class SearchStats:
    """Diagnostics from a single solve."""

    bottoms_considered: int = 0
    bottoms_kept: int = 0
    middles_considered: int = 0
    middles_kept: int = 0
    tops_considered: int = 0
    leaves_evaluated: int = 0
    pruned_by_bound: int = 0
    pruned_foul: int = 0


@dataclass
class FantasyResult:
    top: tuple[int, ...]
    middle: tuple[int, ...]
    bottom: tuple[int, ...]
    discards: tuple[int, ...]
    ev: float
    immediate_royalties: int
    continuation: bool
    continuation_bonus: float
    top_rank: HandRank
    middle_rank: HandRank
    bottom_rank: HandRank
    stats: SearchStats

    def pretty(self) -> str:
        return (
            f"Fantasy result EV={self.ev:.2f}  cont={self.continuation}\n"
            f"  TOP : {cards_str(self.top)}\n"
            f"  MID : {cards_str(self.middle)}\n"
            f"  BOT : {cards_str(self.bottom)}\n"
            f"  DISC: {cards_str(self.discards)}\n"
            f"  royalties={self.immediate_royalties}  cont_bonus={self.continuation_bonus}"
        )


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------
def solve(
    cards: list[int],
    tier: FantasyTier,
    config: FantasyConfig,
) -> FantasyResult:
    """Solve a fantasy layout.

    Returns the best (top, middle, bottom, discard) layout maximizing
    `EV = sum(w_x * royalty_x) + (continue_bonus if continuation else 0)`.

    Foul layouts are never returned; the search filters them out.
    """
    n = len(cards)
    if n < 13:
        raise ValueError(f"fantasy needs >= 13 cards, got {n}")
    if tier == FantasyTier.NORMAL:
        raise ValueError("solve() called with NORMAL tier")

    cfg = config
    stats = SearchStats()

    # Pre-compute cheap bounds
    max_top_roy = upper_bound_top_royalty(cfg.royalty_cfg)
    max_mid_roy = upper_bound_middle_royalty(cfg.royalty_cfg)
    cont_bonus = cfg.continue_bonus(tier)
    cont_bonus_term = cont_bonus if cont_bonus > 0 else 0.0

    def _rank_int(rank: HandRank) -> int:
        # Pack (category, k1..kN) into a single int for fast comparison.
        # Each kicker fits in 4 bits (0..12). Up to 5 kickers (5-card hands).
        cat, kickers = rank
        v = cat
        for k in kickers:
            v = (v << 4) | k
        # Pad so different-length kicker tuples still compare consistently.
        # 5-card hands have 1..5 kickers; we left-pad to 5 slots.
        pad = 5 - len(kickers)
        if pad > 0:
            v <<= 4 * pad
        return v

    # ---------- Precompute every 5- and 3-subset evaluation ----------
    # Every 5-subset is a candidate for BOTH bottom and middle. Evaluating
    # each one once and caching by tuple eliminates ~99% of `evaluate_5`
    # calls. Cost is C(N,5) evaluations once (max 6188 for F17).
    # We also pack the rank into an int for fast comparison in the hot loop.
    eval5_cache: dict[tuple, tuple] = {}
    for combo in combinations(cards, 5):
        rank = evaluate_5(combo)
        eval5_cache[combo] = (
            rank,
            _rank_int(rank),
            bottom_royalty(rank, cfg.royalty_cfg),
            middle_royalty(rank, cfg.royalty_cfg),
            bottom_guarantees_continuation(rank),
        )

    eval3_cache: dict[tuple, tuple] = {}
    for combo in combinations(cards, 3):
        rank = evaluate_3(combo)
        eval3_cache[combo] = (
            rank,
            _rank_int(rank),
            top_royalty_value(rank, cfg.royalty_cfg),
        )

    # Pre-sort 5-subsets by middle royalty descending. Used by the
    # per-bottom upper bound tightening: scan top-down to find the highest
    # middle royalty achievable from a bottom's remaining cards.
    sorted5_by_mroy: list[tuple[frozenset, int]] = [
        (frozenset(combo), entry[3])
        for combo, entry in eval5_cache.items()
    ]
    sorted5_by_mroy.sort(key=lambda r: r[1], reverse=True)

    # Pre-sort 3-subsets by top royalty descending for analogous tightening
    # at the (bottom, middle) level.
    sorted3_by_troy: list[tuple[frozenset, int]] = [
        (frozenset(combo), entry[2])
        for combo, entry in eval3_cache.items()
    ]
    sorted3_by_troy.sort(key=lambda r: r[1], reverse=True)

    # ---------- bottom enumeration & ranking ----------
    bottom_records: list[tuple[tuple[int, ...], HandRank, int, int, bool, float]] = []
    for combo, (rank, rank_i, b_roy, _m_roy, guar) in eval5_cache.items():
        stats.bottoms_considered += 1
        prio_score = (
            (cont_bonus if guar else 0)
            + cfg.w_bottom * b_roy
            + 0.001 * (rank[0] * 10 + (rank[1][0] if rank[1] else 0))
        )
        bottom_records.append((combo, rank, rank_i, b_roy, guar, prio_score))

    # Sort by priority: stronger bottoms first establish a tighter best_ev
    # which lets the bound pruner skip more remaining branches.
    bottom_records.sort(key=lambda r: r[5], reverse=True)
    if cfg.exact or cfg.bottom_beam <= 0:
        kept_bottoms = bottom_records
    else:
        kept_bottoms = bottom_records[: cfg.bottom_beam]
    stats.bottoms_kept = len(kept_bottoms)

    best_ev = float("-inf")
    best: Optional[FantasyResult] = None

    for b_combo, b_rank, b_rank_i, b_roy, b_guar, _b_prio in kept_bottoms:
        b_set = set(b_combo)
        b_fset = frozenset(b_combo)
        rem1 = [c for c in cards if c not in b_set]
        base_ev = cfg.w_bottom * b_roy

        if cfg.use_branch_and_bound:
            # Per-bottom upper bound: instead of using the global maximum
            # middle royalty, find the best middle royalty achievable from
            # rem1. Same for top (subtract bottom only — at this level we
            # don't know which middle yet, so any 3-subset disjoint from b
            # is allowed).
            per_bot_max_mid = 0
            for fset, m_roy_ in sorted5_by_mroy:
                if fset.isdisjoint(b_fset):
                    per_bot_max_mid = m_roy_
                    break
            per_bot_max_top = 0
            for fset, t_roy_ in sorted3_by_troy:
                if fset.isdisjoint(b_fset):
                    per_bot_max_top = t_roy_
                    break

            ub = (
                base_ev
                + cfg.w_middle * per_bot_max_mid
                + cfg.w_top * per_bot_max_top
                + cont_bonus_term
            )
            if ub <= best_ev:
                stats.pruned_by_bound += 1
                continue

        # ---------- middle enumeration ----------
        middle_records: list[tuple[tuple[int, ...], int, int]] = []
        for m_combo in combinations(rem1, 5):
            stats.middles_considered += 1
            _m_rank, m_rank_i, _b_roy_unused, m_roy, _guar = eval5_cache[m_combo]
            if m_rank_i > b_rank_i:
                stats.pruned_foul += 1
                continue
            middle_records.append((m_combo, m_rank_i, m_roy))

        # int-keyed sort: cheap, helps inner top loop find a high best_ev fast
        middle_records.sort(key=lambda r: r[2], reverse=True)
        if cfg.exact or cfg.middle_beam <= 0:
            kept_middles = middle_records
        else:
            kept_middles = middle_records[: cfg.middle_beam]
        stats.middles_kept += len(kept_middles)

        for m_combo, m_rank_i, m_roy in kept_middles:
            mid_ev = base_ev + cfg.w_middle * m_roy

            if cfg.use_branch_and_bound:
                # Inner bound: use the global max top royalty here. The
                # tighter per-(b,m) bound costs more than it saves because
                # this branch fires hundreds of thousands of times.
                ub2 = mid_ev + cfg.w_top * max_top_roy + cont_bonus_term
                if ub2 <= best_ev:
                    stats.pruned_by_bound += 1
                    continue

            # Only now (after bound check passes) build rem2.
            m_set = set(m_combo)
            rem2 = [c for c in rem1 if c not in m_set]

            # ---------- top enumeration ----------
            for t_combo in combinations(rem2, 3):
                stats.tops_considered += 1
                t_rank, t_rank_i, t_roy = eval3_cache[t_combo]
                if t_rank_i > m_rank_i:
                    stats.pruned_foul += 1
                    continue

                cont = is_continuation(tier, t_rank, b_rank)
                cont_v = cont_bonus if cont else 0.0
                ev = (
                    cfg.w_bottom * b_roy
                    + cfg.w_middle * m_roy
                    + cfg.w_top * t_roy
                    + cont_v
                )
                stats.leaves_evaluated += 1
                if ev > best_ev:
                    best_ev = ev
                    immediate_roy = b_roy + m_roy + t_roy
                    placed_set = b_set | m_set | set(t_combo)
                    discards = tuple(c for c in cards if c not in placed_set)
                    # m_rank: re-look up to keep field on FantasyResult
                    m_rank_full = eval5_cache[m_combo][0]
                    best = FantasyResult(
                        top=tuple(t_combo),
                        middle=tuple(m_combo),
                        bottom=tuple(b_combo),
                        discards=discards,
                        ev=ev,
                        immediate_royalties=immediate_roy,
                        continuation=cont,
                        continuation_bonus=cont_v,
                        top_rank=t_rank,
                        middle_rank=m_rank_full,
                        bottom_rank=b_rank,
                        stats=stats,
                    )

    if best is None:
        raise RuntimeError(
            "fantasy solver found no layout (this should never happen)"
        )
    return best


__all__ = ["FantasyConfig", "SearchStats", "FantasyResult", "solve"]
