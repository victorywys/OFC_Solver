"""Analyzer: take a `GameState` + player and return a recommendation + stats.

Workflow per call:
    1. Pick the recommended action via `TableAwarePolicy` (uses every
       precomputed table with fallback to a strong policy).
    2. Enumerate legal actions, score them with the heuristic, keep the
       top-K (always including the recommended action).
    3. For each kept action, run N Monte-Carlo rollouts:
         - clone the state, apply the action,
         - resample the deck (fresh future), populate any unknown opp
           pending from the resampled deck if needed,
         - play to terminal with a heuristic for both seats,
         - record signed score, foul flag, next-hand fantasy tier.
    4. Aggregate per-candidate Welford(EV), foul rate, and fantasy entry
       rate, and return as a JSON-serializable dict.

Precomputed-table lookups
-------------------------
Each candidate is also probed against the loaded `FoulProbTable` and
`PolicyPriorTable` for instant statistics. These complement the rollout
estimates: the table values reflect long-run self-play history, while
rollouts estimate the *current* opponent assumption.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from engine.cards import card_str
from engine.fantasy import FantasyTier, next_fantasy_tier
from state.action import Action
from state.board import (
    PlayerBoard,
    SLOT_BOTTOM,
    SLOT_DISCARD,
    SLOT_MIDDLE,
    SLOT_NAMES,
    SLOT_TOP,
)
from state.game_state import GameState

from ai.heuristic_policy import (
    DEFAULT_WEIGHTS,
    HeuristicPolicy,
    score_action,
)
from ai.policy import Policy
from ai.rollout import legal_actions, play_to_terminal, resample_deck

from tables import (
    FantasyEVTable,
    FoulProbTable,
    PolicyPriorTable,
    TableAwarePolicy,
    canonical_action,
    state_signature,
    turn_state_signature,
)
from tables.signatures import gamestate_signature
from tables.welford import Welford


# ---------------------------------------------------------------------------
# Worker for parallel rollouts (must be at module scope to be picklable).
# ---------------------------------------------------------------------------
def _rollout_chunk(args):
    """Run a chunk of rollouts in a worker process.

    Args is a tuple of (gs, player, action, n_rollouts, base_seed) so this
    function can be used directly with Pool.imap_unordered.

    Returns a tuple (welford_state, n_foul, n_fantasy, n_done, dest_counts)
    where welford_state is (n, mean, M2) for cheap merge in the parent.

    All exceptions are caught and returned as a partial-zero result. A
    worker that dies forces the Pool to fork a replacement, and that
    replacement is liable to inherit a locked synchronize primitive
    (causing every subsequent rollout to deadlock). The Welford merge
    in the parent is robust to a chunk that contributed nothing.
    """
    try:
        return _rollout_chunk_impl(args)
    except BaseException as e:  # pragma: no cover (defensive)
        import traceback as _tb, sys as _sys
        print(
            f"[rollout-worker] swallowed exception: {type(e).__name__}: {e}",
            file=_sys.stderr,
        )
        _tb.print_exc()
        return (0, 0.0, 0.0), 0, 0, 0, {}


def _rollout_chunk_impl(args):
    gs, player, action, n_rollouts, base_seed = args
    # Local imports keep the worker startup small when the parent forks.
    from fantasy.fantasy_solver import FantasySolverPolicy

    rng = random.Random(base_seed)
    ev = Welford()
    n_foul = 0
    n_fantasy = 0
    n_done = 0
    dest_counts: dict[int, int] = {}
    n_seats = gs.n_players
    seat_policies = tuple(
        FantasySolverPolicy(
            fallback=HeuristicPolicy(seed=rng.randint(0, 2**31 - 1))
        )
        for _ in range(n_seats)
    )

    for _ in range(n_rollouts):
        gs2 = gs.clone()
        try:
            gs2.step(player, action)
        except Exception:
            continue
        resample_deck(gs2, rng)

        # Deal pending to any opponent that's empty (synthetic info-set).
        need_skip = False
        for opp in range(n_seats):
            if opp == player:
                continue
            opp_hs = gs2.hands[opp]
            if opp_hs.finished or opp_hs.pending:
                continue
            if (
                gs2.current_street == 1
                and opp_hs.fantasy_tier == FantasyTier.NORMAL
            ):
                n_need = 5
            elif opp_hs.fantasy_tier != FantasyTier.NORMAL:
                n_need = opp_hs.fantasy_tier.n_cards
            else:
                n_need = 3
            if len(gs2.deck) < n_need:
                need_skip = True
                break
            opp_hs.pending = gs2.deck.deal(n_need)
        if need_skip:
            continue

        try:
            play_to_terminal(gs2, *seat_policies)
        except Exception:
            continue

        totals = gs2.score_each()
        fouls = gs2.fouls()
        ev.push(float(totals[player]))
        if fouls[player]:
            n_foul += 1
        try:
            tiers = gs2.next_fantasy_tiers()
        except Exception:
            tiers = tuple(FantasyTier.NORMAL for _ in range(n_seats))
        dest = int(tiers[player])
        dest_counts[dest] = dest_counts.get(dest, 0) + 1
        if tiers[player] != FantasyTier.NORMAL:
            n_fantasy += 1
        n_done += 1

    return (ev.n, ev.mean, ev.M2), n_foul, n_fantasy, n_done, dest_counts


# ---------------------------------------------------------------------------
# Output dataclasses (all fields JSON-serializable via .to_dict())
# ---------------------------------------------------------------------------
@dataclass
class CandidateStats:
    """Per-candidate evaluation."""

    placements: list[tuple[int, int]]   # [(card_id, slot_id), ...]
    placements_str: list[tuple[str, str]]
    heuristic_score: float
    n_rollouts: int
    ev_mean: float
    ev_stderr: float
    foul_rate: float
    fantasy_entry_rate: float
    # Destination-tier histogram from rollouts (next-hand fantasy tier).
    # Keys are int(FantasyTier); values are counts.
    dest_tier_counts: dict[int, int]
    # Future-hand horizon EV bonus, derived from dest_tier_counts +
    # FantasyEVTable.horizon_value_relative(H). 0.0 when no fantasy table
    # was supplied or `future_hands == 0`.
    horizon_ev: float
    table_foul_prob: Optional[float]    # from FoulProbTable, if present
    table_prior_visits: int             # support in PolicyPriorTable
    table_prior_mean_ev: Optional[float]
    is_recommended: bool

    @property
    def combined_ev(self) -> float:
        """This-hand EV plus future-hand horizon bonus."""
        return self.ev_mean + self.horizon_ev

    def to_dict(self) -> dict:
        return {
            "placements": [
                {"card": c, "card_str": card_str(c),
                 "slot": s, "slot_str": SLOT_NAMES[s]}
                for c, s in self.placements
            ],
            "heuristic_score": self.heuristic_score,
            "n_rollouts": self.n_rollouts,
            "ev_mean": self.ev_mean,
            "ev_stderr": self.ev_stderr,
            "foul_rate": self.foul_rate,
            "fantasy_entry_rate": self.fantasy_entry_rate,
            "dest_tier_counts": dict(self.dest_tier_counts),
            "horizon_ev": self.horizon_ev,
            "combined_ev": self.combined_ev,
            "table_foul_prob": self.table_foul_prob,
            "table_prior_visits": self.table_prior_visits,
            "table_prior_mean_ev": self.table_prior_mean_ev,
            "is_recommended": self.is_recommended,
        }


@dataclass
class AnalysisResult:
    """Top-level return of `Analyzer.analyze`."""

    player: int
    n_players: int
    street: int
    fantasy_tier: int
    n_legal_actions: int
    n_evaluated: int
    n_rollouts_per_action: int
    future_hands: int             # the H used for horizon_ev (0 = this hand only, -1 = infinite horizon)
    elapsed_s: float
    candidates: list[CandidateStats]
    state_table_foul_prob: Optional[float]
    state_table_prior_visits: int
    # Per-tier horizon-value table actually used (anchored vs NORMAL).
    # Empty when `future_hands == 0` or fantasy_ev_table is missing.
    tier_horizon_values: dict[int, float]

    def to_dict(self) -> dict:
        return {
            "player": self.player,
            "n_players": self.n_players,
            "street": self.street,
            "fantasy_tier": self.fantasy_tier,
            "n_legal_actions": self.n_legal_actions,
            "n_evaluated": self.n_evaluated,
            "n_rollouts_per_action": self.n_rollouts_per_action,
            "future_hands": self.future_hands,
            "elapsed_s": self.elapsed_s,
            "state_table_foul_prob": self.state_table_foul_prob,
            "state_table_prior_visits": self.state_table_prior_visits,
            "tier_horizon_values": dict(self.tier_horizon_values),
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class Analyzer:
    """Pre-built tables wrapper. Construct once, reuse across requests."""

    def __init__(
        self,
        policy: TableAwarePolicy,
        foul_prob_table: Optional[FoulProbTable] = None,
        policy_prior_table: Optional[PolicyPriorTable] = None,
        fantasy_ev_table: Optional[FantasyEVTable] = None,
        rollout_seed: int = 0,
        pool=None,
        n_workers: int = 1,
    ) -> None:
        self.policy = policy
        self.foul_prob_table = foul_prob_table
        self.policy_prior_table = policy_prior_table
        self.fantasy_ev_table = fantasy_ev_table
        self._rng = random.Random(rollout_seed)
        # Optional multiprocessing.Pool for parallel rollouts. When None,
        # rollouts run sequentially in this process.
        self.pool = pool
        self.n_workers = max(1, int(n_workers))

    # -----------------------------------------------------------------
    def analyze(
        self,
        gs: GameState,
        player: int,
        *,
        n_rollouts: int = 80,
        top_k: int = 5,
        future_hands: int = 0,
    ) -> AnalysisResult:
        t0 = time.perf_counter()
        hs = gs.hands[player]

        # Per-tier horizon-value table actually used to compute the
        # `horizon_ev` of each candidate. When the fantasy table is
        # missing or the user asked for zero future hands, this is empty
        # and every horizon_ev is 0. ``future_hands == -1`` means
        # "infinite horizon" and is forwarded to the EV table, which
        # returns the converged (relative-value) per-tier bonuses.
        if future_hands != 0 and self.fantasy_ev_table is not None:
            tier_horizon_values = self.fantasy_ev_table.horizon_value_relative(
                future_hands
            )
        else:
            tier_horizon_values = {}

        # Reset diagnostic counters on the policy so the recommendation log
        # reflects only this analysis call. Counters are thread-local on the
        # policy, so this reset only affects the current request thread.
        self.policy.n_transposition_hits = 0
        self.policy.n_opening_hits = 0
        self.policy.n_fantasy_hits = 0
        self.policy.n_prior_hits = 0
        self.policy.n_fallback_calls = 0

        # Plumb the per-call horizon-value table into the policy so that
        # the street-1 normal-tier opening-book lookup can re-rank stored
        # candidates by horizon-adjusted EV. Thread-local, so concurrent
        # analyze() calls with different `future_hands` don't mix.
        try:
            self.policy.set_horizon_values(tier_horizon_values)
        except AttributeError:
            # Older or non-TableAware policies — no-op.
            pass

        # Per-call RNG. We draw a single seed from the shared self._rng
        # (GIL-safe even under concurrent calls) and then use a local
        # Random instance for all subsequent draws in this analyze() call.
        # This isolates rollout seeds across concurrent requests.
        call_rng = random.Random(self._rng.randrange(1 << 31))

        # 1. Recommended action from the table-aware policy
        recommended = self.policy.act(gs, player)
        rec_sig = canonical_action(recommended.placements)

        # 2. Enumerate legal actions, score with heuristic, keep top-K.
        # For fantasy hands, the action space is enormous (~1M for F14)
        # and the solver already returns the optimum — there's no useful
        # ranking among the millions of layouts the heuristic would
        # produce. So in fantasy mode we just expose the solver's pick
        # as the sole candidate and skip the wide enumeration.
        if hs.fantasy_tier != FantasyTier.NORMAL:
            legals = [recommended]
            scored = [(0.0, recommended)]
            kept_actions: list[tuple[float, Action]] = [(0.0, recommended)]
            kept_sigs: set[tuple] = {rec_sig}
        else:
            legals = legal_actions(gs, player)
            scored = []
            for a in legals:
                s = score_action(a, hs.board)
                scored.append((s.total, a))
            scored.sort(key=lambda x: -x[0])

            # Always include the recommended action even if it falls below top-K
            kept_actions = []
            kept_sigs = set()
            for sc, a in scored[:top_k]:
                kept_actions.append((sc, a))
                kept_sigs.add(canonical_action(a.placements))
            if rec_sig not in kept_sigs:
                # find recommended's heuristic score
                rec_score = next(
                    (sc for sc, a in scored if canonical_action(a.placements) == rec_sig),
                    0.0,
                )
                kept_actions.append((rec_score, recommended))
                kept_sigs.add(rec_sig)

        # 3. Rollouts per kept action
        cand_stats: list[CandidateStats] = []
        state_sig = gamestate_signature(gs, player)
        state_foul = (
            self.foul_prob_table.lookup(state_sig)
            if self.foul_prob_table is not None
            else None
        )
        state_prior = (
            self.policy_prior_table.actions(state_sig)
            if self.policy_prior_table is not None
            else {}
        )
        state_prior_visits = sum(w.n for w in state_prior.values())

        for h_score, action in kept_actions:
            asig = canonical_action(action.placements)

            # table lookups for this (state, action)
            tbl_foul: Optional[float] = state_foul   # current node only; row-by-row sigs would deepen this
            tbl_prior_w = state_prior.get(asig) if state_prior else None

            ev_w, foul_n, fan_n, n_actual, dest_counts = self._rollout_action(
                gs, player, action, n_rollouts, call_rng
            )

            # Per-candidate horizon EV: probability-weighted bonus across
            # destination tiers observed in this candidate's rollouts.
            if tier_horizon_values and n_actual > 0:
                horizon_ev = sum(
                    (cnt / n_actual) * tier_horizon_values.get(tier, 0.0)
                    for tier, cnt in dest_counts.items()
                )
            else:
                horizon_ev = 0.0

            cand_stats.append(
                CandidateStats(
                    placements=list(action.placements),
                    placements_str=[
                        (card_str(c), SLOT_NAMES[s]) for c, s in action.placements
                    ],
                    heuristic_score=float(h_score),
                    n_rollouts=n_actual,
                    ev_mean=ev_w.mean,
                    ev_stderr=ev_w.stderr,
                    foul_rate=foul_n / n_actual if n_actual else 0.0,
                    fantasy_entry_rate=fan_n / n_actual if n_actual else 0.0,
                    dest_tier_counts=dict(dest_counts),
                    horizon_ev=horizon_ev,
                    table_foul_prob=tbl_foul,
                    table_prior_visits=tbl_prior_w.n if tbl_prior_w else 0,
                    table_prior_mean_ev=tbl_prior_w.mean if tbl_prior_w else None,
                    is_recommended=(asig == rec_sig),
                )
            )

        # 4. Override recommendation: pick the candidate with the highest
        # combined EV (this-hand rollout EV plus horizon bonus). This
        # respects the user's `future_hands` request: at H=0 the bonus
        # is zero and we reduce to argmax(ev_mean); at H>0 we account
        # for the per-candidate fantasy-entry distribution.
        #
        # When no rollouts were requested (n_rollouts == 0) every
        # candidate has ev_mean == 0 and horizon_ev == 0; in that
        # degenerate case fall back to the policy's pick to avoid an
        # arbitrary recommendation.
        any_rollouts = any(c.n_rollouts > 0 for c in cand_stats)
        if any_rollouts:
            best = max(cand_stats, key=lambda c: c.combined_ev)
            for c in cand_stats:
                c.is_recommended = (c is best)

        # 5. Sort by combined EV (desc), with recommended as a tiebreaker.
        cand_stats.sort(key=lambda c: (-c.combined_ev, not c.is_recommended))

        elapsed = time.perf_counter() - t0
        return AnalysisResult(
            player=player,
            n_players=gs.n_players,
            street=gs.current_street,
            fantasy_tier=int(hs.fantasy_tier),
            n_legal_actions=len(legals),
            n_evaluated=len(cand_stats),
            n_rollouts_per_action=n_rollouts,
            future_hands=future_hands,
            elapsed_s=elapsed,
            candidates=cand_stats,
            state_table_foul_prob=state_foul,
            state_table_prior_visits=state_prior_visits,
            tier_horizon_values=dict(tier_horizon_values),
        )

    # -----------------------------------------------------------------
    def _rollout_action(
        self,
        gs: GameState,
        player: int,
        action: Action,
        n_rollouts: int,
        rng: Optional[random.Random] = None,
    ) -> tuple[Welford, int, int, int, dict[int, int]]:
        """Run N rollouts from `gs.step(player, action)` to terminal.

        Returns (Welford(signed_score), foul_count, fantasy_entry_count,
                 actual_rollouts_completed, dest_tier_counts) where the
                 last item maps `int(FantasyTier)` -> count of rollouts
                 ending in that next-hand tier for the analyzed player.
        """
        if n_rollouts <= 0:
            return Welford(), 0, 0, 0, {}

        # Split work into chunks. With a pool we want one chunk per worker
        # (fewer pickle round-trips). Without a pool we run a single chunk
        # in-process.
        if self.pool is not None and self.n_workers > 1:
            n_chunks = min(self.n_workers, n_rollouts)
        else:
            n_chunks = 1

        # Per-call RNG isolates this analyze() invocation's seeds from
        # concurrent calls; fall back to the shared self._rng for direct
        # callers (e.g. tests).
        seed_rng = rng if rng is not None else self._rng
        base = n_rollouts // n_chunks
        rem = n_rollouts % n_chunks
        sizes = [base + (1 if i < rem else 0) for i in range(n_chunks)]
        tasks = [
            (gs, player, action, size,
             seed_rng.randint(0, 2**31 - 1))
            for size in sizes if size > 0
        ]

        if self.pool is not None and self.n_workers > 1:
            results = list(self.pool.imap_unordered(_rollout_chunk, tasks))
        else:
            results = [_rollout_chunk(t) for t in tasks]

        # Merge.
        ev = Welford()
        n_foul = 0
        n_fantasy = 0
        n_done = 0
        dest_counts: dict[int, int] = {}
        for w_state, foul_n, fan_n, done_n, dc in results:
            sub = Welford()
            sub.n, sub.mean, sub.M2 = w_state
            ev.merge(sub)
            n_foul += foul_n
            n_fantasy += fan_n
            n_done += done_n
            for k, v in dc.items():
                dest_counts[k] = dest_counts.get(k, 0) + v

        return ev, n_foul, n_fantasy, n_done, dest_counts


__all__ = ["Analyzer", "AnalysisResult", "CandidateStats"]
