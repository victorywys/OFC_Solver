"""FastAnalyzer — Stack-A optimizations on top of the accurate Analyzer.

Differences from `Analyzer` (in `ui/analyzer.py`):

1. **Common Random Numbers (CRN) across candidates.** All candidates'
   i-th rollout share the same per-rollout seed, so the deck reshuffle
   used to evaluate candidate A's i-th future is identical to the one
   used for candidate B's i-th future. This dramatically reduces the
   variance of pair-wise EV *comparisons*, which is what `is_recommended`
   actually depends on. Pure win; no quality loss.

2. **Cheap opponent policy in rollouts.** The non-acting seat uses
   `FastOpponentPolicy` (rank-based, ~100x cheaper than the heuristic)
   instead of the full `HeuristicPolicy`. The acting seat keeps using
   the full heuristic, since that seat's decisions directly determine
   the EV we're estimating. Small bias on the absolute EV; comparisons
   between our candidates are largely preserved.

3. **Smart-skip on high-confidence prior hits.** If the policy_prior
   table has a clear winner at the current state (visits >=
   `smart_skip_min_visits` and lead over second-best is > 1 std-err of
   the prior means), we return the top-K prior actions directly with
   `n_rollouts == 0`. The user gets table-derived stats instantly.

4. **Lower default top_k.** Defaults to 3 instead of 5. Linear cost saver.

The accurate analyzer in `ui/analyzer.py` is unchanged and remains
available via `/api/analyze_accurate`. Same `CandidateStats` /
`AnalysisResult` dataclasses; responses are wire-compatible.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from engine.cards import card_str
from engine.fantasy import FantasyTier
from state.action import Action
from state.board import SLOT_NAMES
from state.game_state import GameState

from ai.fast_opponent_policy import FastOpponentPolicy
from ai.heuristic_policy import HeuristicPolicy, score_action
from ai.rollout import legal_actions, play_to_terminal, resample_deck

from tables import (
    FantasyEVTable,
    FoulProbTable,
    PolicyPriorTable,
    TableAwarePolicy,
    canonical_action,
)
from tables.signatures import gamestate_signature
from tables.welford import Welford

from .analyzer import (
    AnalysisResult,
    CandidateStats,
    _rollout_chunk,  # reuse the chunk wrapper that swallows worker exceptions
)


# ---------------------------------------------------------------------------
# Module-level worker for fast rollouts. Module scope so it's picklable
# across multiprocessing.spawn workers.
# ---------------------------------------------------------------------------
def _fast_rollout_chunk(args):
    """Worker entry point for the fast analyzer.

    Like `analyzer._rollout_chunk` but:
      * Receives a list of per-rollout seeds (CRN) instead of a single
        chunk seed, so each rollout has a deterministic, candidate-
        independent future.
      * Uses `FastOpponentPolicy` for non-acting seats.

    Args tuple: (gs, player, action, seeds).
    Returns the same (welford_state, foul_n, fantasy_n, done_n,
    dest_counts) shape as the accurate worker.
    """
    try:
        return _fast_rollout_chunk_impl(args)
    except BaseException as e:  # pragma: no cover (defensive)
        import traceback as _tb
        import sys as _sys
        print(
            f"[fast-rollout-worker] swallowed exception: {type(e).__name__}: {e}",
            file=_sys.stderr,
        )
        _tb.print_exc()
        return (0, 0.0, 0.0), 0, 0, 0, {}


def _fast_rollout_chunk_impl(args):
    gs, player, action, seeds = args
    # Local imports keep the worker startup small when spawned.
    from fantasy.fantasy_solver import FantasySolverPolicy

    n_seats = gs.n_players
    ev = Welford()
    n_foul = 0
    n_fantasy = 0
    n_done = 0
    dest_counts: dict[int, int] = {}

    # Seat policies are built once per chunk. The *acting* seat uses the
    # full heuristic (its move quality directly drives EV); other seats
    # use the cheap rank-based policy. Both wrap a FantasySolver so any
    # fantasy hands that arise during rollouts are solved exactly.
    acting_seed = seeds[0] if seeds else 0
    acting_policy = FantasySolverPolicy(
        fallback=HeuristicPolicy(seed=acting_seed)
    )
    cheap_fallback_for_fantasy = FantasySolverPolicy(
        fallback=HeuristicPolicy(seed=acting_seed ^ 0x5A5A)
    )
    fast_policy = FastOpponentPolicy(
        fantasy_fallback=cheap_fallback_for_fantasy
    )
    seat_policies = tuple(
        acting_policy if i == player else fast_policy
        for i in range(n_seats)
    )

    for seed in seeds:
        gs2 = gs.clone()
        try:
            gs2.step(player, action)
        except Exception:
            continue

        rng = random.Random(seed)
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


# Worker wrapper exposed at module scope so multiprocessing.spawn can
# pickle it. Mirrors `analyzer._rollout_chunk`.
def _fast_chunk_entry(args):
    return _fast_rollout_chunk(args)


# ---------------------------------------------------------------------------
class FastAnalyzer:
    """Lower-latency analyzer. Same response shape as `Analyzer`."""

    def __init__(
        self,
        policy: TableAwarePolicy,
        foul_prob_table: Optional[FoulProbTable] = None,
        policy_prior_table: Optional[PolicyPriorTable] = None,
        fantasy_ev_table: Optional[FantasyEVTable] = None,
        rollout_seed: int = 0,
        pool=None,
        n_workers: int = 1,
        smart_skip_min_visits: int = 100,
        smart_skip_min_margin: float = 0.5,
    ) -> None:
        self.policy = policy
        self.foul_prob_table = foul_prob_table
        self.policy_prior_table = policy_prior_table
        self.fantasy_ev_table = fantasy_ev_table
        self._rng = random.Random(rollout_seed)
        self.pool = pool
        self.n_workers = max(1, int(n_workers))
        self.smart_skip_min_visits = int(smart_skip_min_visits)
        self.smart_skip_min_margin = float(smart_skip_min_margin)

    # -----------------------------------------------------------------
    def analyze_heuristic_only(
        self,
        gs: GameState,
        player: int,
        *,
        top_k: int = 3,
    ) -> AnalysisResult:
        """Cheap heuristic-only path. No rollouts, no worker pool.

        Used as an overflow fallback when the analyze semaphore is full:
        users get an instant table+heuristic answer instead of queuing
        behind rollout-bound requests. Recommendation comes from the
        table-aware policy (transposition / opening book / fantasy
        cache / prior, with the heuristic as final fallback); the
        per-candidate ranking is the heuristic `score_action.total`.

        Returns the same `AnalysisResult` shape as `analyze()`, with
        `n_rollouts_per_action=0` and rollout-derived fields zeroed.
        """
        t0 = time.perf_counter()
        hs = gs.hands[player]

        # Reset thread-local diagnostic counters.
        self.policy.n_transposition_hits = 0
        self.policy.n_opening_hits = 0
        self.policy.n_fantasy_hits = 0
        self.policy.n_prior_hits = 0
        self.policy.n_fallback_calls = 0

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

        recommended = self.policy.act(gs, player)
        rec_sig = canonical_action(recommended.placements)

        # Fantasy-tier hands: the solver returns a single arrangement.
        # Skip legal enumeration — it's expensive and just confirms the
        # solver's pick.
        if hs.fantasy_tier != FantasyTier.NORMAL:
            scored: list[tuple[float, Action]] = [(0.0, recommended)]
        else:
            legals = legal_actions(gs, player)
            scored = []
            for a in legals:
                s = score_action(a, hs.board)
                scored.append((s.total, a))
            scored.sort(key=lambda x: -x[0])

        kept = scored[:max(1, top_k)]
        # Ensure recommended is in the kept set.
        if not any(canonical_action(a.placements) == rec_sig for _, a in kept):
            rec_score = next(
                (sc for sc, a in scored if canonical_action(a.placements) == rec_sig),
                0.0,
            )
            kept.append((rec_score, recommended))

        cand_stats: list[CandidateStats] = []
        for h_score, action in kept:
            asig = canonical_action(action.placements)
            tbl_prior_w = state_prior.get(asig) if state_prior else None
            cand_stats.append(
                CandidateStats(
                    placements=list(action.placements),
                    placements_str=[
                        (card_str(c), SLOT_NAMES[s]) for c, s in action.placements
                    ],
                    heuristic_score=float(h_score),
                    n_rollouts=0,
                    ev_mean=float(h_score),
                    ev_stderr=0.0,
                    foul_rate=0.0,
                    fantasy_entry_rate=0.0,
                    dest_tier_counts={},
                    horizon_ev=0.0,
                    table_foul_prob=state_foul,
                    table_prior_visits=tbl_prior_w.n if tbl_prior_w else 0,
                    table_prior_mean_ev=tbl_prior_w.mean if tbl_prior_w else None,
                    is_recommended=False,
                )
            )

        # Highest heuristic score wins recommendation.
        best = max(cand_stats, key=lambda c: c.ev_mean)
        for c in cand_stats:
            c.is_recommended = (c is best)
        cand_stats.sort(key=lambda c: (-c.ev_mean, not c.is_recommended))

        return AnalysisResult(
            player=player,
            n_players=gs.n_players,
            street=gs.current_street,
            fantasy_tier=int(hs.fantasy_tier),
            n_legal_actions=len(scored),
            n_evaluated=len(cand_stats),
            n_rollouts_per_action=0,
            future_hands=0,
            elapsed_s=time.perf_counter() - t0,
            candidates=cand_stats,
            state_table_foul_prob=state_foul,
            state_table_prior_visits=state_prior_visits,
            tier_horizon_values={},
        )

    # -----------------------------------------------------------------
    def analyze(
        self,
        gs: GameState,
        player: int,
        *,
        n_rollouts: int = 80,
        top_k: int = 3,            # lower default than Analyzer (was 5)
        future_hands: int = 0,
    ) -> AnalysisResult:
        t0 = time.perf_counter()
        hs = gs.hands[player]

        # Per-tier horizon-value table — same logic as accurate analyzer.
        # ``future_hands == -1`` requests the converged (infinite-horizon)
        # per-tier bonuses from the fantasy EV table.
        if future_hands != 0 and self.fantasy_ev_table is not None:
            tier_horizon_values = self.fantasy_ev_table.horizon_value_relative(
                future_hands
            )
        else:
            tier_horizon_values = {}

        # Reset diagnostic counters on the policy. Counters are
        # thread-local on the policy, so this reset only affects the
        # current request thread.
        self.policy.n_transposition_hits = 0
        self.policy.n_opening_hits = 0
        self.policy.n_fantasy_hits = 0
        self.policy.n_prior_hits = 0
        self.policy.n_fallback_calls = 0

        # Per-thread per-call horizon-value table for the opening-book
        # lookup. Lets the canonical book re-rank stored candidates by
        # horizon-adjusted EV at the configured `future_hands`.
        try:
            self.policy.set_horizon_values(tier_horizon_values)
        except AttributeError:
            pass

        # Per-call RNG. One GIL-safe draw from the shared self._rng, then
        # a local Random for the rest of this call. Isolates rollout
        # seeds across concurrent analyze() calls.
        call_rng = random.Random(self._rng.randrange(1 << 31))

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

        # ---------------------------------------------------------------
        # 1. SMART-SKIP: if the prior is highly confident at this node,
        # return the top prior actions directly with no rollouts.
        # ---------------------------------------------------------------
        if (
            hs.fantasy_tier == FantasyTier.NORMAL
            and state_prior
            and self._prior_is_confident(state_prior)
        ):
            cand_stats = self._candidates_from_prior(
                state_prior, gs, player, top_k, state_foul
            )
            if cand_stats:
                # Pick highest-mean candidate as recommended.
                best = max(cand_stats, key=lambda c: c.ev_mean)
                for c in cand_stats:
                    c.is_recommended = (c is best)
                cand_stats.sort(key=lambda c: (-c.ev_mean, not c.is_recommended))
                elapsed = time.perf_counter() - t0
                return AnalysisResult(
                    player=player,
                    n_players=gs.n_players,
                    street=gs.current_street,
                    fantasy_tier=int(hs.fantasy_tier),
                    n_legal_actions=len(state_prior),
                    n_evaluated=len(cand_stats),
                    n_rollouts_per_action=0,
                    future_hands=future_hands,
                    elapsed_s=elapsed,
                    candidates=cand_stats,
                    state_table_foul_prob=state_foul,
                    state_table_prior_visits=state_prior_visits,
                    tier_horizon_values=dict(tier_horizon_values),
                )

        # ---------------------------------------------------------------
        # 1b. CANONICAL OPENING-BOOK FAST-PATH:
        # If the policy's opening book is the fully-precomputed canonical
        # table and the current state is a street-1 NORMAL-tier hand,
        # consult the book directly and return its recommendation as the
        # sole candidate. The canonical book entry is provably optimal
        # under its build settings (n_rollouts=60, top_k=5), so Monte
        # Carlo on top of it adds no information — only latency.
        #
        # We can't rely on `policy.n_opening_hits` because
        # `TableAwarePolicy.act` may short-circuit at the transposition
        # cache and never reach the book lookup. So we ask the book
        # ourselves (and ignore any transposition entry from a previous
        # call). The book canonicalizes internally so any orbit-
        # equivalent hand resolves in O(1).
        # ---------------------------------------------------------------
        book = getattr(self.policy, "opening_book", None)
        is_canonical = (
            book is not None
            and type(book).__name__ == "CanonicalOpeningBookTable"
        )
        if (
            is_canonical
            and gs.current_street == 1
            and hs.fantasy_tier == FantasyTier.NORMAL
            and len(hs.pending) == 5
        ):
            # Horizon-aware lookup when the book is rich and we have a
            # non-zero horizon; otherwise plain lookup. Either way, the
            # book's stored result is authoritative for street 1.
            if (
                tier_horizon_values
                and hasattr(book, "is_rich") and book.is_rich()
            ):
                book_asig = book.lookup_horizon(
                    hs.pending, tier_horizon_values=tier_horizon_values
                )
            else:
                book_asig = book.lookup(tuple(sorted(hs.pending)))
            if book_asig is not None:
                book_action = Action(book_asig)
                rec_sig = canonical_action(book_action.placements)
                h_score = score_action(book_action, hs.board).total
                tbl_prior_w = state_prior.get(rec_sig) if state_prior else None
                book_cand = CandidateStats(
                    placements=list(book_action.placements),
                    placements_str=[
                        (card_str(c), SLOT_NAMES[s])
                        for c, s in book_action.placements
                    ],
                    heuristic_score=float(h_score),
                    n_rollouts=0,
                    ev_mean=float(h_score),
                    ev_stderr=0.0,
                    foul_rate=0.0,
                    fantasy_entry_rate=0.0,
                    dest_tier_counts={},
                    horizon_ev=0.0,
                    table_foul_prob=state_foul,
                    table_prior_visits=tbl_prior_w.n if tbl_prior_w else 0,
                    table_prior_mean_ev=tbl_prior_w.mean if tbl_prior_w else None,
                    is_recommended=True,
                )
                return AnalysisResult(
                    player=player,
                    n_players=gs.n_players,
                    street=gs.current_street,
                    fantasy_tier=int(hs.fantasy_tier),
                    n_legal_actions=1,
                    n_evaluated=1,
                    n_rollouts_per_action=0,
                    future_hands=future_hands,
                    elapsed_s=time.perf_counter() - t0,
                    candidates=[book_cand],
                    state_table_foul_prob=state_foul,
                    state_table_prior_visits=state_prior_visits,
                    tier_horizon_values=dict(tier_horizon_values),
                )

        # ---------------------------------------------------------------
        # 2. Build candidate set (same as accurate analyzer).
        # ---------------------------------------------------------------
        recommended = self.policy.act(gs, player)
        rec_sig = canonical_action(recommended.placements)

        if hs.fantasy_tier != FantasyTier.NORMAL:
            legals = [recommended]
            kept_actions: list[tuple[float, Action]] = [(0.0, recommended)]
            kept_sigs: set[tuple] = {rec_sig}
        else:
            legals = legal_actions(gs, player)
            scored = []
            for a in legals:
                s = score_action(a, hs.board)
                scored.append((s.total, a))
            scored.sort(key=lambda x: -x[0])

            kept_actions = []
            kept_sigs = set()
            for sc, a in scored[:top_k]:
                kept_actions.append((sc, a))
                kept_sigs.add(canonical_action(a.placements))
            if rec_sig not in kept_sigs:
                rec_score = next(
                    (sc for sc, a in scored if canonical_action(a.placements) == rec_sig),
                    0.0,
                )
                kept_actions.append((rec_score, recommended))
                kept_sigs.add(rec_sig)

        # ---------------------------------------------------------------
        # 3. Run rollouts with CRN: a single shared seed list, reused
        # across every candidate. Each candidate's i-th rollout faces
        # the same future as every other candidate's i-th rollout.
        # ---------------------------------------------------------------
        if n_rollouts > 0:
            shared_seeds = [
                call_rng.randint(0, 2**31 - 1) for _ in range(n_rollouts)
            ]
        else:
            shared_seeds = []

        cand_stats: list[CandidateStats] = []
        for h_score, action in kept_actions:
            asig = canonical_action(action.placements)
            tbl_foul: Optional[float] = state_foul
            tbl_prior_w = state_prior.get(asig) if state_prior else None

            ev_w, foul_n, fan_n, n_actual, dest_counts = self._rollout_action_crn(
                gs, player, action, shared_seeds
            )

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

        # ---------------------------------------------------------------
        # 4. Pick recommendation by highest combined EV (rollout EV plus
        # horizon bonus). With H=0 the bonus is zero and we reduce to
        # argmax(ev_mean); with H>0 we respect the user's horizon.
        # ---------------------------------------------------------------
        any_rollouts = any(c.n_rollouts > 0 for c in cand_stats)
        if any_rollouts:
            best = max(cand_stats, key=lambda c: c.combined_ev)
            for c in cand_stats:
                c.is_recommended = (c is best)

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
    def _prior_is_confident(self, state_prior: dict) -> bool:
        """True iff the policy prior table has a clear winner at this node.

        Criteria:
          * The best action has >= `smart_skip_min_visits` recorded visits.
          * Its mean EV exceeds the second-best (with enough visits) by at
            least `smart_skip_min_margin`.
        """
        qual = [
            (w.n, w.mean) for w in state_prior.values()
            if w.n >= self.smart_skip_min_visits
        ]
        if not qual:
            return False
        qual.sort(key=lambda nm: -nm[1])
        if len(qual) == 1:
            return True
        return (qual[0][1] - qual[1][1]) >= self.smart_skip_min_margin

    # -----------------------------------------------------------------
    def _candidates_from_prior(
        self,
        state_prior: dict,
        gs: GameState,
        player: int,
        top_k: int,
        state_foul: Optional[float],
    ) -> list[CandidateStats]:
        """Materialize top-K prior actions as zero-rollout candidates.

        Skips entries with too few visits or whose action signature isn't
        a legal action in the current state (the prior may include
        action signatures from a slightly different canonical form).
        """
        entries = [
            (asig, w) for asig, w in state_prior.items()
            if w.n >= self.smart_skip_min_visits
        ]
        entries.sort(key=lambda kv: -kv[1].mean)
        cands: list[CandidateStats] = []
        for asig, w in entries[:max(1, top_k)]:
            try:
                action = Action(asig)
            except Exception:
                continue
            if not self._is_legal(action, gs, player):
                continue
            cands.append(
                CandidateStats(
                    placements=list(action.placements),
                    placements_str=[
                        (card_str(c), SLOT_NAMES[s]) for c, s in action.placements
                    ],
                    heuristic_score=0.0,
                    n_rollouts=0,
                    ev_mean=w.mean,
                    ev_stderr=(w.M2 / (w.n - 1) / w.n) ** 0.5 if w.n > 1 else 0.0,
                    foul_rate=0.0,
                    fantasy_entry_rate=0.0,
                    dest_tier_counts={},
                    horizon_ev=0.0,
                    table_foul_prob=state_foul,
                    table_prior_visits=w.n,
                    table_prior_mean_ev=w.mean,
                    is_recommended=False,
                )
            )
        return cands

    # -----------------------------------------------------------------
    @staticmethod
    def _is_legal(action: Action, gs: GameState, player: int) -> bool:
        """Cheap legality check: does the action place the right cards
        in slots that still have capacity?"""
        hs = gs.hands[player]
        pending = set(hs.pending)
        placed = [c for c, s in action.placements]
        if set(placed) != pending:
            return False
        # capacity check
        need = [0, 0, 0, 0]
        for _c, s in action.placements:
            need[s] += 1
        if (need[0] > hs.board.free_top()
                or need[1] > hs.board.free_middle()
                or need[2] > hs.board.free_bottom()):
            return False
        return True

    # -----------------------------------------------------------------
    def _rollout_action_crn(
        self,
        gs: GameState,
        player: int,
        action: Action,
        shared_seeds: list[int],
    ) -> tuple[Welford, int, int, int, dict[int, int]]:
        """Run rollouts with Common Random Numbers across candidates.

        `shared_seeds[i]` is the same for every candidate's i-th rollout.
        """
        n_rollouts = len(shared_seeds)
        if n_rollouts <= 0:
            return Welford(), 0, 0, 0, {}

        if self.pool is not None and self.n_workers > 1:
            n_chunks = min(self.n_workers, n_rollouts)
        else:
            n_chunks = 1

        # Slice the shared seed list across chunks.
        base = n_rollouts // n_chunks
        rem = n_rollouts % n_chunks
        sizes = [base + (1 if i < rem else 0) for i in range(n_chunks)]
        chunks: list[list[int]] = []
        idx = 0
        for sz in sizes:
            if sz <= 0:
                continue
            chunks.append(shared_seeds[idx: idx + sz])
            idx += sz

        tasks = [(gs, player, action, seeds) for seeds in chunks]

        if self.pool is not None and self.n_workers > 1:
            results = list(self.pool.imap_unordered(_fast_chunk_entry, tasks))
        else:
            results = [_fast_chunk_entry(t) for t in tasks]

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


__all__ = ["FastAnalyzer"]
