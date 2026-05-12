"""Composite policy that consults precomputed tables before falling back.

Lookup order:
    1. **TranspositionTable** — if state has been solved before, return.
    2. **OpeningBook** — street-1 only; map dealt hand to best stored action.
    3. **FantasyArrangementCache** — fantasy hands; reuse cached arrangement.
    4. **PolicyPriorTable** — argmax over cached state-action EV
       (with min_visits guard).
    5. **Fallback policy** — typically `MonteCarloPolicy` or `HeuristicPolicy`.
       Only consulted when no table hits.

Optional `record_in_transposition`: store every decision into the
transposition table so subsequent identical states are O(1).

The composite is itself a `Policy` and can be used anywhere a `Policy`
is expected — including as a `completion_policy` for a *different*
`MonteCarloPolicy`, or as a self-play factory.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from engine.fantasy import FantasyTier
from state.action import Action
from state.game_state import GameState

from ai.policy import Policy


class _PolicyCounters(threading.local):
    """Per-thread diagnostic counters for TableAwarePolicy.

    Each request thread sees its own zeroed counters on first access,
    so concurrent analyze() calls from different threads do not mix
    diagnostics. Rollout worker *processes* (spawned via multiprocessing)
    have their own policy instances and never touch this state.
    """

    def __init__(self) -> None:
        # threading.local.__init__ is invoked once per thread on first
        # attribute access in that thread.
        self.transposition = 0
        self.opening = 0
        self.fantasy = 0
        self.prior = 0
        self.fallback = 0
        # Per-thread per-call horizon-value override for the opening
        # book. None disables horizon-aware lookup (default behaviour).
        # Set via TableAwarePolicy.set_horizon_values() at the start of
        # each request; reset to None to revert.
        self.tier_horizon_values = None  # type: ignore[assignment]

from .fantasy_cache import FantasyArrangementCache
from .opening_book import OpeningBookTable
from .canonical_opening import CanonicalOpeningBookTable
from .policy_prior import PolicyPriorTable
from .signatures import (
    canonical_action,
    gamestate_signature,
    street1_hand_signature,
)
from .transposition import TranspositionTable


@dataclass
class TableAwareConfig:
    """Knobs for the composite policy."""

    # min visits required for a table to be authoritative
    opening_min_visits: int = 4
    prior_min_visits: int = 8
    # whether to memoize live decisions in the transposition table
    record_in_transposition: bool = True
    # whether each table is enabled
    use_transposition: bool = True
    use_opening_book: bool = True
    use_fantasy_cache: bool = True
    use_policy_prior: bool = True


class TableAwarePolicy(Policy):
    """Multi-table composite policy. See module docstring for lookup order."""

    name = "table_aware"

    def __init__(
        self,
        fallback: Policy,
        config: TableAwareConfig = TableAwareConfig(),
        transposition: Optional[TranspositionTable] = None,
        opening_book: Optional[OpeningBookTable] = None,
        fantasy_cache: Optional[FantasyArrangementCache] = None,
        policy_prior: Optional[PolicyPriorTable] = None,
    ) -> None:
        self.fallback = fallback
        self.config = config
        self.transposition = transposition
        self.opening_book = opening_book
        self.fantasy_cache = fantasy_cache
        self.policy_prior = policy_prior
        # diagnostics — thread-local so concurrent analyze() calls do not
        # contaminate each other's counts. Exposed via properties below
        # so that existing call sites (`pol.n_transposition_hits = 0`,
        # `pol.n_transposition_hits += 1`) continue to work unchanged.
        self._counters = _PolicyCounters()

    # ----- thread-local diagnostic counter properties -----
    @property
    def n_transposition_hits(self) -> int:
        return self._counters.transposition

    @n_transposition_hits.setter
    def n_transposition_hits(self, v: int) -> None:
        self._counters.transposition = v

    @property
    def n_opening_hits(self) -> int:
        return self._counters.opening

    @n_opening_hits.setter
    def n_opening_hits(self, v: int) -> None:
        self._counters.opening = v

    @property
    def n_fantasy_hits(self) -> int:
        return self._counters.fantasy

    @n_fantasy_hits.setter
    def n_fantasy_hits(self, v: int) -> None:
        self._counters.fantasy = v

    @property
    def n_prior_hits(self) -> int:
        return self._counters.prior

    @n_prior_hits.setter
    def n_prior_hits(self, v: int) -> None:
        self._counters.prior = v

    @property
    def n_fallback_calls(self) -> int:
        return self._counters.fallback

    @n_fallback_calls.setter
    def n_fallback_calls(self, v: int) -> None:
        self._counters.fallback = v

    # ----- horizon-aware opening book override (per-thread) -----
    def set_horizon_values(self, values: Optional[dict[int, float]]) -> None:
        """Set the per-tier horizon-bonus table used by the next call.

        When non-empty and the loaded opening book is a rich
        :class:`CanonicalOpeningBookTable`, the next street-1
        normal-tier decision re-ranks the book's stored candidates as ::

            ev_mean + sum_t P(next_tier == t | a) * values[t]

        and returns the new argmax. Pass ``None`` (or an empty dict) to
        disable. The value is thread-local — concurrent requests do not
        mix horizons.
        """
        self._counters.tier_horizon_values = values

    def act(self, gs: GameState, player: int) -> Action:
        cfg = self.config
        hs = gs.hands[player]

        # ----- 1. transposition -----
        sig = gamestate_signature(gs, player)
        if cfg.use_transposition and self.transposition is not None:
            cached = self.transposition.lookup(sig)
            if cached is not None and self._is_legal(cached, gs, player):
                self.n_transposition_hits += 1
                return cached

        # ----- 2. opening book (street 1, normal tier) -----
        if (
            cfg.use_opening_book
            and self.opening_book is not None
            and gs.current_street == 1
            and hs.fantasy_tier == FantasyTier.NORMAL
            and len(hs.pending) == 5
        ):
            # Horizon-aware fast path: when a rich canonical book is
            # loaded and a per-thread horizon-bonus table has been
            # supplied via set_horizon_values(...), re-rank the stored
            # candidates by ``ev_mean + sum_t P(t|a) * bonus[t]``. This
            # serves the street-1 decision directly from the book — no
            # rollouts — and the result depends on the requested horizon.
            best_asig: Optional[tuple[tuple[int, int], ...]] = None
            horizon_values = getattr(self._counters, "tier_horizon_values", None)
            if (
                horizon_values
                and isinstance(self.opening_book, CanonicalOpeningBookTable)
                and self.opening_book.is_rich()
            ):
                best_asig = self.opening_book.lookup_horizon(
                    hs.pending, tier_horizon_values=horizon_values
                )
            if best_asig is None:
                hand_key = street1_hand_signature(hs.pending)
                best_asig = self.opening_book.lookup(
                    hand_key, min_visits=cfg.opening_min_visits
                )
            if best_asig is not None:
                action = Action(best_asig)
                if self._is_legal(action, gs, player):
                    self.n_opening_hits += 1
                    self._record(sig, action)
                    return action

        # ----- 3. fantasy arrangement cache -----
        if (
            cfg.use_fantasy_cache
            and self.fantasy_cache is not None
            and hs.fantasy_tier != FantasyTier.NORMAL
        ):
            entry = self.fantasy_cache.lookup(hs.pending, hs.fantasy_tier)
            if entry is not None:
                try:
                    action = entry.to_action(hs.pending)
                except ValueError:
                    action = None
                if action is not None and self._is_legal(action, gs, player):
                    self.n_fantasy_hits += 1
                    self._record(sig, action)
                    return action

        # ----- 4. policy prior argmax -----
        if cfg.use_policy_prior and self.policy_prior is not None:
            best_asig = self.policy_prior.best_action(
                sig, min_visits=cfg.prior_min_visits
            )
            if best_asig is not None:
                action = Action(best_asig)
                if self._is_legal(action, gs, player):
                    self.n_prior_hits += 1
                    self._record(sig, action)
                    return action

        # ----- 5. fallback -----
        self.n_fallback_calls += 1
        action = self.fallback.act(gs, player)
        self._record(sig, action)
        return action

    # ----- helpers -----
    def _record(self, sig, action: Action) -> None:
        if (
            self.config.record_in_transposition
            and self.transposition is not None
        ):
            self.transposition.store(sig, action)

    @staticmethod
    def _is_legal(action: Action, gs: GameState, player: int) -> bool:
        """Cheap legality guard before applying a cached action.

        Verifies (1) the action's cards equal the player's pending and
        (2) row capacities are respected after the placements.

        This protects against cache poisoning if, e.g., a stored opening
        action somehow doesn't match the current pending hand.
        """
        hs = gs.hands[player]
        action_cards = sorted(c for c, _ in action.placements)
        if action_cards != sorted(hs.pending):
            return False
        # capacity check
        from state.board import (
            ROW_CAPACITY,
            SLOT_BOTTOM,
            SLOT_DISCARD,
            SLOT_MIDDLE,
            SLOT_TOP,
        )
        free = (
            hs.board.free_top(),
            hs.board.free_middle(),
            hs.board.free_bottom(),
        )
        need = [0, 0, 0]
        for _c, s in action.placements:
            if s == SLOT_DISCARD:
                continue
            if s < SLOT_TOP or s > SLOT_BOTTOM:
                return False
            need[s] += 1
        if (
            need[SLOT_TOP] > free[SLOT_TOP]
            or need[SLOT_MIDDLE] > free[SLOT_MIDDLE]
            or need[SLOT_BOTTOM] > free[SLOT_BOTTOM]
        ):
            return False
        return True


__all__ = ["TableAwareConfig", "TableAwarePolicy"]
