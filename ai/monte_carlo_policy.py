"""Monte-Carlo rollout policy for OFC.

For each candidate action `a`:

    EV(a) = mean over `n_rollouts` simulated futures of
            score-from-this-player's-perspective at terminal,
            given that we apply `a` now and a fixed completion policy
            plays the rest.

The action with the highest estimated EV is selected. Variance is reduced
via **common random numbers (CRN)**: every candidate action is evaluated
on the *same* set of rollout seeds, so the comparison cancels most of the
shared luck.

Knobs (see `MCConfig`)
----------------------
* `n_rollouts` — number of rollouts per candidate action.
* `completion_policy` — policy used for both players from the post-action
  state to terminal. Default: `HeuristicPolicy`.
* `opp_policy` — policy used for the opponent's *current* pending cards
  (i.e. on the same street as our decision). Default: same as completion.
* `top_k` — if set, prefilter candidate actions by `score_action` and keep
  only the top-k. Lets us afford more rollouts per action.
* `fantasy_solver` — if True (default), fantasy-tier decisions are
  delegated to the deterministic `FantasySolverPolicy` (oracle).
* `fantasy_budget` — cap on enumerated fantasy actions when `fantasy_solver`
  is False. Ignored otherwise.

Reproducibility
---------------
Given the same `seed`, the same `gs`, and the same nested policies, the
output `Action` is bit-deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from engine.fantasy import FantasyTier
from engine.royalties import DEFAULT_ROYALTIES, RoyaltyConfig
from state.action import Action
from state.game_state import GameState

from .heuristic_policy import (
    DEFAULT_WEIGHTS,
    HeuristicPolicy,
    HeuristicWeights,
    score_action,
)
from .policy import Policy
from .rollout import legal_actions, play_to_terminal, resample_deck


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MCConfig:
    """Static configuration for `MonteCarloPolicy`."""

    n_rollouts: int = 64
    top_k: Optional[int] = None  # heuristic prefilter; None = all candidates
    fantasy_solver: bool = True  # delegate fantasy to FantasySolverPolicy
    fantasy_budget: int = 4096   # only used if fantasy_solver=False
    weights: HeuristicWeights = DEFAULT_WEIGHTS  # for top_k prefilter
    royalty_cfg: RoyaltyConfig = DEFAULT_ROYALTIES


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class MonteCarloPolicy(Policy):
    """Monte-Carlo rollout policy with optional heuristic action prefilter.

    Uses CRN: every candidate action is evaluated on the same `n_rollouts`
    seeds for low-variance comparisons.
    """

    name = "monte_carlo"

    def __init__(
        self,
        config: MCConfig = MCConfig(),
        completion_policy: Optional[Policy] = None,
        opp_policy: Optional[Policy] = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self._rng = random.Random(seed)
        # Use a derived but distinct seed for default policies so that
        # policy tie-breaks are reproducible but not perfectly correlated
        # with our rollout-seed stream.
        comp_seed = self._rng.getrandbits(32) if seed is not None else None
        opp_seed = self._rng.getrandbits(32) if seed is not None else None
        self.completion = completion_policy or HeuristicPolicy(
            weights=config.weights,
            royalty_cfg=config.royalty_cfg,
            seed=comp_seed,
        )
        self.opp = opp_policy or self.completion
        # Lazy-imported to avoid hard dependency cycle if `fantasy_solver`
        # is False everywhere.
        self._fantasy_policy: Optional[Policy] = None

    # ------- public API -------
    def act(self, gs: GameState, player: int) -> Action:
        hs = gs.hands[player]
        if hs.fantasy_tier != FantasyTier.NORMAL and self.config.fantasy_solver:
            return self._fantasy_act(gs, player)

        candidates = legal_actions(
            gs, player, fantasy_budget=self.config.fantasy_budget
        )
        if not candidates:
            raise RuntimeError("MonteCarloPolicy.act: no legal actions")

        # Heuristic top-k prefilter (cheap; reduces rollout count linearly)
        if self.config.top_k is not None and len(candidates) > self.config.top_k:
            candidates = self._top_k(candidates, gs, player, self.config.top_k)

        # Single candidate: no rollouts needed.
        if len(candidates) == 1:
            return candidates[0]

        # CRN: same seeds across all candidates.
        seeds = [self._rng.getrandbits(32) for _ in range(self.config.n_rollouts)]

        best_action = candidates[0]
        best_ev = float("-inf")
        for a in candidates:
            ev = self._estimate_ev(gs, player, a, seeds)
            if ev > best_ev:
                best_ev = ev
                best_action = a
        return best_action

    # ------- helpers -------
    def _fantasy_act(self, gs: GameState, player: int) -> Action:
        if self._fantasy_policy is None:
            # Lazy import to avoid pulling fantasy module on every import.
            from fantasy.fantasy_solver import FantasySolverPolicy

            self._fantasy_policy = FantasySolverPolicy(fallback=self.completion)
        return self._fantasy_policy.act(gs, player)

    def _top_k(
        self,
        candidates: list[Action],
        gs: GameState,
        player: int,
        k: int,
    ) -> list[Action]:
        hs = gs.hands[player]
        scored = [
            (
                score_action(
                    a, hs.board, self.config.royalty_cfg, self.config.weights
                ).total,
                i,
                a,
            )
            for i, a in enumerate(candidates)
        ]
        # Sort by score desc; index `i` is a stable tiebreak so two equal
        # scores don't depend on Action equality semantics.
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [a for _, _, a in scored[:k]]

    def _estimate_ev(
        self,
        gs: GameState,
        player: int,
        action: Action,
        seeds: list[int],
    ) -> float:
        total = 0.0
        for s in seeds:
            total += self._one_rollout(gs, player, action, s)
        return total / len(seeds)

    def _one_rollout(
        self,
        gs: GameState,
        player: int,
        action: Action,
        seed: int,
    ) -> float:
        gs2 = gs.clone()
        rng = random.Random(seed)
        resample_deck(gs2, rng)

        # Apply our action first.
        gs2.step(player, action)

        # If the opponent still has pending cards on this same street, let
        # the configured `opp_policy` handle them. After that, both
        # players use the same `completion` policy.
        opp = 1 - player
        if gs2.needs_action(opp):
            gs2.step(opp, self.opp.act(gs2, opp))

        # Reseed completion policy for this rollout (CRN inside completion).
        self._reseed_policy(self.completion, seed ^ 0x9E3779B9)
        if self.opp is not self.completion:
            self._reseed_policy(self.opp, seed ^ 0x6A09E667)

        # Drive to terminal using the completion policy for both seats.
        play_to_terminal(gs2, self.completion, self.completion)

        sb = gs2.score()
        # Returns A-perspective score. Flip sign for player B.
        return float(sb.total_a if player == 0 else sb.total_b)

    @staticmethod
    def _reseed_policy(pol: Policy, seed: int) -> None:
        """Best-effort reseed for policies that expose a `_rng` attribute.

        HeuristicPolicy and RandomPolicy both use `_rng = random.Random(seed)`.
        Other policies may ignore — this is purely a determinism tightener.
        """
        rng = getattr(pol, "_rng", None)
        if rng is not None and hasattr(rng, "seed"):
            rng.seed(seed)


__all__ = ["MCConfig", "MonteCarloPolicy"]
