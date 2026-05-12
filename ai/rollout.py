"""Rollout primitives for Monte-Carlo OFC AI.

This module provides the **simulation kernel** used by `MonteCarloPolicy`:

    * `play_to_terminal(gs, p0, p1)` — drive a `GameState` to terminal using
      two completion policies. Pure orchestration; no decision logic.
    * `resample_deck(gs, rng)` — reshuffle the deck's remaining cards in
      place. Used so each rollout sees a different future even when the
      cloned `GameState` shares deck contents with its origin.
    * `legal_actions(gs, player, fantasy_budget)` — collect all legal
      actions for the player at the current decision point.

Design choices
--------------
* **Information-set semantics**. The opponent's already-dealt pending cards
  are *kept* in the rollout (we treat the cloned `GameState` as ground
  truth). Only the *unrevealed* deck contents are reshuffled. This matches
  the standard OFC simulation model: opponents' boards are public and
  their pending hands, while strictly hidden, are typically treated as
  observable for evaluation purposes.
* **Determinism**. `resample_deck` takes an explicit `random.Random`
  instance, so callers can construct CRN (common random numbers) by
  re-using seeds across candidate actions to reduce comparison variance.
* **No allocation in the hot loop**. `play_to_terminal` does not allocate
  beyond what the underlying `Action` enumerators already produce.
"""

from __future__ import annotations

import random

from engine.fantasy import FantasyTier
from state.action import (
    Action,
    enumerate_initial_actions,
    enumerate_pineapple_actions,
    iter_fantasy_actions,
)
from state.game_state import GameState, N_NORMAL_STREETS

from .policy import Policy


def resample_deck(gs: GameState, rng: random.Random) -> None:
    """Reshuffle the remaining deck cards in-place using `rng`.

    This does NOT change the cards in either player's `pending` or any
    placed/discarded cards — only the order (and hence the future deals)
    of un-dealt deck cards.
    """
    cards = list(gs.deck.cards())
    rng.shuffle(cards)
    # Replace the underlying list (Deck stores by reference under __slots__)
    gs.deck._cards = cards  # type: ignore[attr-defined]


def play_to_terminal(
    gs: GameState,
    policy_p0: Policy,
    policy_p1: Policy,
    *extra_policies: Policy,
) -> GameState:
    """Drive `gs` to terminal using the given per-seat policies (in place).

    Accepts variable arity to support 2- and 3-player games:

        play_to_terminal(gs, p0, p1)         # 2 players
        play_to_terminal(gs, p0, p1, p2)     # 3 players

    The number of supplied policies must equal `gs.n_players`. Returns
    the same `gs` for chaining. Raises if state cannot make progress
    (defensive — should never happen in well-formed states).
    """
    policies: tuple[Policy, ...] = (policy_p0, policy_p1) + tuple(extra_policies)
    if len(policies) != gs.n_players:
        raise ValueError(
            f"play_to_terminal: got {len(policies)} policies, "
            f"but gs has {gs.n_players} players"
        )
    safety = 0
    safety_cap = 4 * N_NORMAL_STREETS + 8
    while not gs.is_terminal():
        # If all non-finished players have empty pending, deal next street.
        if gs.is_round_complete():
            if gs.current_street >= N_NORMAL_STREETS:
                # No more streets but not terminal — corrupted state.
                raise RuntimeError(
                    "play_to_terminal: out of streets but not terminal"
                )
            gs.deal_street()
        for p in range(gs.n_players):
            if gs.needs_action(p):
                gs.step(p, policies[p].act(gs, p))
        safety += 1
        if safety > safety_cap:
            raise RuntimeError(
                "play_to_terminal: exceeded safety iteration cap"
            )
    return gs


def legal_actions(
    gs: GameState,
    player: int,
    fantasy_budget: int = 4096,
) -> list[Action]:
    """Enumerate legal actions for `player` at the current decision point.

    Dispatches by street and fantasy tier. For fantasy hands the action
    space is huge; the `fantasy_budget` truncates enumeration. Callers
    that want exact fantasy play should use `FantasySolverPolicy` instead.
    """
    hs = gs.hands[player]
    cards = list(hs.pending)
    if not cards:
        raise RuntimeError(
            f"legal_actions: player {player} has no pending cards"
        )
    if hs.fantasy_tier != FantasyTier.NORMAL:
        return list(iter_fantasy_actions(cards, hs.board, budget=fantasy_budget))
    if gs.current_street == 1:
        return enumerate_initial_actions(cards)
    return enumerate_pineapple_actions(cards, hs.board)


__all__ = [
    "resample_deck",
    "play_to_terminal",
    "legal_actions",
]
