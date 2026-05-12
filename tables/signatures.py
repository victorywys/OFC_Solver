"""Canonical signatures for state, action, and hand keys.

Why we need this
----------------
Every precomputed table (foul-prob, policy-prior, opening book, ...) needs
a *stable* dictionary key for "this state". Two states that differ only
in card-order within a row should hash to the same key, otherwise the
table fragments badly.

We adopt:

    * row content = `tuple(sorted(cards))`
    * state_signature(board, tier, street) =
          (sorted_top, sorted_middle, sorted_bottom, sorted_discards,
           tier_int, street_int)
    * action_signature(action) =
          tuple(sorted((card_id, slot_id))) — order-independent
    * canonical_action(action) is the same as action_signature; the slot
      itself disambiguates which card went where.

Discards are included in state signatures because two boards with the
same rows but different discards have different residual deck content
and therefore different futures.
"""

from __future__ import annotations

from typing import Sequence

from state.action import Action
from state.board import PlayerBoard
from state.game_state import GameState


# A compact tuple type. Every component is a hashable primitive tuple.
StateSignature = tuple
ActionSignature = tuple


def _sorted_tuple(cards: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(cards))


# ---------------------------------------------------------------------------
# state signatures
# ---------------------------------------------------------------------------
def state_signature(
    board: PlayerBoard,
    tier: int,
    street: int,
    pending: Sequence[int] = (),
) -> StateSignature:
    """Canonical key for a player's situation at decision time.

    Includes:
        * sorted contents of each row + discards
        * fantasy tier (int)
        * current street (int)
        * sorted pending hand (the cards the player is about to place)

    Two states with the same signature are operationally equivalent for
    decision-making purposes.
    """
    return (
        _sorted_tuple(board.top),
        _sorted_tuple(board.middle),
        _sorted_tuple(board.bottom),
        _sorted_tuple(board.discards),
        int(tier),
        int(street),
        _sorted_tuple(pending),
    )


def turn_state_signature(turn) -> StateSignature:
    """Build a state signature from a recorded `simulation.trace.Turn`.

    Avoids reconstructing a `PlayerBoard`. Equivalent output to
    `state_signature(<reconstructed board>, tier, street, pending)`.
    """
    return (
        _sorted_tuple(turn.board_top),
        _sorted_tuple(turn.board_middle),
        _sorted_tuple(turn.board_bottom),
        _sorted_tuple(turn.board_discards),
        int(turn.fantasy_tier),
        int(turn.street),
        _sorted_tuple(turn.pending),
    )


def gamestate_signature(gs: GameState, player: int) -> StateSignature:
    """Build a state signature from a live `GameState`. For runtime lookups."""
    hs = gs.hands[player]
    return state_signature(
        hs.board,
        int(hs.fantasy_tier),
        gs.current_street,
        hs.pending,
    )


# ---------------------------------------------------------------------------
# action signatures
# ---------------------------------------------------------------------------
def action_signature(action: Action) -> ActionSignature:
    """Order-independent key for an `Action`."""
    return tuple(sorted(action.placements))


def canonical_action(placements) -> ActionSignature:
    """Same as `action_signature` but operates on a placements tuple."""
    return tuple(sorted(placements))


# ---------------------------------------------------------------------------
# fantasy / opening keys
# ---------------------------------------------------------------------------
def fantasy_hand_signature(cards: Sequence[int], tier: int) -> tuple:
    """Key for the fantasy arrangement cache."""
    return (_sorted_tuple(cards), int(tier))


def street1_hand_signature(cards: Sequence[int]) -> tuple[int, ...]:
    """Key for the opening book — exactly 5 cards on street 1."""
    if len(cards) != 5:
        raise ValueError(
            f"street1_hand_signature: expected 5 cards, got {len(cards)}"
        )
    return _sorted_tuple(cards)


__all__ = [
    "StateSignature",
    "ActionSignature",
    "state_signature",
    "turn_state_signature",
    "gamestate_signature",
    "action_signature",
    "canonical_action",
    "fantasy_hand_signature",
    "street1_hand_signature",
]
