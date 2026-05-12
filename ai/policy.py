"""Policy interface.

A policy maps a (game_state, player) pair to an `Action`. All AI components
implement this interface so they can be composed: e.g., MCTS uses a heuristic
policy as its rollout completion policy and as a prior for action selection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from state.action import Action
from state.game_state import GameState


class Policy(ABC):
    """Abstract base for a player policy."""

    name: str = "policy"

    @abstractmethod
    def act(self, gs: GameState, player: int) -> Action:
        """Choose an action for `player` given the current `gs`.

        Preconditions:
            - gs.hands[player].pending must be non-empty
            - gs.hands[player] must not be finished

        Returns:
            A legal Action whose card set equals the player's pending hand
            and whose discard count is correct for the current street.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


__all__ = ["Policy"]
