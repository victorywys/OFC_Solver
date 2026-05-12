"""Game state for a 2-player Pineapple OFC hand.

Streets (per the spec's "5 streets" math: 5 + 4*2 = 13 placed cards):
    Street 1: deal 5, place all 5
    Streets 2..5: deal 3, place 2, discard 1

Fantasyland:
    A player whose previous-hand top row entered fantasy starts the new hand
    with N cards (N in {14,15,16,17}) and finishes the entire 13-card layout
    in a single action.

Turn structure:
    For each street:
        - For each non-finished player, deal them their cards (in the order
          P0 then P1 — order is deterministic given seed).
        - The two players' actions are independent; a `step()` consumes one
          (player, action) pair.

This module is a *thin orchestrator*. It does not embed AI logic — that
lives in `/ai`. It also does not score until the hand ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine.deck import Deck
from engine.fantasy import FantasyTier, next_fantasy_tier
from engine.royalties import DEFAULT_ROYALTIES, RoyaltyConfig
from engine.scoring import ScoreBreakdown, score_match

from .action import Action
from .board import PlayerBoard, SLOT_DISCARD


# Default seat count. The engine supports N ∈ {2, 3} via
# `GameState.new(n_players=...)` and the `hands` tuple. Self-play / table
# infrastructure was historically calibrated to 2 players; see callsites.
N_PLAYERS = 2
MAX_PLAYERS = 3
N_NORMAL_STREETS = 5  # 1 + 4 (1 deal-of-5, 4 deals-of-3)


@dataclass
class HandState:
    """Per-player live hand state."""

    board: PlayerBoard = field(default_factory=PlayerBoard)
    # Cards currently in hand (already dealt to this player but not yet
    # consigned to the board via an Action).
    pending: list[int] = field(default_factory=list)
    fantasy_tier: FantasyTier = FantasyTier.NORMAL
    finished: bool = False  # board full

    def clone(self) -> "HandState":
        return HandState(
            board=self.board.clone(),
            pending=list(self.pending),
            fantasy_tier=self.fantasy_tier,
            finished=self.finished,
        )


@dataclass
class GameState:
    """Pineapple OFC game state for 2 or 3 players.

    The state machine has these transitions:
        deal_street() -> populates `pending` for each non-finished player
        legal_actions(player) -> list/iterator of Actions for that player
        step(player, action) -> applies the action to that player's board
        is_terminal() -> all players have full boards (or all finished)
        score() -> 2-player ScoreBreakdown from P0 perspective (only
                   valid when n_players == 2). For N > 2, use
                   `score_each()` which returns one signed total per
                   player via summed pairwise scoring.
    """

    deck: Deck
    royalty_cfg: RoyaltyConfig = DEFAULT_ROYALTIES
    hands: tuple[HandState, ...] = field(
        default_factory=lambda: (HandState(), HandState())
    )
    # Current street number (1-indexed). For fantasy players this is logical
    # only — a fantasy player gets their entire deal on street 1.
    current_street: int = 0  # 0 means "not yet started"

    @property
    def n_players(self) -> int:
        return len(self.hands)

    # ------- lifecycle -------
    @classmethod
    def new(
        cls,
        seed: int | None = None,
        royalty_cfg: RoyaltyConfig = DEFAULT_ROYALTIES,
        fantasy_p0: FantasyTier = FantasyTier.NORMAL,
        fantasy_p1: FantasyTier = FantasyTier.NORMAL,
        *,
        n_players: int = 2,
        fantasy_tiers: tuple[FantasyTier, ...] | None = None,
    ) -> "GameState":
        """Construct a fresh game.

        For 2 players the legacy `fantasy_p0` / `fantasy_p1` keyword
        arguments are honoured. For 3+ players (or to override) pass
        `fantasy_tiers` as a length-N tuple.
        """
        if not (2 <= n_players <= MAX_PLAYERS):
            raise ValueError(
                f"n_players must be in [2, {MAX_PLAYERS}]; got {n_players}"
            )
        gs = cls(
            deck=Deck(seed=seed),
            royalty_cfg=royalty_cfg,
            hands=tuple(HandState() for _ in range(n_players)),
        )
        if fantasy_tiers is None:
            tiers = [FantasyTier.NORMAL] * n_players
            tiers[0] = fantasy_p0
            if n_players >= 2:
                tiers[1] = fantasy_p1
        else:
            if len(fantasy_tiers) != n_players:
                raise ValueError(
                    f"fantasy_tiers length {len(fantasy_tiers)} != "
                    f"n_players {n_players}"
                )
            tiers = list(fantasy_tiers)
        for hs, t in zip(gs.hands, tiers):
            hs.fantasy_tier = t
        return gs

    # ------- dealing -------
    def deal_street(self) -> None:
        """Advance to the next street and deal cards to non-finished players.

        For street 1: deals 5 to each normal player; deals fantasy_tier.n_cards
            to each fantasy player and they will place all in a single action.
        For streets 2..5: deals 3 to each non-finished normal player.
        """
        if self.current_street >= N_NORMAL_STREETS:
            raise RuntimeError("no more streets to deal")
        self.current_street += 1
        for hs in self.hands:
            if hs.finished:
                continue
            if hs.pending:
                raise RuntimeError(
                    f"player still has {len(hs.pending)} pending cards; "
                    "cannot deal next street"
                )
            if self.current_street == 1:
                if hs.fantasy_tier == FantasyTier.NORMAL:
                    hs.pending = self.deck.deal(5)
                else:
                    hs.pending = self.deck.deal(hs.fantasy_tier.n_cards)
            else:
                if hs.fantasy_tier != FantasyTier.NORMAL:
                    # Fantasy players finish in one action on street 1; should
                    # already be `finished` by now.
                    continue
                hs.pending = self.deck.deal(3)

    # ------- queries -------
    def needs_action(self, player: int) -> bool:
        return not self.hands[player].finished and bool(self.hands[player].pending)

    def is_round_complete(self) -> bool:
        """All non-finished players have acted on their pending cards."""
        return all(
            hs.finished or not hs.pending
            for hs in self.hands
        )

    def is_terminal(self) -> bool:
        return all(hs.finished for hs in self.hands)

    # ------- transitions -------
    def step(self, player: int, action: Action) -> None:
        """Apply an action to one player's hand.

        Validates that:
            - The player has pending cards.
            - The action's placements exactly cover the pending cards.
            - The action respects row capacities (PlayerBoard.place enforces).
            - For pineapple streets, exactly 1 discard.
            - For fantasy, exactly (n_pending - 13) discards.
        """
        hs = self.hands[player]
        if hs.finished:
            raise RuntimeError(f"player {player} already finished")
        if not hs.pending:
            raise RuntimeError(f"player {player} has no pending cards")

        pending_set = set(hs.pending)
        action_cards = [c for c, _ in action.placements]
        if len(action_cards) != len(hs.pending) or set(action_cards) != pending_set:
            raise ValueError(
                f"action cards {sorted(action_cards)} do not match "
                f"pending {sorted(hs.pending)}"
            )

        # discard count check
        n_disc = sum(1 for _, s in action.placements if s == SLOT_DISCARD)
        n_pending = len(hs.pending)
        if hs.fantasy_tier != FantasyTier.NORMAL:
            expected_disc = n_pending - 13
        elif self.current_street == 1:
            expected_disc = 0
        else:
            expected_disc = 1
        if n_disc != expected_disc:
            raise ValueError(
                f"action has {n_disc} discards; expected {expected_disc}"
            )

        action.apply_inplace(hs.board)
        hs.pending = []
        if hs.board.is_full():
            hs.finished = True

    # ------- scoring & cloning -------
    def score(self) -> ScoreBreakdown:
        """2-player head-to-head score from P0 perspective.

        Only valid when `n_players == 2`. For N > 2, use `score_each()`
        which returns one signed total per player via summed pairwise
        scoring.
        """
        if not self.is_terminal():
            raise RuntimeError("score() called on non-terminal state")
        if self.n_players != 2:
            raise RuntimeError(
                f"score() requires 2 players; got {self.n_players}. "
                "Use score_each() instead."
            )
        a = self.hands[0].board.to_final_board()
        b = self.hands[1].board.to_final_board()
        return score_match(a, b, self.royalty_cfg)

    def score_each(self) -> tuple[int, ...]:
        """Per-player signed total via summed pairwise scoring.

        For N players, player `i` plays a head-to-head match against
        each other player `j`; their total is the sum of those
        breakdowns' `total_a` (from `i`'s perspective). For N == 2 this
        equals `(score().total_a, -score().total_a)`.
        """
        if not self.is_terminal():
            raise RuntimeError("score_each() called on non-terminal state")
        boards = [hs.board.to_final_board() for hs in self.hands]
        n = len(boards)
        totals = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                sb = score_match(boards[i], boards[j], self.royalty_cfg)
                totals[i] += sb.total_a
                totals[j] -= sb.total_a
        return tuple(totals)

    def fouls(self) -> tuple[bool, ...]:
        """Per-player foul flags at the terminal state."""
        if not self.is_terminal():
            raise RuntimeError("fouls() called on non-terminal state")
        return tuple(
            not hs.board.to_final_board().is_valid() for hs in self.hands
        )

    def next_fantasy_tiers(self) -> tuple[FantasyTier, ...]:
        """The fantasy tiers each player would carry into the *next* hand.

        Returns a tuple of length `n_players`. Only meaningful on terminal
        (and non-fouled) boards; fouls drop to NORMAL automatically since
        maintenance/entry can't be evaluated on an invalid board.
        """
        if not self.is_terminal():
            raise RuntimeError("next_fantasy_tiers() called on non-terminal state")
        out: list[FantasyTier] = []
        for hs in self.hands:
            board = hs.board.to_final_board()
            if not board.is_valid():
                out.append(FantasyTier.NORMAL)
                continue
            top, _, bot = board.evaluate()
            out.append(next_fantasy_tier(hs.fantasy_tier, top, bot))
        return tuple(out)

    def clone(self) -> "GameState":
        # Deck has its own RNG; we deep-copy via reset/state. For exact
        # reproducibility callers should manage seeds at the harness level.
        new_deck = Deck.__new__(Deck)
        new_deck._cards = list(self.deck._cards)  # type: ignore[attr-defined]
        import random as _random

        new_deck._rng = _random.Random()  # type: ignore[attr-defined]
        new_deck._rng.setstate(self.deck._rng.getstate())  # type: ignore[attr-defined]
        return GameState(
            deck=new_deck,
            royalty_cfg=self.royalty_cfg,
            hands=tuple(hs.clone() for hs in self.hands),
            current_street=self.current_street,
        )


__all__ = ["GameState", "HandState", "N_PLAYERS", "MAX_PLAYERS", "N_NORMAL_STREETS"]
