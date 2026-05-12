"""Game record schema for self-play.

A `GameRecord` is a fully-picklable, post-hoc summary of one OFC hand. It
contains enough information for any Phase-6 table builder to operate
without re-running the game.

Two recording levels are supported:

    * **summary-only** (default): final boards + scores + initial/final
      fantasy tiers + seeds + policy names. ~kB per game. Sufficient for
      foul-rate / royalty-distribution / fantasy-transition collectors.
    * **full trace** (`record_turns=True`): all decisions in order, with
      pre-step pending hand + board snapshot + chosen action. ~10kB per
      game. Required for opening-book / state-action prior / partial-board
      foul-probability collectors.

Trace recording is opt-in because it ~10x's the in-memory cost. Most
collectors should declare which level they need so callers can avoid
recording when nobody asks for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine.fantasy import FantasyTier
from engine.scoring import Board, ScoreBreakdown
from state.action import Action
from state.game_state import GameState, N_NORMAL_STREETS

from ai.policy import Policy


# ---------------------------------------------------------------------------
# Per-decision turn record (only present when record_turns=True)
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    """One decision point in the game, snapshotted *before* the action."""

    street: int                              # 1..N_NORMAL_STREETS
    player: int                              # 0 or 1
    fantasy_tier: int                        # int(FantasyTier)
    pending: tuple[int, ...]
    board_top: tuple[int, ...]
    board_middle: tuple[int, ...]
    board_bottom: tuple[int, ...]
    board_discards: tuple[int, ...]
    placements: tuple[tuple[int, int], ...]  # the chosen Action

    def action(self) -> Action:
        return Action(self.placements)


# ---------------------------------------------------------------------------
# Whole-game record
# ---------------------------------------------------------------------------
@dataclass
class GameRecord:
    """All information needed to reconstruct a finished hand."""

    seed: int
    policy_a_name: str
    policy_b_name: str
    initial_tiers: tuple[int, int]   # ints (FantasyTier values)
    final_tiers: tuple[int, int]     # ints (next-hand tiers)

    # Final boards (immutable)
    final_a: Board
    final_b: Board

    # Discards aren't part of `Board`; record separately.
    discards_a: tuple[int, ...]
    discards_b: tuple[int, ...]

    score: ScoreBreakdown

    # Optional full trace
    turns: list[Turn] = field(default_factory=list)

    # ---------- accessors ----------
    @property
    def total_a(self) -> int:
        return self.score.total_a

    @property
    def total_b(self) -> int:
        return -self.score.total_a

    def initial_tier(self, player: int) -> FantasyTier:
        return FantasyTier(self.initial_tiers[player])

    def final_tier(self, player: int) -> FantasyTier:
        return FantasyTier(self.final_tiers[player])

    def board(self, player: int) -> Board:
        return self.final_a if player == 0 else self.final_b

    def is_foul(self, player: int) -> bool:
        return self.score.a_foul if player == 0 else self.score.b_foul


# ---------------------------------------------------------------------------
# Single-game driver with optional full trace
# ---------------------------------------------------------------------------
def simulate_one_game(
    gs: GameState,
    policy_p0: Policy,
    policy_p1: Policy,
    *,
    record_turns: bool = False,
    seed_for_record: Optional[int] = None,
) -> GameRecord:
    """Drive `gs` to terminal, optionally recording each decision.

    The policies' RNG state is *not* reseeded here — callers should set up
    seeds before calling this function. `seed_for_record` is only stored
    in the resulting `GameRecord` for traceability.
    """
    policies = (policy_p0, policy_p1)
    initial_tiers = (
        int(gs.hands[0].fantasy_tier),
        int(gs.hands[1].fantasy_tier),
    )
    turns: list[Turn] = []

    safety = 0
    safety_cap = 4 * N_NORMAL_STREETS + 8
    while not gs.is_terminal():
        if gs.is_round_complete():
            if gs.current_street >= N_NORMAL_STREETS:
                raise RuntimeError(
                    "simulate_one_game: out of streets but not terminal"
                )
            gs.deal_street()
        for p in (0, 1):
            if not gs.needs_action(p):
                continue
            if record_turns:
                hs = gs.hands[p]
                turns.append(
                    Turn(
                        street=gs.current_street,
                        player=p,
                        fantasy_tier=int(hs.fantasy_tier),
                        pending=tuple(hs.pending),
                        board_top=tuple(hs.board.top),
                        board_middle=tuple(hs.board.middle),
                        board_bottom=tuple(hs.board.bottom),
                        board_discards=tuple(hs.board.discards),
                        placements=(),  # filled in below once chosen
                    )
                )
                action = policies[p].act(gs, p)
                # mutate the just-appended turn (Turn is not frozen)
                turns[-1].placements = action.placements
            else:
                action = policies[p].act(gs, p)
            gs.step(p, action)
        safety += 1
        if safety > safety_cap:
            raise RuntimeError(
                "simulate_one_game: exceeded safety iteration cap"
            )

    final_tiers = gs.next_fantasy_tiers()
    record = GameRecord(
        seed=seed_for_record if seed_for_record is not None else -1,
        policy_a_name=getattr(policy_p0, "name", type(policy_p0).__name__),
        policy_b_name=getattr(policy_p1, "name", type(policy_p1).__name__),
        initial_tiers=initial_tiers,
        final_tiers=(int(final_tiers[0]), int(final_tiers[1])),
        final_a=gs.hands[0].board.to_final_board(),
        final_b=gs.hands[1].board.to_final_board(),
        discards_a=tuple(gs.hands[0].board.discards),
        discards_b=tuple(gs.hands[1].board.discards),
        score=gs.score(),
        turns=turns,
    )
    return record


__all__ = ["Turn", "GameRecord", "simulate_one_game"]
