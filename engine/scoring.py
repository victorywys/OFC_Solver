"""Two-player row-vs-row scoring with scoop bonus and foul handling.

Conventions:
    - All scores are signed from player A's perspective.
    - score(A, B) = total points A earns minus total points B earns.
    - Royalty values come from a `RoyaltyConfig` (default = user spec).

Joker awareness: when a board contains jokers, `Board.evaluate()` returns
the joint, foul-aware joker resolution from `engine.rules.resolve_board`,
i.e. the player's optimal substitution choice (non-fouling, max royalty).
This means `evaluate()` and `is_valid()` here may differ from the naive
per-row evaluators on boards that contain jokers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .evaluator import HandRank, evaluate_3, evaluate_5
from .royalties import (
    DEFAULT_ROYALTIES,
    RoyaltyConfig,
    royalty_bottom,
    royalty_middle,
    royalty_top,
)


@dataclass(frozen=True)
class Board:
    """Final 13-card placement for one player. May be foul."""

    top: tuple[int, ...]      # 3 cards
    middle: tuple[int, ...]   # 5 cards
    bottom: tuple[int, ...]   # 5 cards

    def __post_init__(self) -> None:
        if len(self.top) != 3 or len(self.middle) != 5 or len(self.bottom) != 5:
            raise ValueError("board rows must be 3/5/5")

    def evaluate(self) -> tuple[HandRank, HandRank, HandRank]:
        """Joker-aware joint resolution. See `engine.rules.resolve_board`."""
        # Local import to avoid a top-level cycle (rules imports royalties,
        # royalties imports evaluator; scoring already imports both).
        from .rules import resolve_board
        t, m, b, _ = resolve_board(self.top, self.middle, self.bottom)
        return t, m, b

    def is_valid(self) -> bool:
        """True iff some joker assignment yields a non-fouled layout."""
        from .rules import resolve_board
        return not resolve_board(self.top, self.middle, self.bottom)[3]


@dataclass(frozen=True)
class ScoreBreakdown:
    """Detail of a head-to-head scoring event (A vs B)."""

    a_royalties: int
    b_royalties: int
    line_score_a: int           # +1/0/-1 sums per row
    scoop_bonus_a: int          # +3 if A scoops, -3 if B scoops, else 0
    a_foul: bool
    b_foul: bool

    @property
    def total_a(self) -> int:
        return self.line_score_a + self.scoop_bonus_a + self.a_royalties - self.b_royalties

    @property
    def total_b(self) -> int:
        return -self.total_a


def _row_royalties(b: Board, cfg: RoyaltyConfig) -> tuple[int, int, int]:
    t, m, bot = b.evaluate()
    return royalty_top(t, cfg), royalty_middle(m, cfg), royalty_bottom(bot, cfg)


def total_royalties(board: Board, cfg: RoyaltyConfig = DEFAULT_ROYALTIES) -> int:
    """Royalty total for a non-foul board. (Foul boards conventionally score 0
    royalties; opponent gets line+scoop only. Some variants double opposing
    royalties on foul; configure outside if needed.)"""
    if not board.is_valid():
        return 0
    return sum(_row_royalties(board, cfg))


def score_match(
    a: Board,
    b: Board,
    cfg: RoyaltyConfig = DEFAULT_ROYALTIES,
) -> ScoreBreakdown:
    """Score one head-to-head matchup; returns A-perspective breakdown.

    Per spec:
        - row winner gets +1 per row, loser -1, tie 0
        - scoop (winning all 3 rows) doubles the score (we model as +3 line
          + +3 scoop bonus = +6 net swing per spec "doubled"; equivalently
          treat scoop bonus = +3 on top of +3 line)
        - foul: opponent receives the doubled non-foul royalties + sweep all rows
    """
    a_foul = not a.is_valid()
    b_foul = not b.is_valid()

    if a_foul and b_foul:
        return ScoreBreakdown(0, 0, 0, 0, True, True)

    if a_foul:
        # opponent sweeps all 3 rows and collects their royalties
        b_roy = total_royalties(b, cfg)
        # Spec: "opponent receives doubled row scores" -> +6 line+scoop swing for B
        return ScoreBreakdown(
            a_royalties=0,
            b_royalties=b_roy,
            line_score_a=-3,
            scoop_bonus_a=-3,  # scoop doubles to -6 for A
            a_foul=True,
            b_foul=False,
        )
    if b_foul:
        a_roy = total_royalties(a, cfg)
        return ScoreBreakdown(
            a_royalties=a_roy,
            b_royalties=0,
            line_score_a=+3,
            scoop_bonus_a=+3,
            a_foul=False,
            b_foul=True,
        )

    at, am, ab = a.evaluate()
    bt, bm, bb = b.evaluate()

    line = 0
    line += (1 if at > bt else -1 if at < bt else 0)
    line += (1 if am > bm else -1 if am < bm else 0)
    line += (1 if ab > bb else -1 if ab < bb else 0)

    scoop = 0
    if at > bt and am > bm and ab > bb:
        scoop = +3
    elif at < bt and am < bm and ab < bb:
        scoop = -3

    a_roy = sum(_row_royalties(a, cfg))
    b_roy = sum(_row_royalties(b, cfg))

    return ScoreBreakdown(
        a_royalties=a_roy,
        b_royalties=b_roy,
        line_score_a=line,
        scoop_bonus_a=scoop,
        a_foul=False,
        b_foul=False,
    )


__all__ = ["Board", "ScoreBreakdown", "total_royalties", "score_match"]
