"""Tests for the fantasy solver."""

import pytest

from engine.cards import parse_cards, cards_str
from engine.evaluator import (
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    ROYAL_FLUSH,
    STRAIGHT_FLUSH,
    TRIPS,
    evaluate_3,
    evaluate_5,
)
from engine.fantasy import FantasyTier
from engine.scoring import Board
from fantasy.cards_mask import ALL_CARDS_MASK, cards_of, mask_of, popcount
from fantasy.fantasy_eval import (
    bottom_guarantees_continuation,
    eval_3_by_mask,
    eval_5_by_mask,
    is_continuation,
    top_guarantees_continuation,
)
from fantasy.fantasy_search import FantasyConfig
from fantasy.fantasy_solver import (
    FantasySolverPolicy,
    default_config_for,
    fantasy_result_to_action,
    solve_fantasy,
)


# ===================== bitmask helpers =====================
def test_mask_of_and_cards_of_roundtrip():
    cards = parse_cards("As Kd Qc 7h 2s")
    m = mask_of(cards)
    assert popcount(m) == 5
    assert sorted(cards_of(m)) == sorted(cards)


def test_all_cards_mask_size():
    assert popcount(ALL_CARDS_MASK) == 54


# ===================== evaluator wrappers =====================
def test_eval_5_by_mask_matches_evaluate_5():
    cards = parse_cards("As Ks Qs Js Ts")
    assert eval_5_by_mask(mask_of(cards)) == evaluate_5(cards)


def test_eval_3_by_mask_matches_evaluate_3():
    cards = parse_cards("Qs Qd 2c")
    assert eval_3_by_mask(mask_of(cards)) == evaluate_3(cards)


# ===================== continuation =====================
def test_continuation_f14_with_qq_top():
    top = evaluate_3(parse_cards("Qs Qd 2c"))
    bot = evaluate_5(parse_cards("As Kd Qc Jh 9s"))
    assert is_continuation(FantasyTier.F14, top, bot)


def test_continuation_f14_with_aa_top_still_maintains():
    # F14 entered via QQ. Even AA on a new fantasy round maintains F14.
    top = evaluate_3(parse_cards("As Ad 2c"))
    bot = evaluate_5(parse_cards("As Kd Qc Jh 9s"))
    assert is_continuation(FantasyTier.F14, top, bot)


def test_continuation_f17_only_with_trips_or_quads_bottom():
    top_pair = evaluate_3(parse_cards("Ks Kd 2c"))
    bot_weak = evaluate_5(parse_cards("As Kd Qc Jh 9s"))
    assert not is_continuation(FantasyTier.F17, top_pair, bot_weak)
    bot_quads = evaluate_5(parse_cards("9s 9d 9c 9h Ks"))
    assert is_continuation(FantasyTier.F17, top_pair, bot_quads)


def test_continuation_f16_requires_aa_or_quads():
    top_kk = evaluate_3(parse_cards("Ks Kd 2c"))
    bot_weak = evaluate_5(parse_cards("As Kd Qc Jh 9s"))
    assert not is_continuation(FantasyTier.F16, top_kk, bot_weak)
    top_aa = evaluate_3(parse_cards("As Ad 2c"))
    assert is_continuation(FantasyTier.F16, top_aa, bot_weak)


def test_bottom_quads_guarantees_continuation():
    bot = evaluate_5(parse_cards("9s 9d 9c 9h Ks"))
    assert bottom_guarantees_continuation(bot)
    bot2 = evaluate_5(parse_cards("As Kd Qc Jh 9s"))
    assert not bottom_guarantees_continuation(bot2)


def test_top_guarantees_continuation_per_tier():
    qq = evaluate_3(parse_cards("Qs Qd 2c"))
    kk = evaluate_3(parse_cards("Ks Kd 2c"))
    aa = evaluate_3(parse_cards("As Ad 2c"))
    trips = evaluate_3(parse_cards("2s 2d 2c"))
    assert top_guarantees_continuation(FantasyTier.F14, qq)
    assert top_guarantees_continuation(FantasyTier.F14, kk)
    assert top_guarantees_continuation(FantasyTier.F14, aa)
    assert top_guarantees_continuation(FantasyTier.F15, kk)
    assert not top_guarantees_continuation(FantasyTier.F15, qq)
    assert top_guarantees_continuation(FantasyTier.F16, aa)
    assert not top_guarantees_continuation(FantasyTier.F16, kk)
    assert top_guarantees_continuation(FantasyTier.F17, trips)


# ===================== solver correctness =====================
def _result_is_valid(result, n_cards: int) -> bool:
    """Check the result is a non-fouled, full layout from `n_cards` inputs."""
    placed = list(result.top) + list(result.middle) + list(result.bottom)
    discards = list(result.discards)
    all_used = placed + discards
    if len(all_used) != n_cards:
        return False
    if len(set(all_used)) != n_cards:
        return False
    if len(result.top) != 3 or len(result.middle) != 5 or len(result.bottom) != 5:
        return False
    b = Board(tuple(result.top), tuple(result.middle), tuple(result.bottom))
    return b.is_valid()


def test_solver_f14_returns_valid_layout():
    # 14-card hand
    cards = parse_cards("As Ad Ac 2c 3c 4c 5c 6c 7c 8c Kh Kd Qh Qd")
    result = solve_fantasy(cards, FantasyTier.F14)
    assert _result_is_valid(result, 14)


def test_solver_finds_strong_bottom_when_available():
    # 4 aces + 9 spades flush + extras -> obvious bottom = 4 aces or royal flush
    cards = parse_cards("As Ks Qs Js Ts 9s 8s 7s 6s 5s 4s 3s 2s 2c")  # 14 cards, all spades + one 2c
    # bottom royal flush (As Ks Qs Js Ts) is the dominant choice
    result = solve_fantasy(cards, FantasyTier.F14)
    assert result.bottom_rank[0] == ROYAL_FLUSH
    assert result.continuation  # bottom RF = SF, which is strictly less than QUADS,
    # so continuation must come from top. Let's verify the layout actually maintains.
    # Actually RF is STRAIGHT_FLUSH+1=ROYAL_FLUSH which IS >= QUADS in our ordering.
    # ROYAL_FLUSH=9, QUADS=7 -> yes, royal flush bottom guarantees continuation.


def test_solver_prefers_continuation_over_immediate_royalty():
    # construct a hand where keeping QQ on top costs some immediate royalty
    # but maintains F14. Solver should choose continuation.
    cards = parse_cards("Qs Qd 2c 3d 4c 5d 6c 7d 8c 9d Th Jh Kc Ac")
    # If QQ stays on top -> F14 maintained. If QQ goes elsewhere, we lose
    # continuation bonus (~20). The QQ pair-on-top royalty alone is 2,
    # whereas the continuation bonus is 20 -> should keep QQ on top.
    result = solve_fantasy(cards, FantasyTier.F14)
    # the top should have a pair of Q or stronger (i.e. continuation must hold)
    assert result.continuation, "solver should maintain continuation when easy"


def test_solver_no_foul_ever():
    # Many varied hands; solver must never produce a foul.
    import random
    rng = random.Random(0)
    for _ in range(20):
        cards = rng.sample(range(54), 14)
        for tier in [FantasyTier.F14]:
            result = solve_fantasy(cards, tier)
            assert _result_is_valid(result, 14)


def test_solver_exact_vs_beam_consistency_on_f14():
    """On F14 (exact is feasible), exact and a generous-beam search should
    return the same EV (or at least be very close)."""
    cards = parse_cards("As Ad 2c 3c 4c 5d 6d 7h 8h 9s Tc Jc Qc Kc")
    exact = solve_fantasy(cards, FantasyTier.F14, FantasyConfig(exact=True))
    beam = solve_fantasy(
        cards,
        FantasyTier.F14,
        FantasyConfig(bottom_beam=200, middle_beam=80),
    )
    # beam should match or be within 5% of exact on F14
    assert beam.ev >= exact.ev - 0.05 * abs(exact.ev) - 1
    assert beam.ev <= exact.ev


def test_continuation_bonus_value_used():
    cards = parse_cards("Qs Qd 2c 3d 4c 5d 6c 7d 8c 9d Th Jh Kc Ac")
    cfg = FantasyConfig(continue_f14=100.0)
    result = solve_fantasy(cards, FantasyTier.F14, cfg)
    if result.continuation:
        assert result.continuation_bonus == 100.0


def test_solver_search_stats_populated():
    cards = parse_cards("As Ad 2c 3c 4c 5d 6d 7h 8h 9s Tc Jc Qc Kc")
    result = solve_fantasy(cards, FantasyTier.F14, FantasyConfig(exact=True))
    assert result.stats.bottoms_considered > 0
    assert result.stats.leaves_evaluated > 0


def test_fantasy_action_conversion():
    cards = parse_cards("As Ad 2c 3c 4c 5d 6d 7h 8h 9s Tc Jc Qc Kc")
    result = solve_fantasy(cards, FantasyTier.F14)
    action = fantasy_result_to_action(result)
    placed_cards = {c for c, _ in action.placements}
    assert placed_cards == set(cards)


def test_fantasy_solver_policy_integrates_with_game_state():
    """End-to-end: a fantasy player is dealt their N cards; the solver policy
    produces an Action; the engine accepts it; the player finishes in 1 step."""
    from ai.heuristic_policy import HeuristicPolicy
    from state.game_state import GameState

    fb = HeuristicPolicy(seed=0)
    pol = FantasySolverPolicy(fallback=fb)

    gs = GameState.new(seed=5, fantasy_p0=FantasyTier.F14)
    gs.deal_street()
    assert len(gs.hands[0].pending) == 14
    a = pol.act(gs, 0)
    gs.step(0, a)
    assert gs.hands[0].finished
    assert gs.hands[0].board.is_full()


def test_solver_handles_fewer_than_13_cards_raises():
    with pytest.raises(ValueError):
        solve_fantasy(parse_cards("As Kd Qc"), FantasyTier.F14)


def test_solver_normal_tier_raises():
    cards = parse_cards("As Ad 2c 3c 4c 5d 6d 7h 8h 9s Tc Jc Qc Kc")
    with pytest.raises(ValueError):
        solve_fantasy(cards, FantasyTier.NORMAL)


def test_solver_with_jokers():
    cards = parse_cards("As Ks Qs Js *1 *2 7s 6s 2c 3c 4c 5d 6d 7h")
    result = solve_fantasy(cards, FantasyTier.F14)
    assert _result_is_valid(result, 14)
