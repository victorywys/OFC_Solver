import pytest

from engine.cards import parse_cards
from state.action import (
    Action,
    enumerate_initial_actions,
    enumerate_pineapple_actions,
    iter_fantasy_actions,
)
from state.board import (
    PlayerBoard,
    SLOT_BOTTOM,
    SLOT_DISCARD,
    SLOT_MIDDLE,
    SLOT_TOP,
    TOTAL_PLACED,
)


# ---------- PlayerBoard ----------
def test_empty_board_capacities():
    b = PlayerBoard()
    assert b.free_top() == 3
    assert b.free_middle() == 5
    assert b.free_bottom() == 5
    assert b.total_placed() == 0
    assert not b.is_full()


def test_place_and_full():
    b = PlayerBoard()
    cards = parse_cards("As Ad Ac Kh Ks 2c 3c 4c 5c 6c 2d 3d 4d")
    # top: As Ad Ac (3); middle: Kh Ks 2c 3c 4c (5); bottom: 5c 6c 2d 3d 4d (5)
    for c in cards[:3]:
        b.place(c, SLOT_TOP)
    for c in cards[3:8]:
        b.place(c, SLOT_MIDDLE)
    for c in cards[8:]:
        b.place(c, SLOT_BOTTOM)
    assert b.is_full()
    assert b.total_placed() == TOTAL_PLACED


def test_place_overflow_raises():
    b = PlayerBoard()
    for c in parse_cards("2c 3c 4c"):
        b.place(c, SLOT_TOP)
    with pytest.raises(ValueError):
        b.place(parse_cards("5c")[0], SLOT_TOP)


def test_clone_independent():
    b = PlayerBoard()
    b.place(parse_cards("As")[0], SLOT_TOP)
    c = b.clone()
    c.place(parse_cards("Kd")[0], SLOT_TOP)
    assert len(b.top) == 1
    assert len(c.top) == 2


def test_to_final_board_requires_full():
    b = PlayerBoard()
    with pytest.raises(ValueError):
        b.to_final_board()


# ---------- enumerate_initial_actions ----------
def test_initial_action_count():
    cards = parse_cards("As Kd Qc Jh Ts")
    acts = enumerate_initial_actions(cards)
    # 232 capacity-respecting assignments (3^5 = 243 minus 11 invalid where
    # >3 cards on top: a=4 -> C(5,4)*2 = 10, a=5 -> 1 -> total 11)
    assert len(acts) == 232


def test_initial_action_uses_all_cards():
    cards = parse_cards("As Kd Qc Jh Ts")
    acts = enumerate_initial_actions(cards)
    cs = set(cards)
    for a in acts:
        placed = [c for c, _ in a.placements]
        assert set(placed) == cs
        assert len(placed) == 5


def test_initial_action_no_discards():
    cards = parse_cards("As Kd Qc Jh Ts")
    acts = enumerate_initial_actions(cards)
    for a in acts:
        for _, s in a.placements:
            assert s != SLOT_DISCARD


def test_initial_action_apply():
    cards = parse_cards("As Kd Qc Jh Ts")
    acts = enumerate_initial_actions(cards)
    # all should produce a valid in-progress board
    for a in acts:
        b = PlayerBoard()
        a.apply_inplace(b)
        assert b.total_placed() == 5


# ---------- enumerate_pineapple_actions ----------
def test_pineapple_action_count_empty_board():
    # empty board: free top=3, middle=5, bottom=5 -> all 9 placements legal
    # for each of 3 discard choices -> 27 actions
    cards = parse_cards("9c 8c 7c")
    acts = enumerate_pineapple_actions(cards, PlayerBoard())
    assert len(acts) == 27


def test_pineapple_action_capacity_pruning():
    # board with top full -> top no longer legal placement
    b = PlayerBoard()
    for c in parse_cards("2c 3d 4h"):
        b.place(c, SLOT_TOP)
    cards = parse_cards("9c 8c 7c")
    acts = enumerate_pineapple_actions(cards, b)
    for a in acts:
        for c, s in a.placements:
            assert s != SLOT_TOP or False  # no top placements allowed
    # for each discard choice (3), 2 cards each placed in {M,B} -> 4 combos
    # so total = 3 * 4 = 12
    assert len(acts) == 12


def test_pineapple_action_exactly_one_discard():
    cards = parse_cards("9c 8c 7c")
    acts = enumerate_pineapple_actions(cards, PlayerBoard())
    for a in acts:
        n_disc = sum(1 for _, s in a.placements if s == SLOT_DISCARD)
        assert n_disc == 1


# ---------- iter_fantasy_actions ----------
def test_fantasy_actions_yield_at_least_one():
    cards = parse_cards("As Ks Qs Js Ts 9s 8s 7s 6s 5s 4s 3s 2s 2c")  # 14 cards
    it = iter_fantasy_actions(cards, PlayerBoard(), budget=10)
    acts = list(it)
    assert len(acts) == 10
    for a in acts:
        n_placed = sum(1 for _, s in a.placements if s != SLOT_DISCARD)
        n_disc = sum(1 for _, s in a.placements if s == SLOT_DISCARD)
        assert n_placed == TOTAL_PLACED
        assert n_disc == 1


def test_fantasy_actions_use_all_cards():
    cards = parse_cards("As Ks Qs Js Ts 9s 8s 7s 6s 5s 4s 3s 2s 2c")
    a = next(iter_fantasy_actions(cards, PlayerBoard(), budget=1))
    assert {c for c, _ in a.placements} == set(cards)
