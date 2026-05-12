import pytest

from engine.fantasy import FantasyTier
from state.action import (
    Action,
    enumerate_initial_actions,
    enumerate_pineapple_actions,
)
from state.game_state import GameState, N_NORMAL_STREETS
from state.board import SLOT_BOTTOM, SLOT_DISCARD, SLOT_MIDDLE, SLOT_TOP


def _greedy_play_one_player(gs: GameState, player: int) -> None:
    """Play through all 5 streets for `player` using a deterministic greedy
    rule: place each card into the lowest non-full row, discard last card on
    pineapple streets."""
    for street in range(1, N_NORMAL_STREETS + 1):
        if gs.current_street < street:
            gs.deal_street()
        hs = gs.hands[player]
        if hs.finished:
            continue
        cards = list(hs.pending)

        if street == 1:
            placements: list[tuple[int, int]] = []
            for c in cards:
                if hs.board.free_top() > 0 and len([p for p in placements if p[1] == SLOT_TOP]) < hs.board.free_top():
                    placements.append((c, SLOT_TOP))
                elif hs.board.free_middle() > 0 and len([p for p in placements if p[1] == SLOT_MIDDLE]) < hs.board.free_middle():
                    placements.append((c, SLOT_MIDDLE))
                else:
                    placements.append((c, SLOT_BOTTOM))
            gs.step(player, Action(tuple(placements)))
        else:
            # discard last; place first two greedily
            kept = cards[:2]
            disc = cards[2]
            placements = []
            for c in kept:
                if hs.board.free_bottom() > 0 and sum(1 for p in placements if p[1] == SLOT_BOTTOM) < hs.board.free_bottom():
                    placements.append((c, SLOT_BOTTOM))
                elif hs.board.free_middle() > 0 and sum(1 for p in placements if p[1] == SLOT_MIDDLE) < hs.board.free_middle():
                    placements.append((c, SLOT_MIDDLE))
                else:
                    placements.append((c, SLOT_TOP))
            placements.append((disc, SLOT_DISCARD))
            gs.step(player, Action(tuple(placements)))


def test_full_hand_runs_to_terminal():
    gs = GameState.new(seed=123)
    for street in range(1, N_NORMAL_STREETS + 1):
        gs.deal_street()
        for p in (0, 1):
            hs = gs.hands[p]
            assert hs.pending  # cards waiting
            cards = list(hs.pending)
            # use enumerate_*_actions then pick the first
            if street == 1:
                acts = enumerate_initial_actions(cards)
            else:
                acts = enumerate_pineapple_actions(cards, hs.board)
            assert acts
            gs.step(p, acts[0])
    assert gs.is_terminal()
    sb = gs.score()
    assert isinstance(sb.total_a, int)


def test_each_card_used_in_action_must_match_pending():
    gs = GameState.new(seed=1)
    gs.deal_street()
    hs = gs.hands[0]
    bad = Action(((0, SLOT_TOP), (1, SLOT_TOP), (2, SLOT_TOP), (3, SLOT_MIDDLE), (4, SLOT_MIDDLE)))
    with pytest.raises(ValueError):
        # likely doesn't match dealt cards
        gs.step(0, bad)


def test_pineapple_street_requires_one_discard():
    gs = GameState.new(seed=2)
    gs.deal_street()
    # play street 1 minimally for both
    for p in (0, 1):
        cards = list(gs.hands[p].pending)
        gs.step(p, enumerate_initial_actions(cards)[0])
    gs.deal_street()
    # build an invalid action with 0 discards (place all 3)
    cards = list(gs.hands[0].pending)
    invalid = Action((
        (cards[0], SLOT_BOTTOM),
        (cards[1], SLOT_BOTTOM),
        (cards[2], SLOT_BOTTOM),
    ))
    with pytest.raises(ValueError):
        gs.step(0, invalid)


def test_fantasy_player_dealt_correct_cards():
    gs = GameState.new(seed=7, fantasy_p0=FantasyTier.F14, fantasy_p1=FantasyTier.NORMAL)
    gs.deal_street()
    assert len(gs.hands[0].pending) == 14
    assert len(gs.hands[1].pending) == 5


def test_dealing_is_deterministic_with_seed():
    a = GameState.new(seed=42)
    a.deal_street()
    b = GameState.new(seed=42)
    b.deal_street()
    assert list(a.hands[0].pending) == list(b.hands[0].pending)
    assert list(a.hands[1].pending) == list(b.hands[1].pending)


def test_no_double_deal_until_step():
    gs = GameState.new(seed=3)
    gs.deal_street()
    with pytest.raises(RuntimeError):
        gs.deal_street()  # both players still have pending cards


def test_clone_isolated():
    gs = GameState.new(seed=10)
    gs.deal_street()
    gs2 = gs.clone()
    cards = list(gs.hands[0].pending)
    gs.step(0, enumerate_initial_actions(cards)[0])
    # clone untouched
    assert len(gs2.hands[0].pending) == 5
    assert gs2.hands[0].board.total_placed() == 0


def test_terminal_score_against_self_play_greedy():
    gs = GameState.new(seed=99)
    # interleave both players so deck consumption is shared
    for street in range(1, N_NORMAL_STREETS + 1):
        gs.deal_street()
        for p in (0, 1):
            hs = gs.hands[p]
            cards = list(hs.pending)
            if street == 1:
                acts = enumerate_initial_actions(cards)
            else:
                acts = enumerate_pineapple_actions(cards, hs.board)
            gs.step(p, acts[0])
    assert gs.is_terminal()
    sb = gs.score()
    # score is signed int; just sanity check
    assert sb.total_a == -sb.total_b
