"""Unit tests for `ai/rollout.py` primitives."""

from __future__ import annotations

import random

from ai.heuristic_policy import HeuristicPolicy
from ai.random_policy import RandomPolicy
from ai.rollout import legal_actions, play_to_terminal, resample_deck
from engine.cards import NUM_CARDS
from state.game_state import GameState, N_NORMAL_STREETS


# ------------------------- resample_deck -------------------------
def test_resample_deck_preserves_card_set():
    gs = GameState.new(seed=1)
    gs.deal_street()
    before = sorted(gs.deck.cards())
    rng = random.Random(123)
    resample_deck(gs, rng)
    after = sorted(gs.deck.cards())
    assert before == after


def test_resample_deck_changes_order_for_distinct_seeds():
    gs1 = GameState.new(seed=1)
    gs2 = gs1.clone()
    gs1.deal_street()
    gs2.deal_street()
    resample_deck(gs1, random.Random(11))
    resample_deck(gs2, random.Random(22))
    # Same multiset, different order (with overwhelming probability).
    assert sorted(gs1.deck.cards()) == sorted(gs2.deck.cards())
    assert list(gs1.deck.cards()) != list(gs2.deck.cards())


def test_resample_deck_does_not_touch_pending():
    gs = GameState.new(seed=4)
    gs.deal_street()
    p0_pending_before = list(gs.hands[0].pending)
    resample_deck(gs, random.Random(99))
    assert gs.hands[0].pending == p0_pending_before


# ------------------------- play_to_terminal -------------------------
def test_play_to_terminal_reaches_terminal_with_random():
    gs = GameState.new(seed=7)
    p0 = RandomPolicy(seed=10)
    p1 = RandomPolicy(seed=20)
    play_to_terminal(gs, p0, p1)
    assert gs.is_terminal()
    # 13 placed + N discards on each side, with all cards conserved.
    for hs in gs.hands:
        assert hs.board.is_full()


def test_play_to_terminal_produces_valid_score():
    gs = GameState.new(seed=99)
    play_to_terminal(gs, HeuristicPolicy(seed=1), HeuristicPolicy(seed=2))
    assert gs.is_terminal()
    sb = gs.score()
    # total_a + total_b == 0 (zero sum)
    assert sb.total_a + sb.total_b == 0


def test_play_to_terminal_card_conservation():
    """Each card appears at most once across both players' boards+discards."""
    gs = GameState.new(seed=42)
    play_to_terminal(gs, HeuristicPolicy(seed=1), HeuristicPolicy(seed=2))
    seen: set[int] = set()
    for hs in gs.hands:
        for row in hs.board.rows:
            for c in row:
                assert c not in seen
                seen.add(c)
        for c in hs.board.discards:
            assert c not in seen
            seen.add(c)
    # Total placed = 26; total discards = 8 (4 streets x 2 players).
    assert len(seen) == 26 + 8
    # Every card observed must be valid.
    for c in seen:
        assert 0 <= c < NUM_CARDS


def test_play_to_terminal_idempotent_on_terminal_state():
    gs = GameState.new(seed=3)
    play_to_terminal(gs, HeuristicPolicy(seed=1), HeuristicPolicy(seed=2))
    # Calling again on a terminal state should be a no-op.
    play_to_terminal(gs, HeuristicPolicy(seed=1), HeuristicPolicy(seed=2))
    assert gs.is_terminal()


# ------------------------- legal_actions -------------------------
def test_legal_actions_initial_street():
    gs = GameState.new(seed=2)
    gs.deal_street()
    acts = legal_actions(gs, 0)
    # Initial street: 5 cards across 3 rows -> 232 capacity-respecting layouts
    assert len(acts) == 232
    for a in acts:
        # No discards on street 1
        assert a.discards() == ()
        cs = sorted(c for c, _ in a.placements)
        assert cs == sorted(gs.hands[0].pending)


def test_legal_actions_pineapple_street():
    gs = GameState.new(seed=3)
    gs.deal_street()
    # Place all 5 cards somewhere legal so that street 2 is reachable.
    gs.step(0, RandomPolicy(seed=1).act(gs, 0))
    gs.step(1, RandomPolicy(seed=2).act(gs, 1))
    gs.deal_street()
    acts = legal_actions(gs, 0)
    # Pineapple street: 3 cards -> at most 27 actions; depending on board
    # capacity occupied by street 1, some may be pruned. Bound it loosely.
    assert 1 <= len(acts) <= 27
    for a in acts:
        # Exactly 1 discard
        assert len(a.discards()) == 1
