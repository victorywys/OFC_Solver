"""Tests for the `tables` package.

Covers:
    - Welford push / merge / equivalence to all-in-one
    - Signature determinism and order-independence
    - Each Collector: observe / merge / pickle / class-merge guard
    - Each Table: lookup correctness on synthetic data
    - TranspositionTable persistence round-trip
    - TableAwarePolicy: legality + table-hit precedence + fallback
"""

from __future__ import annotations

import pickle
from copy import deepcopy

import pytest

from engine.fantasy import FantasyTier
from simulation.policy_factories import heuristic_factory
from simulation.self_play import SelfPlay, play_game
from simulation.trace import simulate_one_game
from state.action import Action
from state.board import (
    SLOT_BOTTOM,
    SLOT_DISCARD,
    SLOT_MIDDLE,
    SLOT_TOP,
)
from state.game_state import GameState

from ai.heuristic_policy import HeuristicPolicy

from tables import (
    FantasyArrangementCache,
    FantasyArrangementCacheCollector,
    FantasyEVCollector,
    FantasyEVTable,
    FoulProbCollector,
    FoulProbTable,
    OpeningBookCollector,
    OpeningBookTable,
    PolicyPriorCollector,
    PolicyPriorTable,
    TableAwareConfig,
    TableAwarePolicy,
    TranspositionTable,
    Welford,
    action_signature,
    canonical_action,
    fantasy_hand_signature,
    state_signature,
    street1_hand_signature,
    turn_state_signature,
)
from tables.signatures import gamestate_signature


# ============================================================================
# Welford
# ============================================================================
def test_welford_push_and_mean():
    w = Welford()
    for x in (1.0, 2.0, 3.0, 4.0, 5.0):
        w.push(x)
    assert w.n == 5
    assert w.mean == pytest.approx(3.0)
    # variance = sum((x-3)^2)/5 = (4+1+0+1+4)/5 = 2
    assert w.variance == pytest.approx(2.0)


def test_welford_merge_equiv_full_stream():
    full = Welford()
    for x in range(10):
        full.push(float(x))

    a = Welford()
    b = Welford()
    for x in range(0, 5):
        a.push(float(x))
    for x in range(5, 10):
        b.push(float(x))
    a.merge(b)

    assert a.n == full.n
    assert a.mean == pytest.approx(full.mean)
    assert a.M2 == pytest.approx(full.M2)


def test_welford_merge_with_empty():
    w = Welford()
    w.push(1.0)
    w.push(3.0)
    w.merge(Welford())  # no-op
    assert w.n == 2
    empty = Welford()
    empty.merge(w)
    assert empty.n == 2
    assert empty.mean == pytest.approx(2.0)


# ============================================================================
# Signatures
# ============================================================================
def test_state_signature_deterministic_and_order_independent():
    gs = GameState.new(seed=0)
    gs.deal_street()
    sig1 = gamestate_signature(gs, 0)

    gs2 = GameState.new(seed=0)
    gs2.deal_street()
    # Permute pending; signature should be the same
    gs2.hands[0].pending = list(reversed(gs.hands[0].pending))
    sig2 = gamestate_signature(gs2, 0)
    assert sig1 == sig2


def test_action_signature_order_independent():
    p1 = ((10, SLOT_TOP), (20, SLOT_MIDDLE), (30, SLOT_BOTTOM))
    p2 = ((30, SLOT_BOTTOM), (10, SLOT_TOP), (20, SLOT_MIDDLE))
    assert canonical_action(p1) == canonical_action(p2)
    assert action_signature(Action(p1)) == action_signature(Action(p2))


def test_street1_signature_validates_length():
    with pytest.raises(ValueError):
        street1_hand_signature((1, 2, 3))


def test_fantasy_hand_signature_includes_tier():
    cards = (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 0, 3)
    a = fantasy_hand_signature(cards, int(FantasyTier.F14))
    b = fantasy_hand_signature(cards, int(FantasyTier.F15))
    assert a != b
    assert a[0] == b[0]


# ============================================================================
# Helpers
# ============================================================================
def _record_full_trace(seed: int):
    """Play a game with both heuristic policies and full trace recording."""
    gs = GameState.new(seed=seed)
    p0 = HeuristicPolicy(seed=seed * 2)
    p1 = HeuristicPolicy(seed=seed * 2 + 1)
    return simulate_one_game(gs, p0, p1, record_turns=True, seed_for_record=seed)


def _records_full_trace(n: int):
    return [_record_full_trace(s) for s in range(n)]


# ============================================================================
# FoulProbCollector / FoulProbTable
# ============================================================================
def test_foul_prob_collector_observe_and_lookup():
    col = FoulProbCollector()
    for rec in _records_full_trace(5):
        col.observe(rec)
    table = col.result()
    assert isinstance(table, FoulProbTable)
    assert len(table) > 0
    # default for unseen state is 0.0
    assert table.lookup(("garbage",), default=0.0) == 0.0


def test_foul_prob_collector_merge_and_pickle():
    a = FoulProbCollector()
    b = FoulProbCollector()
    recs = _records_full_trace(6)
    for r in recs[:3]:
        a.observe(r)
    for r in recs[3:]:
        b.observe(r)

    full = FoulProbCollector()
    for r in recs:
        full.observe(r)

    blob = pickle.dumps(b)
    b2 = pickle.loads(blob)
    a.merge(b2)

    # Every state seen by `full` should be in `a` with the same counts.
    for sig, cell in full.cells.items():
        ac = a.cells[sig]
        assert ac.n == cell.n
        assert ac.fouls == cell.fouls


def test_foul_prob_collector_merge_class_guard():
    a = FoulProbCollector()
    b = PolicyPriorCollector()
    with pytest.raises(TypeError):
        a.merge(b)  # type: ignore[arg-type]


# ============================================================================
# PolicyPriorCollector / PolicyPriorTable
# ============================================================================
def test_policy_prior_collector_records_outcomes():
    col = PolicyPriorCollector()
    for rec in _records_full_trace(5):
        col.observe(rec)
    table = col.result()
    assert isinstance(table, PolicyPriorTable)
    assert len(table) > 0
    assert table.total_visits() > 0


def test_policy_prior_table_best_action_respects_min_visits():
    table = PolicyPriorTable()
    sig = ("s",)
    asig1 = ("a1",)
    asig2 = ("a2",)
    w1 = Welford()
    for _ in range(3):
        w1.push(100.0)  # high mean but low visits
    w2 = Welford()
    for _ in range(20):
        w2.push(1.0)
    table.cells[sig] = {asig1: w1, asig2: w2}

    # min_visits=4 disqualifies a1
    assert table.best_action(sig, min_visits=4) == asig2
    # min_visits=2 admits a1 (higher mean)
    assert table.best_action(sig, min_visits=2) == asig1
    # unseen state
    assert table.best_action(("never",)) is None


def test_policy_prior_collector_merge_pickle():
    a = PolicyPriorCollector()
    b = PolicyPriorCollector()
    recs = _records_full_trace(6)
    for r in recs[:3]:
        a.observe(r)
    for r in recs[3:]:
        b.observe(r)

    full = PolicyPriorCollector()
    for r in recs:
        full.observe(r)

    a.merge(pickle.loads(pickle.dumps(b)))

    # Same total visit count
    a_total = sum(w.n for st in a.cells.values() for w in st.values())
    full_total = sum(w.n for st in full.cells.values() for w in st.values())
    assert a_total == full_total


# ============================================================================
# OpeningBookCollector / OpeningBookTable
# ============================================================================
def test_opening_book_collector_only_street1():
    col = OpeningBookCollector()
    for rec in _records_full_trace(8):
        col.observe(rec)
    table = col.result()
    assert isinstance(table, OpeningBookTable)
    # street-1 has 5 cards exactly
    for hand_key in table.entries:
        assert len(hand_key) == 5


def test_opening_book_lookup_defaults_to_none():
    table = OpeningBookTable()
    assert table.lookup((1, 2, 3, 4, 5)) is None


def test_opening_book_pickle_merge():
    a = OpeningBookCollector()
    b = OpeningBookCollector()
    recs = _records_full_trace(8)
    for r in recs[:4]:
        a.observe(r)
    for r in recs[4:]:
        b.observe(r)
    full = OpeningBookCollector()
    for r in recs:
        full.observe(r)

    a.merge(pickle.loads(pickle.dumps(b)))

    # Same set of hand keys
    assert set(a.entries.keys()) == set(full.entries.keys())


# ============================================================================
# FantasyEVCollector / FantasyEVTable
# ============================================================================
def test_fantasy_ev_collector_no_full_trace_needed():
    assert FantasyEVCollector.needs_full_trace is False


def test_fantasy_ev_collector_observes_records():
    col = FantasyEVCollector()
    for s in range(8):
        rec = play_game(seed=s, p0_factory=heuristic_factory, p1_factory=heuristic_factory)
        col.observe(rec)
    table = col.result()
    assert isinstance(table, FantasyEVTable)
    # NORMAL tier should have all observations (no fantasy injected)
    normal_stats = table.for_tier(FantasyTier.NORMAL)
    assert normal_stats.n_games == 16  # 8 games * 2 players


def test_fantasy_ev_value_function_anchored_at_normal():
    table = FantasyEVTable(stats={
        int(FantasyTier.NORMAL): __import__("tables").fantasy_ev.TierStats(
            n_games=100, sum_score=0.0, n_to_normal=90, n_to_fantasy=10,
            transitions={int(FantasyTier.NORMAL): 90, int(FantasyTier.F14): 10},
        ),
        int(FantasyTier.F14): __import__("tables").fantasy_ev.TierStats(
            n_games=20, sum_score=200.0, n_to_normal=15, n_to_fantasy=5,
            transitions={int(FantasyTier.NORMAL): 15, int(FantasyTier.F14): 5},
        ),
    })
    V = table.value_function()
    assert V[int(FantasyTier.NORMAL)] == pytest.approx(0.0, abs=1e-4)
    # F14 is much better than NORMAL
    assert V[int(FantasyTier.F14)] > 0.0
    bonuses = table.continue_bonuses()
    assert bonuses[int(FantasyTier.F14)] > 0.0


def test_fantasy_ev_pickle_merge():
    a = FantasyEVCollector()
    b = FantasyEVCollector()
    for s in range(0, 4):
        a.observe(play_game(seed=s, p0_factory=heuristic_factory, p1_factory=heuristic_factory))
    for s in range(4, 8):
        b.observe(play_game(seed=s, p0_factory=heuristic_factory, p1_factory=heuristic_factory))

    full = FantasyEVCollector()
    for s in range(8):
        full.observe(play_game(seed=s, p0_factory=heuristic_factory, p1_factory=heuristic_factory))

    a.merge(pickle.loads(pickle.dumps(b)))
    for tier in a.stats:
        assert a.stats[tier].n_games == full.stats[tier].n_games
        assert a.stats[tier].sum_score == pytest.approx(full.stats[tier].sum_score)


# ============================================================================
# FantasyArrangementCacheCollector / FantasyArrangementCache
# ============================================================================
def _force_fantasy_record(tier: FantasyTier, seed: int):
    """Play one hand with player 0 in fantasy and full-trace recording."""
    gs = GameState.new(seed=seed, fantasy_p0=tier)
    p0 = HeuristicPolicy(seed=seed)
    p1 = HeuristicPolicy(seed=seed + 1)
    return simulate_one_game(gs, p0, p1, record_turns=True, seed_for_record=seed)


def test_fantasy_cache_collector_extracts_fantasy_turns():
    col = FantasyArrangementCacheCollector()
    for s in range(3):
        rec = _force_fantasy_record(FantasyTier.F14, seed=s)
        col.observe(rec)
    cache = col.result()
    assert isinstance(cache, FantasyArrangementCache)
    assert len(cache) >= 1


def test_fantasy_cache_lookup_and_to_action_round_trip():
    rec = _force_fantasy_record(FantasyTier.F14, seed=0)
    col = FantasyArrangementCacheCollector()
    col.observe(rec)
    cache = col.result()

    # find the fantasy turn that was cached
    fantasy_turn = next(
        t for t in rec.turns if t.fantasy_tier != int(FantasyTier.NORMAL)
    )
    entry = cache.lookup(fantasy_turn.pending, FantasyTier(fantasy_turn.fantasy_tier))
    assert entry is not None
    # Re-build the action from the cache entry
    action = entry.to_action(fantasy_turn.pending)
    # cards must match
    assert sorted(c for c, _ in action.placements) == sorted(fantasy_turn.pending)


def test_fantasy_cache_collector_pickle_merge():
    a = FantasyArrangementCacheCollector()
    b = FantasyArrangementCacheCollector()
    for s in range(0, 3):
        a.observe(_force_fantasy_record(FantasyTier.F14, seed=s))
    for s in range(3, 6):
        b.observe(_force_fantasy_record(FantasyTier.F14, seed=s))
    a.merge(pickle.loads(pickle.dumps(b)))
    assert len(a.entries) >= 1


# ============================================================================
# TranspositionTable
# ============================================================================
def test_transposition_lookup_and_store():
    tt = TranspositionTable(max_entries=100)
    sig = ("s",)
    a = Action(((1, SLOT_TOP),))
    assert tt.lookup(sig) is None
    tt.store(sig, a)
    assert tt.lookup(sig) == a
    assert tt.hits == 1
    assert tt.misses == 1


def test_transposition_eviction_fifo():
    tt = TranspositionTable(max_entries=3)
    for i in range(5):
        tt.store((i,), Action(((i, SLOT_TOP),)))
    # only the last 3 keys remain
    assert len(tt) == 3
    assert tt.lookup((0,)) is None
    assert tt.lookup((4,)) is not None


def test_transposition_save_load(tmp_path):
    tt = TranspositionTable(max_entries=100)
    tt.store(("a",), Action(((0, SLOT_TOP),)))
    tt.store(("b",), Action(((1, SLOT_MIDDLE),)))
    p = tmp_path / "tt.pkl"
    tt.save(p)

    loaded = TranspositionTable.load(p)
    assert len(loaded) == 2
    assert loaded.lookup(("a",)) == Action(((0, SLOT_TOP),))


def test_transposition_from_policy_prior():
    prior = PolicyPriorTable()
    sig = ("s",)
    asig_good = ((1, SLOT_TOP),)
    asig_bad = ((1, SLOT_BOTTOM),)
    w_good = Welford()
    for _ in range(20):
        w_good.push(10.0)
    w_bad = Welford()
    for _ in range(20):
        w_bad.push(-10.0)
    prior.cells[sig] = {asig_good: w_good, asig_bad: w_bad}

    tt = TranspositionTable.from_policy_prior(prior, min_visits=5)
    assert tt.lookup(sig) == Action(asig_good)


# ============================================================================
# TableAwarePolicy
# ============================================================================
class _FakePolicy:
    """Minimal Policy stub that records calls."""

    name = "fake"

    def __init__(self):
        self.calls = 0

    def act(self, gs, player):
        self.calls += 1
        return HeuristicPolicy(seed=0).act(gs, player)


def test_table_aware_falls_back_when_no_tables():
    fake = _FakePolicy()
    pol = TableAwarePolicy(fallback=fake)
    gs = GameState.new(seed=0)
    gs.deal_street()
    a = pol.act(gs, 0)
    assert fake.calls == 1
    # fallback action should be legal on this hand
    cards = sorted(c for c, _ in a.placements)
    assert cards == sorted(gs.hands[0].pending)


def test_table_aware_uses_transposition_when_present():
    fake = _FakePolicy()
    tt = TranspositionTable()
    pol = TableAwarePolicy(fallback=fake, transposition=tt)

    gs = GameState.new(seed=0)
    gs.deal_street()

    # First call: miss in TT, fallback called, result memoized.
    a1 = pol.act(gs, 0)
    assert fake.calls == 1
    assert pol.n_fallback_calls == 1
    sig = gamestate_signature(gs, 0)
    assert tt.lookup(sig) == a1

    # Second call on identical state: TT hit, no fallback.
    gs2 = GameState.new(seed=0)
    gs2.deal_street()
    a2 = pol.act(gs2, 0)
    assert fake.calls == 1                    # still 1 — fallback not called
    assert pol.n_transposition_hits == 1
    assert a2 == a1


def test_table_aware_uses_opening_book_when_visits_sufficient():
    # Build an opening book with one entry.
    gs = GameState.new(seed=0)
    gs.deal_street()
    pending = tuple(gs.hands[0].pending)
    hand_key = street1_hand_signature(pending)

    # Manually craft a "preferred" action: place all 5 on the bottom-ish.
    # Use the heuristic's choice as the canonical action.
    chosen = HeuristicPolicy(seed=99).act(gs, 0)
    asig = canonical_action(chosen.placements)

    book = OpeningBookTable()
    w = Welford()
    for _ in range(10):
        w.push(50.0)
    book.entries[hand_key] = {asig: w}

    fake = _FakePolicy()
    pol = TableAwarePolicy(
        fallback=fake,
        opening_book=book,
        config=TableAwareConfig(opening_min_visits=4),
    )
    a = pol.act(gs, 0)
    assert pol.n_opening_hits == 1
    assert fake.calls == 0
    assert canonical_action(a.placements) == asig


def test_table_aware_rejects_illegal_cached_action():
    # Pre-store an action whose cards don't match the current pending.
    tt = TranspositionTable()
    gs = GameState.new(seed=0)
    gs.deal_street()
    sig = gamestate_signature(gs, 0)
    bogus = Action(((52, SLOT_TOP), (53, SLOT_TOP)))  # jokers, wrong count
    tt.entries[sig] = bogus  # bypass store() so legality isn't checked at insert

    fake = _FakePolicy()
    pol = TableAwarePolicy(
        fallback=fake,
        transposition=tt,
        config=TableAwareConfig(record_in_transposition=False),
    )
    a = pol.act(gs, 0)
    # Illegal cached action rejected → fallback used.
    assert fake.calls == 1
    cards = sorted(c for c, _ in a.placements)
    assert cards == sorted(gs.hands[0].pending)
