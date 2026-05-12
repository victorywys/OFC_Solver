"""Tests for the rich (v2) canonical opening-book schema.

Covers:
  * Backward compatibility: legacy (v1) action-sig entries still load
    and `lookup` works.
  * Rich entries pickle/unpickle, expose `CandidateRecord`, and
    `lookup` returns the same canonical-best action.
  * `lookup_horizon` re-ranks candidates by ``ev_mean + sum_t P(t|a) *
    bonus[t]`` and respects suit-symmetry inverse.
  * `TableAwarePolicy.set_horizon_values(...)` flips the street-1
    decision when the rich book's per-candidate stats favour a
    different action under a non-zero horizon.
"""

from __future__ import annotations

import pickle

import pytest

from engine.fantasy import FantasyTier
from state.action import Action
from state.board import SLOT_BOTTOM, SLOT_DISCARD, SLOT_MIDDLE, SLOT_TOP
from state.game_state import GameState
from tables.canonical_opening import (
    CandidateRecord,
    CanonicalOpeningBookTable,
    canonicalize,
)
from tables.table_aware_policy import TableAwareConfig, TableAwarePolicy
from ai.heuristic_policy import HeuristicPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
HAND = (0, 1, 2, 3, 4)   # 2c 2d 2h 2s 3c — already canonical


def _action(slot_for_3c: int) -> tuple[tuple[int, int], ...]:
    """All four 2s on bottom; the lone 3c into the requested row."""
    return tuple(sorted([
        (0, SLOT_BOTTOM),
        (1, SLOT_BOTTOM),
        (2, SLOT_BOTTOM),
        (3, SLOT_BOTTOM),
        (4, slot_for_3c),
    ]))


def _rich_table() -> CanonicalOpeningBookTable:
    """Synthetic 1-entry rich book mimicking the smoke build's output."""
    records = (
        # Candidate A: 3c on middle. Higher this-hand EV, low fantasy entry.
        CandidateRecord(
            placements=_action(SLOT_MIDDLE),
            ev_mean=7.0,
            ev_se=0.5,
            n_rollouts=100,
            foul_rate=0.0,
            fantasy_entry_rate=0.05,
            dest_tier_counts=((int(FantasyTier.NORMAL), 95),
                              (int(FantasyTier.F14), 5)),
        ),
        # Candidate B: 3c on top. Lower this-hand EV but high fantasy entry.
        CandidateRecord(
            placements=_action(SLOT_TOP),
            ev_mean=5.0,
            ev_se=0.5,
            n_rollouts=100,
            foul_rate=0.0,
            fantasy_entry_rate=0.50,
            dest_tier_counts=(
                (int(FantasyTier.NORMAL), 50),
                (int(FantasyTier.F14), 35),
                (int(FantasyTier.F15), 15),
            ),
        ),
    )
    return CanonicalOpeningBookTable({HAND: records})


def _legacy_table() -> CanonicalOpeningBookTable:
    """Synthetic 1-entry legacy (v1) book — a single action signature."""
    return CanonicalOpeningBookTable({HAND: _action(SLOT_MIDDLE)})


# ---------------------------------------------------------------------------
# Schema detection + lookup
# ---------------------------------------------------------------------------
def test_legacy_table_loads_and_lookup_works() -> None:
    tbl = _legacy_table()
    assert not tbl.is_rich()
    assert tbl.lookup(list(HAND)) == _action(SLOT_MIDDLE)
    assert tbl.candidates(list(HAND)) == []


def test_rich_table_lookup_returns_best_by_ev_mean() -> None:
    tbl = _rich_table()
    assert tbl.is_rich()
    # Best by ev_mean is candidate A (3c on middle).
    assert tbl.lookup(list(HAND)) == _action(SLOT_MIDDLE)


def test_rich_table_candidates_exposes_full_record_list() -> None:
    tbl = _rich_table()
    cands = tbl.candidates(list(HAND))
    assert len(cands) == 2
    # Sorted by ev_mean desc by the builder.
    assert cands[0].ev_mean > cands[1].ev_mean
    # Probability distribution sums to 1 (modulo rounding).
    dist = cands[1].dest_tier_distribution
    assert pytest.approx(sum(dist.values()), abs=1e-9) == 1.0


def test_rich_table_pickles_round_trip(tmp_path) -> None:
    tbl = _rich_table()
    p = tmp_path / "book.pkl"
    with p.open("wb") as f:
        pickle.dump(tbl, f)
    with p.open("rb") as f:
        tbl2 = pickle.load(f)
    assert isinstance(tbl2, CanonicalOpeningBookTable)
    assert tbl2.is_rich()
    assert tbl2.lookup(list(HAND)) == tbl.lookup(list(HAND))


# ---------------------------------------------------------------------------
# Horizon-aware lookup
# ---------------------------------------------------------------------------
def test_lookup_horizon_zero_bonus_matches_plain_lookup() -> None:
    tbl = _rich_table()
    # Empty bonuses -> identical to lookup().
    assert tbl.lookup_horizon(list(HAND), tier_horizon_values={}) == \
        tbl.lookup(list(HAND))


def test_lookup_horizon_flips_action_when_fantasy_bonus_large() -> None:
    """The rich book stores A (ev=7.0, 5% fantasy) vs B (ev=5.0, 50% fantasy).

    Without horizon: A wins.  With a horizon bonus of +20 chips for
    every F14 future, B's expected score becomes 5.0 + 0.35*20 + 0.15*22 = ~15.3
    which beats A's 7.0 + 0.05*20 = 8.0.
    """
    tbl = _rich_table()
    bonuses = {
        int(FantasyTier.NORMAL): 0.0,
        int(FantasyTier.F14): 20.0,
        int(FantasyTier.F15): 22.0,
    }
    asig = tbl.lookup_horizon(list(HAND), tier_horizon_values=bonuses)
    assert asig == _action(SLOT_TOP)


def test_lookup_horizon_legacy_falls_back_to_plain_lookup() -> None:
    tbl = _legacy_table()
    bonuses = {int(FantasyTier.F14): 100.0}
    # Legacy entries have no per-candidate stats → horizon has no effect.
    assert tbl.lookup_horizon(list(HAND), tier_horizon_values=bonuses) == \
        tbl.lookup(list(HAND))


# ---------------------------------------------------------------------------
# Suit-symmetry round trip with rich entries
# ---------------------------------------------------------------------------
def test_lookup_horizon_re_suits_correctly() -> None:
    """Build a rich entry under canonical suits, query under a re-suited
    hand, verify the returned action uses the live hand's card ids."""
    tbl = _rich_table()
    # Same hand re-suited: rotate clubs<->spades.
    # Cards (rank=0, suits 0..3) -> remap to (suits 3,1,2,0).
    perm = (3, 1, 2, 0)
    real_hand = []
    for c in HAND:
        rank, suit = divmod(c, 4)
        real_hand.append(rank * 4 + perm[suit])
    canon_hand, _ = canonicalize(real_hand)
    assert canon_hand == HAND  # canonicalization undoes our perm

    bonuses = {int(FantasyTier.F14): 20.0, int(FantasyTier.F15): 22.0}
    asig = tbl.lookup_horizon(real_hand, tier_horizon_values=bonuses)
    assert asig is not None
    # Re-suited result must use exactly the live hand's card ids.
    assert sorted(c for c, _ in asig) == sorted(real_hand)


# ---------------------------------------------------------------------------
# TableAwarePolicy: set_horizon_values flips the decision
# ---------------------------------------------------------------------------
def test_table_aware_policy_set_horizon_values_flips_decision() -> None:
    tbl = _rich_table()
    pol = TableAwarePolicy(
        fallback=HeuristicPolicy(seed=0),
        config=TableAwareConfig(),
        opening_book=tbl,
    )

    # Build a street-1 normal-tier state whose pending equals the
    # canonical hand we encoded.
    gs = GameState.new(n_players=2, seed=42)
    gs.deal_street()
    gs.hands[0].pending = list(HAND)
    # Opponent gets 5 cards not in our hand (any deal — doesn't affect
    # the opening-book lookup).
    others = [c for c in range(54) if c not in HAND]
    gs.hands[1].pending = others[:5]

    # Without horizon: returns candidate A (3c on middle).
    pol.set_horizon_values(None)
    a_no_horizon = pol.act(gs, 0)
    assert tuple(sorted(a_no_horizon.placements)) == _action(SLOT_MIDDLE)

    # With strong horizon bonus: returns candidate B (3c on top).
    pol.set_horizon_values({
        int(FantasyTier.F14): 20.0,
        int(FantasyTier.F15): 22.0,
    })
    a_with_horizon = pol.act(gs, 0)
    assert tuple(sorted(a_with_horizon.placements)) == _action(SLOT_TOP)

    # Reset clears the override.
    pol.set_horizon_values(None)
    a_reset = pol.act(gs, 0)
    assert tuple(sorted(a_reset.placements)) == _action(SLOT_MIDDLE)


def test_table_aware_policy_horizon_is_thread_local() -> None:
    """Two threads with different horizons must not mix.

    We don't actually parallel-call here — just verify the attribute is
    on `threading.local` storage so concurrent users would isolate.
    """
    import threading
    tbl = _rich_table()
    pol = TableAwarePolicy(
        fallback=HeuristicPolicy(seed=0),
        config=TableAwareConfig(),
        opening_book=tbl,
    )
    pol.set_horizon_values({int(FantasyTier.F14): 100.0})

    seen_in_other_thread: list = []

    def _other_thread():
        # New thread sees default (None) horizon, not the main thread's.
        val = getattr(pol._counters, "tier_horizon_values", "MISSING")
        seen_in_other_thread.append(val)

    t = threading.Thread(target=_other_thread)
    t.start()
    t.join()
    # The other thread either sees None (default) or no attribute yet;
    # crucially it must NOT see {F14: 100.0}.
    assert seen_in_other_thread[0] in (None, "MISSING")
