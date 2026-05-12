"""Tests for the self-play framework.

Covers:
    - Single-game record correctness (summary-only & full-trace)
    - Collector observe / merge contract
    - SelfPlay sequential reproducibility
    - SelfPlay parallel results match sequential
    - Save / load roundtrip
    - Evaluation harness (head-to-head)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from engine.fantasy import FantasyTier
from simulation.collectors import (
    FantasyTransitionCollector,
    FoulByTierCollector,
    MatchSummaryCollector,
    RoyaltyByRowCollector,
    TraceCollector,
)
from simulation.evaluation import evaluate_matchup
from simulation.policy_factories import (
    heuristic_factory,
    random_factory,
)
from simulation.self_play import SelfPlay, play_game
from simulation.storage import load_run, load_table, save_collectors
from simulation.trace import simulate_one_game
from state.game_state import GameState


# ============================================================================
# play_game / simulate_one_game
# ============================================================================
def test_play_game_summary_only_record():
    rec = play_game(seed=0, p0_factory=heuristic_factory, p1_factory=heuristic_factory)
    assert rec.seed == 0
    assert rec.policy_a_name == "heuristic"
    assert rec.policy_b_name == "heuristic"
    # Summary-only: turns is empty
    assert rec.turns == []
    # Final boards always 13 cards each
    assert len(rec.final_a.top) == 3
    assert len(rec.final_a.middle) == 5
    assert len(rec.final_a.bottom) == 5
    # Score sanity: zero-sum
    assert rec.total_a + rec.total_b == 0


def test_play_game_full_trace_records_all_decisions():
    gs = GameState.new(seed=1)
    from ai.heuristic_policy import HeuristicPolicy

    p0 = HeuristicPolicy(seed=1)
    p1 = HeuristicPolicy(seed=2)
    rec = simulate_one_game(gs, p0, p1, record_turns=True, seed_for_record=1)
    # 5 streets x 2 players = 10 turns for normal play.
    assert len(rec.turns) == 10
    # Each turn's placements covers exactly the pending hand.
    for t in rec.turns:
        action_cards = sorted(c for c, _ in t.placements)
        assert action_cards == sorted(t.pending)
    # First two turns are street 1 (one per player)
    streets = [t.street for t in rec.turns]
    assert streets[0] == 1
    assert streets[1] == 1
    assert streets[-1] == 5


def test_play_game_card_conservation():
    rec = play_game(seed=2, p0_factory=heuristic_factory, p1_factory=heuristic_factory)
    seen: set[int] = set()
    for board, discards in (
        (rec.final_a, rec.discards_a),
        (rec.final_b, rec.discards_b),
    ):
        for c in board.top + board.middle + board.bottom + discards:
            assert c not in seen
            seen.add(c)
    # 13 placed + 4 discards per player x 2 = 34 cards total
    assert len(seen) == 34


def test_play_game_deterministic_for_fixed_seed():
    rec1 = play_game(seed=42, p0_factory=heuristic_factory, p1_factory=heuristic_factory)
    rec2 = play_game(seed=42, p0_factory=heuristic_factory, p1_factory=heuristic_factory)
    assert rec1.final_a == rec2.final_a
    assert rec1.final_b == rec2.final_b
    assert rec1.score == rec2.score


# ============================================================================
# Collectors
# ============================================================================
def test_match_summary_collector_basic():
    c = MatchSummaryCollector()
    for s in range(8):
        rec = play_game(seed=s, p0_factory=heuristic_factory, p1_factory=random_factory)
        c.observe(rec)
    res = c.result()
    assert res.n_games == 8
    # Heuristic vs random: P0 should win or tie most games.
    assert res.p0_wins >= res.p1_wins


def test_match_summary_merge_is_associative():
    """Splitting the stream and merging should match observing the union."""
    seeds_all = list(range(10))
    full = MatchSummaryCollector()
    for s in seeds_all:
        full.observe(play_game(seed=s, p0_factory=heuristic_factory, p1_factory=random_factory))

    half_a = MatchSummaryCollector()
    half_b = MatchSummaryCollector()
    for s in seeds_all[:5]:
        half_a.observe(play_game(seed=s, p0_factory=heuristic_factory, p1_factory=random_factory))
    for s in seeds_all[5:]:
        half_b.observe(play_game(seed=s, p0_factory=heuristic_factory, p1_factory=random_factory))
    half_a.merge(half_b)

    full_res = full.result()
    merged_res = half_a.result()
    assert full_res == merged_res


def test_foul_by_tier_collector():
    c = FoulByTierCollector()
    for s in range(6):
        rec = play_game(seed=s, p0_factory=random_factory, p1_factory=random_factory)
        c.observe(rec)
    res = c.result()
    assert "NORMAL" in res
    # All players started in NORMAL (default tier factory)
    assert res["NORMAL"]["n_games"] == 12  # 6 games x 2 players
    assert 0 <= res["NORMAL"]["foul_rate"] <= 1


def test_royalty_collector_only_counts_non_foul_boards():
    c = RoyaltyByRowCollector()
    for s in range(5):
        rec = play_game(seed=s, p0_factory=heuristic_factory, p1_factory=heuristic_factory)
        c.observe(rec)
    res = c.result()
    assert res["n_boards"] >= 0
    # Sum of histograms == n_boards on each row
    assert sum(res["top"].values()) == res["n_boards"]
    assert sum(res["middle"].values()) == res["n_boards"]
    assert sum(res["bottom"].values()) == res["n_boards"]


def test_fantasy_transition_collector():
    c = FantasyTransitionCollector()
    for s in range(8):
        rec = play_game(seed=s, p0_factory=heuristic_factory, p1_factory=heuristic_factory)
        c.observe(rec)
    res = c.result()
    # All players started in NORMAL
    assert "NORMAL" in res


def test_trace_collector_requires_full_trace_flag():
    c = TraceCollector()
    assert c.needs_full_trace is True


def test_collector_merge_type_safety():
    a = MatchSummaryCollector()
    b = FoulByTierCollector()
    with pytest.raises(TypeError):
        a.merge(b)


def test_collector_pickle_roundtrip():
    """All baseline collectors must be picklable (required for parallel)."""
    for cls in (
        MatchSummaryCollector,
        FoulByTierCollector,
        RoyaltyByRowCollector,
        FantasyTransitionCollector,
        TraceCollector,
    ):
        c = cls()
        rec = play_game(seed=0, p0_factory=heuristic_factory, p1_factory=heuristic_factory)
        c.observe(rec)
        blob = pickle.dumps(c, protocol=4)
        c2 = pickle.loads(blob)
        assert c2.result() == c.result() or c2.records == c.records  # type: ignore


# ============================================================================
# SelfPlay sequential
# ============================================================================
def test_self_play_run_sequential():
    sp = SelfPlay(p0_factory=heuristic_factory, p1_factory=random_factory)
    out = sp.run(
        n_games=8,
        collector_factories=[MatchSummaryCollector, FoulByTierCollector],
        seed=100,
    )
    assert "match_summary" in out
    assert "foul_by_tier" in out
    assert out["match_summary"].n_games == 8


def test_self_play_run_reproducible():
    sp1 = SelfPlay(p0_factory=heuristic_factory, p1_factory=random_factory)
    sp2 = SelfPlay(p0_factory=heuristic_factory, p1_factory=random_factory)
    out1 = sp1.run(8, [MatchSummaryCollector], seed=7)
    out2 = sp2.run(8, [MatchSummaryCollector], seed=7)
    assert out1["match_summary"] == out2["match_summary"]


def test_self_play_seeds_arg_overrides():
    sp = SelfPlay(p0_factory=heuristic_factory, p1_factory=random_factory)
    out = sp.run(
        n_games=4,
        collector_factories=[MatchSummaryCollector],
        seeds=[100, 200, 300, 400],
    )
    assert out["match_summary"].n_games == 4


# ============================================================================
# SelfPlay parallel — only run if enough games to make sense
# ============================================================================
def test_self_play_parallel_matches_sequential():
    """Parallel result must match sequential exactly (associative merge)."""
    sp_seq = SelfPlay(p0_factory=heuristic_factory, p1_factory=random_factory)
    sp_par = SelfPlay(p0_factory=heuristic_factory, p1_factory=random_factory)

    out_seq = sp_seq.run(16, [MatchSummaryCollector, FoulByTierCollector], seed=0)
    out_par = sp_par.run_parallel(
        16,
        [MatchSummaryCollector, FoulByTierCollector],
        seed=0,
        n_workers=2,
        chunk_size=4,
    )
    assert out_seq["match_summary"] == out_par["match_summary"]
    assert out_seq["foul_by_tier"] == out_par["foul_by_tier"]


# ============================================================================
# Storage
# ============================================================================
def test_save_and_load_run(tmp_path: Path):
    sp = SelfPlay(p0_factory=heuristic_factory, p1_factory=random_factory)
    out = sp.run(4, [MatchSummaryCollector, FoulByTierCollector], seed=0)
    save_collectors(
        out["_collectors"],
        tmp_path,
        metadata={"n_games": 4, "policies": ("heuristic", "random")},
    )
    # Files exist
    assert (tmp_path / "match_summary.pkl").exists()
    assert (tmp_path / "foul_by_tier.pkl").exists()
    assert (tmp_path / "metadata.pkl").exists()

    # Roundtrip
    loaded = load_run(tmp_path)
    assert loaded["match_summary"] == out["match_summary"]
    assert loaded["foul_by_tier"] == out["foul_by_tier"]
    assert loaded["metadata"]["n_games"] == 4


def test_load_table_single_file(tmp_path: Path):
    rec = play_game(seed=0, p0_factory=heuristic_factory, p1_factory=heuristic_factory)
    c = MatchSummaryCollector()
    c.observe(rec)
    p = tmp_path / "summary.pkl"
    with p.open("wb") as f:
        pickle.dump(c.result(), f, protocol=4)
    loaded = load_table(p)
    assert loaded == c.result()


# ============================================================================
# Evaluation harness
# ============================================================================
def test_evaluate_matchup_seat_symmetric_basic():
    res = evaluate_matchup(
        a_factory=heuristic_factory,
        b_factory=random_factory,
        n_games=6,
        seat_symmetric=True,
        seed=0,
    )
    assert res.n_games == 12  # 6 pairs (A=P0 + A=P1)
    # Heuristic should beat random on average over both seats.
    assert res.a_avg_score > 0


def test_evaluate_matchup_asymmetric():
    res = evaluate_matchup(
        a_factory=heuristic_factory,
        b_factory=random_factory,
        n_games=4,
        seat_symmetric=False,
        seed=0,
    )
    assert res.n_games == 4


# ============================================================================
# Trace recording end-to-end
# ============================================================================
def test_self_play_records_turns_when_needed():
    sp = SelfPlay(p0_factory=heuristic_factory, p1_factory=random_factory)
    out = sp.run(
        n_games=2,
        collector_factories=[MatchSummaryCollector, TraceCollector],
        seed=0,
    )
    traces = out["trace"]
    assert len(traces) == 2
    # Every record must have non-empty turns
    for rec in traces:
        assert len(rec.turns) == 10
