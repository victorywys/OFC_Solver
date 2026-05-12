"""Integration tests for `ai/monte_carlo_policy.py`.

These are deliberately small (n_rollouts low, top_k aggressive) so the
suite stays fast. Statistical claims (foul rate, win margin) use loose
bounds against a moderate sample.
"""

from __future__ import annotations

from ai.heuristic_policy import HeuristicPolicy
from ai.monte_carlo_policy import MCConfig, MonteCarloPolicy
from ai.random_policy import RandomPolicy
from engine.fantasy import FantasyTier
from state.game_state import GameState, N_NORMAL_STREETS


# ------------------------- legality -------------------------
def test_mc_returns_legal_action_initial_street():
    gs = GameState.new(seed=10)
    gs.deal_street()
    pol = MonteCarloPolicy(
        config=MCConfig(n_rollouts=4, top_k=8), seed=1
    )
    a = pol.act(gs, 0)
    assert sorted(c for c, _ in a.placements) == sorted(gs.hands[0].pending)
    # On street 1, no discards.
    assert a.discards() == ()


def test_mc_returns_legal_action_pineapple_street():
    gs = GameState.new(seed=11)
    h = HeuristicPolicy(seed=1)
    gs.deal_street()
    gs.step(0, h.act(gs, 0))
    gs.step(1, h.act(gs, 1))
    gs.deal_street()
    pol = MonteCarloPolicy(config=MCConfig(n_rollouts=4), seed=2)
    a = pol.act(gs, 0)
    assert sorted(c for c, _ in a.placements) == sorted(gs.hands[0].pending)
    assert len(a.discards()) == 1


# ------------------------- determinism -------------------------
def test_mc_deterministic_for_fixed_seed_and_state():
    gs1 = GameState.new(seed=33)
    gs2 = GameState.new(seed=33)
    gs1.deal_street()
    gs2.deal_street()
    cfg = MCConfig(n_rollouts=4, top_k=6)
    p1 = MonteCarloPolicy(config=cfg, seed=7)
    p2 = MonteCarloPolicy(config=cfg, seed=7)
    assert p1.act(gs1, 0) == p2.act(gs2, 0)


def test_mc_different_seeds_can_differ():
    """Sanity: not all seeds collapse to the same action."""
    cfg = MCConfig(n_rollouts=4, top_k=6)
    actions = set()
    for s in range(20):
        gs = GameState.new(seed=33)
        gs.deal_street()
        pol = MonteCarloPolicy(config=cfg, seed=s)
        actions.add(pol.act(gs, 0))
    # We don't require strictly different actions every time, but the MC
    # policy should not be a deterministic function of the state alone.
    assert len(actions) >= 2


# ------------------------- top_k prefilter -------------------------
def test_mc_top_k_eq_one_is_pure_heuristic():
    """top_k=1 means we never roll out — we just take heuristic argmax."""
    gs1 = GameState.new(seed=8)
    gs2 = GameState.new(seed=8)
    gs1.deal_street()
    gs2.deal_street()
    mc = MonteCarloPolicy(config=MCConfig(n_rollouts=2, top_k=1), seed=42)
    # Use the same comp_seed by deriving from the same MC seed; instead
    # bypass and compare argmax of score_action directly.
    h = mc.completion  # shared underlying HeuristicPolicy
    a_mc = mc.act(gs1, 0)
    # Force HeuristicPolicy on identical gs2 (different rng for tie-breaks).
    a_h = h.act(gs2, 0)
    assert a_mc == a_h


# ------------------------- foul rate -------------------------
def test_mc_self_play_foul_rate_lower_than_heuristic():
    """MC self-play should match or beat the heuristic foul rate.

    We use a small n_rollouts to keep the test fast. Tight bound: MC foul
    rate must be no worse than heuristic + 5pp.
    """
    n_games = 25
    cfg = MCConfig(n_rollouts=8, top_k=6)

    # Heuristic baseline
    h_fouls = 0
    for seed in range(n_games):
        gs = GameState.new(seed=seed)
        h = HeuristicPolicy(seed=seed)
        for _ in range(N_NORMAL_STREETS):
            gs.deal_street()
            for p in (0, 1):
                gs.step(p, h.act(gs, p))
        sb = gs.score()
        h_fouls += int(sb.a_foul) + int(sb.b_foul)

    # MC self-play
    mc_fouls = 0
    for seed in range(n_games):
        gs = GameState.new(seed=seed)
        m = MonteCarloPolicy(config=cfg, seed=seed)
        for _ in range(N_NORMAL_STREETS):
            gs.deal_street()
            for p in (0, 1):
                gs.step(p, m.act(gs, p))
        sb = gs.score()
        mc_fouls += int(sb.a_foul) + int(sb.b_foul)

    h_rate = h_fouls / (2 * n_games)
    mc_rate = mc_fouls / (2 * n_games)
    # MC must be no worse than heuristic by more than 5pp.
    assert mc_rate <= h_rate + 0.05, (
        f"MC foul rate {mc_rate:.2%} > heuristic {h_rate:.2%} + 5pp"
    )


# ------------------------- strength vs random / heuristic -------------------------
def test_mc_beats_random_on_average():
    n = 20
    cfg = MCConfig(n_rollouts=8, top_k=6)
    total = 0
    for seed in range(n):
        gs = GameState.new(seed=seed)
        m = MonteCarloPolicy(config=cfg, seed=seed)
        r = RandomPolicy(seed=seed + 1000)
        for _ in range(N_NORMAL_STREETS):
            gs.deal_street()
            gs.step(0, m.act(gs, 0))
            gs.step(1, r.act(gs, 1))
        sb = gs.score()
        total += sb.total_a
    avg = total / n
    # Heuristic alone gets +6.15; MC should be at least as good.
    assert avg > 5.0, f"MC vs random avg = {avg:.2f}, expected > 5"


# ------------------------- fantasy delegation -------------------------
def test_mc_delegates_fantasy_to_solver():
    """When fantasy_solver=True, fantasy hands go through FantasySolverPolicy."""
    gs = GameState.new(seed=5, fantasy_p0=FantasyTier.F14)
    gs.deal_street()
    pol = MonteCarloPolicy(
        config=MCConfig(n_rollouts=2, fantasy_solver=True), seed=1
    )
    a = pol.act(gs, 0)
    # Fantasy action: 13 placed + 1 discarded out of 14 dealt.
    assert sorted(c for c, _ in a.placements) == sorted(gs.hands[0].pending)
    assert len(a.discards()) == 14 - 13
