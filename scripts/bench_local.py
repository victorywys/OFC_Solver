"""Quick wall-time benchmark for FastAnalyzer single-call latency.

Loads tables once, runs N analyze() calls back-to-back, reports per-call ms.
"""
from __future__ import annotations

import statistics
import time

from ai.heuristic_policy import HeuristicPolicy
from fantasy.fantasy_solver import FantasySolverPolicy
from tables import TableAwareConfig, TableAwarePolicy, TranspositionTable
from tables.loader import load_tables
from ui.analyzer_fast import FastAnalyzer
from ui.state_builder import build_game_state


def main():
    run_dir = "artifacts/run_100k_20260508_103044/"
    t = load_tables(run_dir)
    cfg = TableAwareConfig(prior_min_visits=1, opening_min_visits=1)
    heur = HeuristicPolicy(seed=0)
    fallback = FantasySolverPolicy(fallback=heur)
    pol = TableAwarePolicy(
        fallback=fallback, config=cfg,
        transposition=TranspositionTable(max_entries=200_000),
        opening_book=t.get("opening_book_canonical") or t.get("opening_book"),
        fantasy_cache=t.get("fantasy_arrangement"),
        policy_prior=t.get("policy_prior"),
    )
    an = FastAnalyzer(
        policy=pol,
        foul_prob_table=t.get("foul_prob"),
        policy_prior_table=t.get("policy_prior"),
        fantasy_ev_table=t.get("fantasy_ev"),
        rollout_seed=0,
    )

    spec = {
        "to_act": 0,
        "street": 1,
        "auto_fill_opponent": True,
        "players": [
            {"fantasy_tier": 0, "board": {"top": [], "middle": [], "bottom": [], "discards": []},
             "pending": ["As", "Kd", "Qh", "Jc", "Ts"]},
            {"fantasy_tier": 0, "board": {"top": [], "middle": [], "bottom": [], "discards": []},
             "pending": []},
        ],
        "dead_cards": [],
    }
    gs = build_game_state(spec)
    # warm
    for _ in range(3):
        an.analyze(gs, 0, n_rollouts=20, top_k=3)

    n_trials = 15
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        an.analyze(gs, 0, n_rollouts=20, top_k=3)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"n_rollouts=20 top_k=3 — {n_trials} trials")
    print(f"  min={times[0]:.0f}ms  med={statistics.median(times):.0f}ms  mean={statistics.mean(times):.0f}ms  max={times[-1]:.0f}ms")

    # Heuristic-only path
    if hasattr(an, "analyze_heuristic_only"):
        h_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            an.analyze_heuristic_only(gs, 0, top_k=3)
            h_times.append((time.perf_counter() - t0) * 1000)
        h_times.sort()
        print(f"\nheuristic_only (top_k=3) — 20 trials")
        print(f"  min={h_times[0]:.1f}ms  med={statistics.median(h_times):.1f}ms  mean={statistics.mean(h_times):.1f}ms  max={h_times[-1]:.1f}ms")


if __name__ == "__main__":
    main()
