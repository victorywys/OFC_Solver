"""Profile FastAnalyzer hot path.

Builds a tables-loaded FastAnalyzer in-process (no HTTP) and runs
several analyze() calls under cProfile. Used to identify hot functions
for Stack B optimization work.

Run:
    python -m scripts.profile_fast
"""
from __future__ import annotations

import cProfile
import io
import pstats

from ai.heuristic_policy import HeuristicPolicy
from fantasy.fantasy_solver import FantasySolverPolicy
from tables import (
    TableAwareConfig,
    TableAwarePolicy,
    TranspositionTable,
)
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

    # Warm-up
    an.analyze(gs, 0, n_rollouts=20, top_k=3)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        an.analyze(gs, 0, n_rollouts=20, top_k=3)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(30)
    print("\n=== by total time (own time) ===\n")
    print(s2.getvalue())


if __name__ == "__main__":
    main()
