"""Benchmark Monte-Carlo rollout policy throughput.

Run as:
    python -m benchmarks.bench_rollout
"""

from __future__ import annotations

import time

from ai.heuristic_policy import HeuristicPolicy
from ai.monte_carlo_policy import MCConfig, MonteCarloPolicy
from state.game_state import GameState, N_NORMAL_STREETS


def bench_one(cfg: MCConfig, n_games: int, label: str) -> None:
    n_decisions = 0
    t0 = time.perf_counter()
    for seed in range(n_games):
        gs = GameState.new(seed=seed)
        m = MonteCarloPolicy(config=cfg, seed=seed)
        for _ in range(N_NORMAL_STREETS):
            gs.deal_street()
            for p in (0, 1):
                gs.step(p, m.act(gs, p))
                n_decisions += 1
    dt = time.perf_counter() - t0
    print(
        f"{label:32s}  games={n_games:>3d}  decs={n_decisions:>4d}  "
        f"{dt*1000:>7.0f} ms total  {dt*1000/n_decisions:>6.1f} ms/dec  "
        f"{n_decisions/dt:>5.1f} dec/s"
    )


def bench_strength(cfg: MCConfig, n_games: int) -> None:
    """Average MC vs heuristic score per hand."""
    total = 0
    t0 = time.perf_counter()
    for seed in range(n_games):
        gs = GameState.new(seed=seed)
        m = MonteCarloPolicy(config=cfg, seed=seed)
        h = HeuristicPolicy(seed=seed + 5000)
        for _ in range(N_NORMAL_STREETS):
            gs.deal_street()
            gs.step(0, m.act(gs, 0))
            gs.step(1, h.act(gs, 1))
        sb = gs.score()
        total += sb.total_a
    dt = time.perf_counter() - t0
    print(
        f"  MC vs heuristic over {n_games} games: avg={total/n_games:+.2f} chips/hand "
        f"({dt:.1f}s total, {dt*1000/n_games:.0f} ms/game)"
    )


def main() -> None:
    print("Monte-Carlo rollout benchmarks")
    print("-" * 80)
    bench_one(MCConfig(n_rollouts=8, top_k=6), n_games=4, label="n=8 top_k=6")
    bench_one(MCConfig(n_rollouts=16, top_k=6), n_games=4, label="n=16 top_k=6")
    bench_one(MCConfig(n_rollouts=32, top_k=6), n_games=2, label="n=32 top_k=6")
    bench_one(MCConfig(n_rollouts=16, top_k=12), n_games=2, label="n=16 top_k=12")
    bench_one(MCConfig(n_rollouts=16, top_k=None), n_games=1, label="n=16 top_k=ALL")
    print()
    print("Strength vs heuristic:")
    bench_strength(MCConfig(n_rollouts=16, top_k=8), n_games=20)


if __name__ == "__main__":
    main()
