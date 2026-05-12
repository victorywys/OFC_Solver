"""Benchmark policy decision throughput. Run as:

    python -m benchmarks.bench_policy
"""

from __future__ import annotations

import time

from ai.heuristic_policy import HeuristicPolicy
from ai.random_policy import RandomPolicy
from state.game_state import GameState, N_NORMAL_STREETS


def bench_policy(make_policy, n_games: int = 200) -> None:
    pol_a = make_policy(seed=1)
    pol_b = make_policy(seed=2)

    n_decisions = 0
    t0 = time.perf_counter()
    for seed in range(n_games):
        gs = GameState.new(seed=seed)
        for _ in range(N_NORMAL_STREETS):
            gs.deal_street()
            for p in (0, 1):
                gs.step(p, (pol_a if p == 0 else pol_b).act(gs, p))
                n_decisions += 1
    dt = time.perf_counter() - t0
    name = type(pol_a).__name__
    print(
        f"{name:20s}  {n_games:>5d} games  {n_decisions:>6d} decisions  "
        f"{dt*1000:.1f} ms total  {n_decisions/dt:7.0f} dec/s  "
        f"{dt*1e6/n_decisions:6.1f} us/dec"
    )


def main() -> None:
    bench_policy(RandomPolicy, n_games=400)
    bench_policy(HeuristicPolicy, n_games=200)


if __name__ == "__main__":
    main()
