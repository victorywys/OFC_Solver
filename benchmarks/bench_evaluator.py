"""Quick benchmark of the evaluator. Run as:

    python -m benchmarks.bench_evaluator
"""

from __future__ import annotations

import random
import time

from engine.cards import NUM_STD_CARDS
from engine.evaluator import evaluate_3, evaluate_5


def random_5(rng: random.Random) -> list[int]:
    return rng.sample(range(NUM_STD_CARDS), 5)


def random_3(rng: random.Random) -> list[int]:
    return rng.sample(range(NUM_STD_CARDS), 3)


def random_5_with_jokers(rng: random.Random, n_jokers: int) -> list[int]:
    nonj = rng.sample(range(NUM_STD_CARDS), 5 - n_jokers)
    if n_jokers == 1:
        return nonj + [52]
    if n_jokers == 2:
        return nonj + [52, 53]
    return nonj


def bench(label: str, fn, n: int) -> None:
    rng = random.Random(0)
    hands = [random_5(rng) for _ in range(n)]
    t0 = time.perf_counter()
    for h in hands:
        fn(h)
    dt = time.perf_counter() - t0
    print(f"{label:40s}  {n:>9d} hands   {dt*1e6/n:7.2f} us/hand   {n/dt/1e3:7.1f} k/s")


def main() -> None:
    rng = random.Random(0)

    n = 200_000
    bench("evaluate_5 (no jokers, cold cache)", evaluate_5, n)
    bench("evaluate_5 (no jokers, warm cache)", evaluate_5, n)

    n3 = 500_000
    rng3 = random.Random(0)
    hands3 = [random_3(rng3) for _ in range(n3)]
    t0 = time.perf_counter()
    for h in hands3:
        evaluate_3(h)
    dt = time.perf_counter() - t0
    print(f"{'evaluate_3 (no jokers, warm cache)':40s}  {n3:>9d} hands   {dt*1e6/n3:7.2f} us/hand   {n3/dt/1e3:7.1f} k/s")

    # joker hands
    for nj in (1, 2):
        nh = 2_000 if nj == 2 else 20_000
        hands = [random_5_with_jokers(rng, nj) for _ in range(nh)]
        t0 = time.perf_counter()
        for h in hands:
            evaluate_5(h)
        dt = time.perf_counter() - t0
        print(
            f"{'evaluate_5 with ' + str(nj) + ' joker(s)':40s}  "
            f"{nh:>9d} hands   {dt*1e6/nh:7.2f} us/hand   {nh/dt/1e3:7.1f} k/s"
        )


if __name__ == "__main__":
    main()
