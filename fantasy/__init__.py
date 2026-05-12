"""Fantasyland solver — deterministic combinatorial optimization.

Fantasyland (in this Pineapple OFC variant) is *not* a rollout problem:
all cards are visible and there is no future randomness. We treat it as a
search problem with bottom-first DFS, beam pruning, and branch-and-bound.

Public entry point: `fantasy.fantasy_solver.solve_fantasy`.
"""
