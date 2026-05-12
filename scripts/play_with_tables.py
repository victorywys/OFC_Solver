"""Example: load a trained table-build run and play with it.

Usage:
    python -m scripts.play_with_tables --run artifacts/run_<TS>/

Demonstrates:
    1. Loading every table from disk into a single `TableAwarePolicy`.
    2. Calling `policy.act(gs, player)` directly on a fresh game state.
    3. Inspecting which table served each decision.
"""

from __future__ import annotations

import argparse

from engine.cards import cards_str
from state.board import SLOT_NAMES
from state.game_state import GameState

from tables import TableAwareConfig, load_run_as_policy


def _format_action(action) -> str:
    parts = []
    for c, s in action.placements:
        from engine.cards import cards_str
        parts.append(f"{cards_str([c])}->{SLOT_NAMES[s]}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to artifacts/run_<TS>/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior-min-visits", type=int, default=8)
    parser.add_argument("--opening-min-visits", type=int, default=4)
    args = parser.parse_args()

    config = TableAwareConfig(
        prior_min_visits=args.prior_min_visits,
        opening_min_visits=args.opening_min_visits,
    )
    pol = load_run_as_policy(args.run, config=config, seed=args.seed)

    print(f"Loaded policy: {pol}")
    print(f"  opening_book entries  : "
          f"{len(pol.opening_book) if pol.opening_book else 0}")
    print(f"  policy_prior states   : "
          f"{len(pol.policy_prior) if pol.policy_prior else 0}")
    print(f"  fantasy_cache entries : "
          f"{len(pol.fantasy_cache) if pol.fantasy_cache else 0}")
    print()

    # Play one game vs itself, watching which tables fire.
    gs = GameState.new(seed=args.seed)
    pol_b = load_run_as_policy(args.run, config=config, seed=args.seed + 1)

    while not gs.is_terminal():
        gs.deal_street()
        # Both players act this street (P0 then P1).
        for player in (0, 1):
            if gs.hands[player].finished:
                continue
            actor = pol if player == 0 else pol_b
            print(f"-- street {gs.current_street} P{player} | "
                  f"pending: {cards_str(gs.hands[player].pending)}")
            action = actor.act(gs, player)
            print(f"   action: {_format_action(action)}")
            gs.step(player, action)

    score = gs.score()
    print()
    print(f"final score (from P0): {score.total_a:+d}  "
          f"(P0 foul={score.a_foul}, P1 foul={score.b_foul})")
    print()
    print(f"P0 lookups : "
          f"TT={pol.n_transposition_hits} "
          f"opening={pol.n_opening_hits} "
          f"fantasy={pol.n_fantasy_hits} "
          f"prior={pol.n_prior_hits} "
          f"fallback={pol.n_fallback_calls}")
    print(f"P1 lookups : "
          f"TT={pol_b.n_transposition_hits} "
          f"opening={pol_b.n_opening_hits} "
          f"fantasy={pol_b.n_fantasy_hits} "
          f"prior={pol_b.n_prior_hits} "
          f"fallback={pol_b.n_fallback_calls}")


if __name__ == "__main__":
    main()
