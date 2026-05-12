"""Build the full canonical opening book.

Enumerates all **152,646** canonical street-1 hands (jokers included,
suit-symmetric, joker-symmetric — see :mod:`tables.canonical_opening`)
and solves each one with :class:`ai.monte_carlo_policy.MonteCarloPolicy`.

The result is a single pickle file with shape
``dict[canonical_hand_key, action_signature]`` that
:class:`tables.canonical_opening.CanonicalOpeningBookTable` consumes.

Defaults are tuned for the **fast** quality tier so a build fits in
roughly a working day on a 16-core box. Override via flags for stronger
(but slower) results.

Usage
-----

    # Default: 16 workers, fast tier, full enumeration.
    python -m scripts.build_full_opening_book

    # Stronger book; expect ~4 days on 16 cores.
    python -m scripts.build_full_opening_book \\
        --n-rollouts 240 --top-k 5

    # Resume / append to an existing build (skips already-solved hands).
    python -m scripts.build_full_opening_book \\
        --out artifacts/opening_full.pkl --resume

    # Smoke test: solve only the first 16 canonical hands.
    python -m scripts.build_full_opening_book --limit 16 --workers 4

Output
------
By default writes to
``artifacts/opening_book_canonical/opening_book_canonical.pkl`` plus a
small ``metadata.json`` sidecar. The directory layout matches the rest
of the table artifacts so :func:`simulation.storage.load_run` can pick
it up.

Reliability
-----------
* Each canonical hand is solved with a deterministic seed derived from
  the hand bytes — re-running yields identical results.
* Progress is checkpointed every ``--checkpoint-every`` solved hands
  (default 500). On Ctrl+C, the partial book is flushed and the script
  exits cleanly. ``--resume`` picks up where it left off.
* Pool is built with the ``"spawn"`` start method to avoid the fork +
  multi-thread deadlock observed elsewhere in the codebase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import signal
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

# Solver worker module imports its dependencies lazily so the parent
# process stays small at fork-time (we use spawn, but consistency is nice).
HAND_KEY = tuple[int, ...]
ACTION_SIG = tuple[tuple[int, int], ...]


# ---------------------------------------------------------------------------
# Worker (module-level so it is picklable for spawn)
# ---------------------------------------------------------------------------
def _solve_one(args: tuple[HAND_KEY, int, int]) -> tuple[HAND_KEY, ACTION_SIG]:
    """Solve a single canonical street-1 hand. Module-scope for pickling.

    Returns ``(canonical_hand_key, canonical_action_signature)``.
    """
    canon_hand, n_rollouts, top_k = args

    # Lazy imports keep the spawned interpreter's startup cost low and
    # avoid pulling import-time side effects into the parent.
    from ai.heuristic_policy import HeuristicPolicy
    from ai.monte_carlo_policy import MCConfig, MonteCarloPolicy
    from state.game_state import GameState

    # Deterministic seed per canonical hand.
    digest = hashlib.blake2b(
        repr(canon_hand).encode("utf-8"), digest_size=4
    ).digest()
    seed = int.from_bytes(digest, "big")

    # Build the street-1 game state with the canonical hand as player 0's
    # pending. Player 1 receives 5 random cards from the rest of the deck
    # using a deterministic shuffle so two runs with the same hand match.
    gs = GameState.new(seed=seed)
    gs.deal_street()
    used = set(canon_hand)
    others = [c for c in range(54) if c not in used]
    random.Random(seed).shuffle(others)
    gs.hands[0].pending = list(canon_hand)
    gs.hands[1].pending = others[:5]
    gs.deck._cards = others[5:]  # type: ignore[attr-defined]

    pol = MonteCarloPolicy(
        config=MCConfig(n_rollouts=n_rollouts, top_k=top_k),
        completion_policy=HeuristicPolicy(seed=seed),
        seed=seed,
    )
    action = pol.act(gs, 0)
    # `canonical_action` here is just `tuple(sorted(placements))`.
    return canon_hand, tuple(sorted(action.placements))


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------
def _format_eta(elapsed: float, done: int, total: int) -> str:
    if done <= 0:
        return "ETA --:--:--"
    rate = done / elapsed
    remaining = (total - done) / rate
    h, rem = divmod(remaining, 3600)
    m, s = divmod(rem, 60)
    return f"ETA {int(h):02d}:{int(m):02d}:{int(s):02d}"


def build(
    out_path: Path,
    *,
    n_workers: int,
    n_rollouts: int,
    top_k: int,
    limit: Optional[int],
    resume: bool,
    checkpoint_every: int,
) -> None:
    from tables.canonical_opening import (
        CanonicalOpeningBookTable,
        enumerate_canonical_hands,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Enumerate (cheap: ~3 seconds).
    print("Enumerating canonical hands…", flush=True)
    t0 = time.perf_counter()
    all_hands = enumerate_canonical_hands()
    print(f"  {len(all_hands):,} canonical hands "
          f"(took {time.perf_counter() - t0:.1f}s)", flush=True)
    if limit is not None:
        all_hands = all_hands[:limit]
        print(f"  --limit set: solving only the first {len(all_hands):,}",
              flush=True)

    # Resume support.
    book: dict[HAND_KEY, ACTION_SIG] = {}
    if resume and out_path.is_file():
        with out_path.open("rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, CanonicalOpeningBookTable):
            book = dict(loaded.entries)
        elif isinstance(loaded, dict):
            book = loaded
        else:
            raise TypeError(
                f"resume: unrecognised pickle type {type(loaded).__name__}"
            )
        print(f"  resume: loaded {len(book):,} previously-solved hands",
              flush=True)
    pending = [h for h in all_hands if h not in book]
    print(f"  to solve: {len(pending):,} "
          f"(workers={n_workers}, n_rollouts={n_rollouts}, top_k={top_k})",
          flush=True)
    if not pending:
        print("Nothing to do. Saving and exiting.")
        _flush(out_path, book)
        return

    # Trap Ctrl+C: drain partial results then exit.
    interrupted = {"flag": False}

    def _handler(signum, frame):  # noqa: ARG001
        if interrupted["flag"]:
            # Second Ctrl+C: hard exit.
            print("\nForce exit.", flush=True)
            sys.exit(130)
        interrupted["flag"] = True
        print("\nInterrupt received — finishing in-flight tasks and "
              "checkpointing…", flush=True)

    signal.signal(signal.SIGINT, _handler)

    tasks = [(h, n_rollouts, top_k) for h in pending]
    total = len(book) + len(tasks)
    t_start = time.perf_counter()
    last_ckpt = len(book)

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        try:
            for canon_hand, action_sig in pool.imap_unordered(
                _solve_one, tasks, chunksize=4
            ):
                book[canon_hand] = action_sig
                done = len(book)
                if done % 100 == 0 or done == total:
                    elapsed = time.perf_counter() - t_start
                    n_new = done - (total - len(tasks))
                    per = elapsed / max(1, n_new)
                    eta = _format_eta(elapsed, n_new, len(tasks))
                    print(
                        f"  [{done:>6,}/{total:>6,}]  "
                        f"per-hand={per*1000:6.0f} ms  {eta}",
                        flush=True,
                    )
                if done - last_ckpt >= checkpoint_every:
                    _flush(out_path, book)
                    last_ckpt = done
                if interrupted["flag"]:
                    pool.terminate()
                    break
        finally:
            _flush(out_path, book)

    elapsed = time.perf_counter() - t_start
    print(f"\nSolved {len(book):,} hands in {elapsed/3600:.2f} h "
          f"({elapsed:.0f}s). Saved to {out_path}.", flush=True)

    # Sidecar metadata.
    meta = {
        "n_canonical_hands_total": 152_646,
        "n_solved": len(book),
        "n_rollouts": n_rollouts,
        "top_k": top_k,
        "n_workers": n_workers,
        "elapsed_s": elapsed,
        "elapsed_h": elapsed / 3600,
        "out_path": str(out_path),
    }
    sidecar = out_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2))
    print(f"Metadata written to {sidecar}.")


def _flush(out_path: Path, book: dict) -> None:
    # Save as a wrapped `CanonicalOpeningBookTable` so consumers can
    # `isinstance()`-dispatch on it without unwrapping the dict.
    from tables.canonical_opening import CanonicalOpeningBookTable

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(CanonicalOpeningBookTable(book), f, protocol=4)
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--out", type=Path,
        default=Path("artifacts/opening_book_canonical/"
                     "opening_book_canonical.pkl"),
        help="Destination pickle path.",
    )
    ap.add_argument(
        "--workers", type=int, default=16,
        help="Worker processes (default: 16).",
    )
    ap.add_argument(
        "--n-rollouts", type=int, default=60,
        help="Monte-Carlo rollouts per candidate (default: 60 = fast tier).",
    )
    ap.add_argument(
        "--top-k", type=int, default=5,
        help="Heuristic top-K prefilter (default: 5).",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Solve only the first N canonical hands (smoke testing).",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Resume: skip hands already in the output pickle.",
    )
    ap.add_argument(
        "--checkpoint-every", type=int, default=500,
        help="Flush the pickle every N newly-solved hands (default: 500).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    build(
        args.out.resolve(),
        n_workers=args.workers,
        n_rollouts=args.n_rollouts,
        top_k=args.top_k,
        limit=args.limit,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
