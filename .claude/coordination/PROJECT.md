# Project context

> Read this first when you join this repo as an agent. It captures what the
> project is for and how collaborating agents should think about it.

## Summary

OFC Solver is an Open-Face Chinese Poker (Pineapple + Fantasyland) AI
framework — a pure-Python game engine, several decision policies
(heuristic, Monte-Carlo with rollouts, fast-opponent), an exact
Fantasyland solver, precomputed lookup tables, a self-play harness, a
local HTTP analysis server, and a public `OFCAI` Python API for
embedding into other apps. Python 3.10+, no third-party runtime
dependencies; multiprocessing uses the `"spawn"` start method (fork
deadlocks against the analyzer's internal thread state). The runtime
tables under `artifacts/` are large and managed out-of-band (gitignored);
they are rebuilt by the scripts under `scripts/`.

## Stack

Python 3.10+ (PEP 604 union types), pure stdlib runtime; pytest for tests.

## Key components

- `engine/` + `state/` — game model (cards, deck, evaluator, royalties, board, actions)
- `ai/` + `fantasy/` — decision policies and the exact Fantasyland solver
- `tables/` + `scripts/` — precomputed lookup tables and their builders / benchmarks
- `simulation/` — self-play harness and trace collectors
- `ui/` + `ofc_ai.py` — HTTP analysis server (with `FastAnalyzer` and bounded-semaphore concurrency) and the public `OFCAI` Python API
- `tests/` — pytest suite (200+ tests)
- `docs/API.md` — HTTP API reference for `ui/server.py`

## Notes for collaborating agents

None yet — add as conventions emerge.
