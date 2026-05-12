"""Public minimal API for the OFC Solver AI.

This module is the *single entry point* for external users who just want
to plug the AI into their own program. It wraps:

    * table loading             (from a directory of `*.pkl` files)
    * policy construction       (TableAwarePolicy + heuristic/fantasy fallback)
    * the rollout analyzer      (top-K candidates with Monte-Carlo EV)
    * a minimal multiprocessing pool (optional, opt-in via config)

Two public methods:

    ai.recommend(spec) -> dict      # best action only
    ai.analyze(spec)   -> dict      # full top-K analysis with stats

`spec` shape matches `ui/state_builder.build_game_state`:

    {
      "street": 3,
      "to_act": 0,
      "auto_fill_opponent": true,
      "dead_cards": ["2c", "7d"],          # optional
      "players": [
        {
          "fantasy_tier": 0,               # 0=NORMAL, 14..17=F14..F17
          "board": {"top": [...], "middle": [...],
                    "bottom": [...], "discards": [...]},
          "pending": ["6c", "6d", "9s"]
        },
        { ... }
      ]
    }

Configuration may be provided as:
    * a path to a JSON file (see `config.example.json`)
    * a plain dict
    * keyword arguments to `OFCAI.__init__`

All three coalesce into the `Config` dataclass below.

Quick start
-----------
    from ofc_ai import OFCAI

    ai = OFCAI(config="config.example.json")   # or OFCAI(tables_dir=..., n_rollouts=200)
    result = ai.analyze({
        "street": 1, "to_act": 0,
        "players": [
            {"fantasy_tier": 0, "board": {"top":[],"middle":[],"bottom":[],"discards":[]},
             "pending": ["As","Kd","Qh","Jc","Tc"]},
            {"fantasy_tier": 0, "board": {"top":[],"middle":[],"bottom":[],"discards":[]},
             "pending": []},
        ],
    })
    print(result["candidates"][0]["placements"])

Thread / process safety
-----------------------
A single `OFCAI` instance owns a `threading.Lock` around its analyzer
because the underlying `TableAwarePolicy` mutates diagnostic counters
per call. Concurrent `analyze`/`recommend` calls serialize cleanly.

Across processes, rollouts can be parallelized by setting `n_workers >
1` in the config; an internal `multiprocessing.Pool` (spawn context) is
built lazily on first use.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ai.heuristic_policy import HeuristicPolicy
from ai.policy import Policy
from fantasy.fantasy_solver import FantasySolverPolicy
from simulation.storage import load_run
from tables import (
    FantasyArrangementCache,
    FantasyEVTable,
    OpeningBookTable,
    CanonicalOpeningBookTable,
    PolicyPriorTable,
    TableAwareConfig,
    TableAwarePolicy,
    TranspositionTable,
)
from tables.foul_prob import FoulProbTable
from ui.analyzer import AnalysisResult, Analyzer
from ui.state_builder import build_game_state


__all__ = ["Config", "OFCAI", "AnalysisResult"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Runtime configuration for `OFCAI`.

    All fields have defaults; nothing is mandatory. Override via JSON,
    a dict, or keyword arguments to `OFCAI(...)`.
    """

    # ---- AI behaviour ----
    # Number of Monte-Carlo rollouts per candidate action. Higher = lower
    # variance but slower. 0 disables rollouts (heuristic ranking only).
    n_rollouts: int = 240
    # Keep the top-K heuristic candidates for rollout evaluation. The
    # table-recommended action is always added on top of the top-K set.
    top_k: int = 5
    # Future-hand horizon used to value entering / maintaining fantasy.
    # 0  -> ignore future hands (this hand only)
    # N  -> value next N hands using the FantasyEVTable
    # -1 -> infinite horizon (converged value function)
    future_hands: int = 5
    # Seed for the rollout RNG. Same seed + same state + same nested
    # policies = bit-deterministic output.
    rollout_seed: int = 0

    # ---- Parallelism ----
    # Number of worker processes for parallel rollouts. 1 = sequential
    # in the current process (no pool). >1 builds a multiprocessing.Pool
    # using the "spawn" start method.
    n_workers: int = 1

    # ---- Tables ----
    # Directory of `*.pkl` files produced by the bundled trainer. May be
    # None to disable all table lookups (pure heuristic + fantasy solver).
    # Missing pkls are silently ignored — only what's present is used.
    tables_dir: Optional[str] = None
    # Visit-count thresholds that gate table authority. Lower = trust
    # smaller-sample table entries (faster but noisier).
    prior_min_visits: int = 1
    opening_min_visits: int = 1
    # In-memory transposition-table size cap. None = unbounded.
    transposition_max_entries: Optional[int] = 200_000

    # ---- Output ----
    # If True, candidate placements are also returned as ("As", "top")
    # tuples for human-readable inspection in addition to raw card ids.
    include_card_strings: bool = True

    # ----- factories -----
    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        # Accept either a flat dict or sections ("ai", "tables", "parallel",
        # "output") for nicer config files.
        flat: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        known = {f for f in cls.__dataclass_fields__}
        ignored = set(flat) - known
        if ignored:
            # Don't fail — just inform the caller via a stderr-style warning.
            import sys
            print(
                f"[ofc_ai] warning: ignoring unknown config keys: "
                f"{sorted(ignored)}",
                file=sys.stderr,
            )
        return cls(**{k: v for k, v in flat.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
class OFCAI:
    """High-level OFC AI facade. Construct once, reuse across requests."""

    def __init__(
        self,
        config: str | Path | dict | Config | None = None,
        **overrides: Any,
    ) -> None:
        # Resolve config: file path / dict / Config instance / None, then
        # apply keyword overrides on top.
        if config is None:
            cfg = Config()
        elif isinstance(config, Config):
            cfg = cfg = Config(**{**asdict(config)})
        elif isinstance(config, dict):
            cfg = Config.from_dict(config)
        elif isinstance(config, (str, Path)):
            cfg = Config.from_json(config)
        else:
            raise TypeError(f"unsupported config type: {type(config).__name__}")
        if overrides:
            merged = {**asdict(cfg), **overrides}
            cfg = Config(**{k: v for k, v in merged.items()
                            if k in Config.__dataclass_fields__})
        self.config = cfg

        # ---- Load tables (gracefully skip whatever isn't present) ----
        blobs: dict[str, Any] = {}
        self.tables_loaded: list[str] = []
        if cfg.tables_dir:
            p = Path(cfg.tables_dir)
            if not p.is_dir():
                raise FileNotFoundError(
                    f"tables_dir not found: {cfg.tables_dir!r}"
                )
            blobs = load_run(p)
            self.tables_loaded = sorted(blobs.keys())

        def _typed(key: str, expected: type):
            v = blobs.get(key)
            return v if isinstance(v, expected) else None

        # Prefer the fully-precomputed canonical book if present.
        opening_book = _typed("opening_book_canonical", CanonicalOpeningBookTable)
        if opening_book is None:
            opening_book = _typed("opening_book", OpeningBookTable)
        policy_prior = _typed("policy_prior", PolicyPriorTable)
        fantasy_cache = _typed("fantasy_arrangement", FantasyArrangementCache)
        foul_prob = _typed("foul_prob", FoulProbTable)
        fantasy_ev = _typed("fantasy_ev", FantasyEVTable)

        # ---- Build the policy ----
        heuristic = HeuristicPolicy(seed=cfg.rollout_seed)
        fallback: Policy = FantasySolverPolicy(fallback=heuristic)
        ta_config = TableAwareConfig(
            prior_min_visits=cfg.prior_min_visits,
            opening_min_visits=cfg.opening_min_visits,
        )
        self.policy = TableAwarePolicy(
            fallback=fallback,
            config=ta_config,
            transposition=TranspositionTable(
                max_entries=cfg.transposition_max_entries
            ),
            opening_book=opening_book,
            fantasy_cache=fantasy_cache,
            policy_prior=policy_prior,
        )

        # ---- Lazy multiprocessing pool ----
        self._pool: Optional[multiprocessing.pool.Pool] = None
        self._pool_lock = threading.Lock()
        self._lock = threading.Lock()  # guards analyzer state across threads

        # ---- Analyzer ----
        self.analyzer = Analyzer(
            policy=self.policy,
            foul_prob_table=foul_prob,
            policy_prior_table=policy_prior,
            fantasy_ev_table=fantasy_ev,
            rollout_seed=cfg.rollout_seed,
            pool=None,           # filled in on first analyze() if n_workers > 1
            n_workers=cfg.n_workers,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def analyze(self, spec: dict) -> dict:
        """Full analysis. Returns a JSON-serializable dict.

        Top-level keys mirror `ui.analyzer.AnalysisResult.to_dict()`:
            player, n_players, street, fantasy_tier, n_legal_actions,
            n_evaluated, n_rollouts_per_action, future_hands, elapsed_s,
            tier_horizon_values, candidates: [ ... ].

        Each candidate has `placements`, `heuristic_score`, `ev_mean`,
        `ev_stderr`, `foul_rate`, `fantasy_entry_rate`, `horizon_ev`,
        `combined_ev`, `is_recommended`, ...
        """
        gs = build_game_state(spec)
        player = int(spec.get("to_act", 0))
        self._ensure_pool()
        with self._lock:
            result = self.analyzer.analyze(
                gs,
                player,
                n_rollouts=self.config.n_rollouts,
                top_k=self.config.top_k,
                future_hands=self.config.future_hands,
            )
        return result.to_dict()

    def recommend(self, spec: dict) -> dict:
        """Just the best action. Equivalent to `analyze(spec)["candidates"][0]`
        but skips the wider top-K enumeration if rollouts are disabled.
        """
        if self.config.n_rollouts <= 0:
            # Fast path: don't run rollouts, just use the policy's pick.
            gs = build_game_state(spec)
            player = int(spec.get("to_act", 0))
            with self._lock:
                # Plumb horizon values so the canonical book's
                # lookup_horizon path can re-rank stored candidates
                # before policy.act() returns.
                tier_values: dict = {}
                fev = getattr(self.analyzer, "fantasy_ev_table", None)
                if (
                    self.config.future_hands != 0
                    and fev is not None
                ):
                    tier_values = fev.horizon_value_relative(
                        self.config.future_hands
                    )
                try:
                    self.policy.set_horizon_values(tier_values)
                except AttributeError:
                    pass
                try:
                    action = self.policy.act(gs, player)
                finally:
                    # Reset to avoid leaking horizon between calls.
                    try:
                        self.policy.set_horizon_values(None)
                    except AttributeError:
                        pass
            from engine.cards import card_str
            from state.board import SLOT_NAMES
            return {
                "placements": [
                    {"card": c, "card_str": card_str(c),
                     "slot": s, "slot_str": SLOT_NAMES[s]}
                    for c, s in action.placements
                ],
                "is_recommended": True,
            }
        result = self.analyze(spec)
        # The top candidate after sort is the best by combined EV; the
        # `is_recommended` flag marks the policy's pick. We surface the
        # policy's pick as the recommendation (matches UI semantics).
        for c in result["candidates"]:
            if c.get("is_recommended"):
                return c
        return result["candidates"][0]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Shut down the worker pool (if any). Safe to call multiple times."""
        with self._pool_lock:
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
                self._pool = None
                self.analyzer.pool = None

    def __enter__(self) -> "OFCAI":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_pool(self) -> None:
        """Build the multiprocessing pool on first analyze() if needed."""
        if self.config.n_workers <= 1:
            return
        with self._pool_lock:
            if self._pool is not None:
                return
            # Spawn (not fork) so forking from a multi-threaded parent
            # doesn't inherit locked sync primitives.
            ctx = multiprocessing.get_context("spawn")
            self._pool = ctx.Pool(processes=self.config.n_workers)
            self.analyzer.pool = self._pool
