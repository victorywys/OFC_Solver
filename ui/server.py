"""Tiny stdlib HTTP server for the interactive UI.

Two HTTP servers run side-by-side in the same process:

    UI server  (default port 4040): serves the static page (HTML/CSS/JS).
                                    Injects the API base URL into the page
                                    so the JS knows where to POST.
    API server (default port 8180): serves /api/health and /api/analyze
                                    as JSON. Enables CORS so the UI page
                                    can call it cross-origin.

Run:
    python -m ui.server --run artifacts/run_<TS>/
    python -m ui.server --run artifacts/run_<TS>/ --ui-port 4040 --api-port 8180
    open http://localhost:4040/

Both servers share the same `_ServerState` (loaded tables + Analyzer).

Concurrency: the analyzer is reentrant-safe (thread-local diagnostic
counters, per-call RNG), so concurrent /api/analyze requests run in
parallel. A bounded semaphore (`--max-concurrent`) caps simultaneous
in-flight analyses to protect memory under bursts. Rollout work is
funneled through a single multiprocessing.Pool which fairly interleaves
chunks from different requests at its internal queue.
"""

from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import multiprocessing
import os
import threading
import time
import traceback
from http import HTTPStatus
from pathlib import Path
from typing import Optional

from ai.heuristic_policy import HeuristicPolicy
from fantasy.fantasy_solver import FantasySolverPolicy
from simulation.storage import load_run

from tables import (
    CanonicalOpeningBookTable,
    FantasyArrangementCache,
    FantasyEVTable,
    FoulProbTable,
    OpeningBookTable,
    PolicyPriorTable,
    TableAwareConfig,
    TableAwarePolicy,
    TranspositionTable,
)

from .analyzer import Analyzer
from .analyzer_fast import FastAnalyzer
from .state_builder import build_game_state


STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Server-wide state (built at startup)
# ---------------------------------------------------------------------------
class _ServerState:
    def __init__(
        self,
        run_dir: Optional[str],
        min_visits: dict,
        api_base: str,
        rollout_workers: int = 1,
        max_concurrent_analyze: int = 8,
    ) -> None:
        self.run_dir = run_dir
        self.api_base = api_base
        self.tables_loaded: list[str] = []

        opening_book = None
        policy_prior = None
        fantasy_cache = None
        foul_prob = None
        fantasy_ev = None
        if run_dir:
            blobs = load_run(run_dir)
            self.tables_loaded = sorted(blobs.keys())
            # Prefer the canonical opening book (152,646 entries, 100%
            # street-1 coverage, ~1us lookups). Fall back to the legacy
            # sampled book if only that is present.
            opening_book = blobs.get("opening_book_canonical")
            if not isinstance(opening_book, CanonicalOpeningBookTable):
                opening_book = blobs.get("opening_book") if isinstance(
                    blobs.get("opening_book"), OpeningBookTable
                ) else None
            policy_prior = blobs.get("policy_prior") if isinstance(
                blobs.get("policy_prior"), PolicyPriorTable
            ) else None
            fantasy_cache = blobs.get("fantasy_arrangement") if isinstance(
                blobs.get("fantasy_arrangement"), FantasyArrangementCache
            ) else None
            foul_prob = blobs.get("foul_prob") if isinstance(
                blobs.get("foul_prob"), FoulProbTable
            ) else None
            fantasy_ev = blobs.get("fantasy_ev") if isinstance(
                blobs.get("fantasy_ev"), FantasyEVTable
            ) else None

        config = TableAwareConfig(
            prior_min_visits=int(min_visits.get("prior", 1)),
            opening_min_visits=int(min_visits.get("opening", 1)),
        )
        # Wrap the heuristic in FantasySolverPolicy so fantasy-tier hands
        # use the exact (beam) solver instead of brute-force enumeration.
        heuristic = HeuristicPolicy(seed=0)
        fallback = FantasySolverPolicy(fallback=heuristic)
        self.policy = TableAwarePolicy(
            fallback=fallback,
            config=config,
            transposition=TranspositionTable(max_entries=200_000),
            opening_book=opening_book,
            fantasy_cache=fantasy_cache,
            policy_prior=policy_prior,
        )
        # Build a process pool for parallel rollouts when requested.
        # We use the "spawn" start method (not "fork"). Forking from a
        # multi-threaded parent is unsafe: if any background thread (e.g.
        # the Pool's own _handle_workers/_handle_results helpers, or any
        # http.server request thread) is holding a lock at fork time,
        # the child inherits the locked state and deadlocks forever.
        # We saw this manifest as repopulated workers stuck in
        # synchronize.py:95 __enter__ after a small number of analyses.
        self.pool = None
        if rollout_workers > 1:
            ctx = multiprocessing.get_context("spawn")
            self.pool = ctx.Pool(processes=rollout_workers)

        self.analyzer = Analyzer(
            policy=self.policy,
            foul_prob_table=foul_prob,
            policy_prior_table=policy_prior,
            fantasy_ev_table=fantasy_ev,
            rollout_seed=0,
            pool=self.pool,
            n_workers=rollout_workers,
        )
        # Fast analyzer: same policy + tables, but uses Common Random
        # Numbers across candidates, a cheap rank-based opponent in
        # rollouts, smart-skips when the prior is highly confident, and
        # defaults to a smaller top_k. Exposed via /api/analyze.
        self.fast_analyzer = FastAnalyzer(
            policy=self.policy,
            foul_prob_table=foul_prob,
            policy_prior_table=policy_prior,
            fantasy_ev_table=fantasy_ev,
            rollout_seed=0,
            pool=self.pool,
            n_workers=rollout_workers,
        )
        # Bounded semaphore caps simultaneous in-flight analyses to keep
        # memory and tail latency under control during bursts. Smart-skip
        # / opening-book hits release the slot within milliseconds; only
        # rollout-bound requests hold it long. Tune via --max-concurrent.
        self.analyze_semaphore = threading.BoundedSemaphore(
            value=max(1, int(max_concurrent_analyze))
        )
        # Legacy lock retained for ad-hoc serialized operations (currently
        # unused on the analyze path; analyzer is reentrant-safe).
        self.lock = threading.Lock()


# ---------------------------------------------------------------------------
def _send_json(handler, status: int, body: dict, *, cors: bool = False) -> None:
    data = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    if cors:
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(data)


def _send_bytes(handler, status: int, data: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _read_json_body(handler) -> dict:
    n = int(handler.headers.get("Content-Length", "0") or 0)
    if n <= 0:
        return {}
    raw = handler.rfile.read(n)
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON body: {e}")


def _ts() -> str:
    """Local-time timestamp for log lines: 'YYYY-MM-DD HH:MM:SS.mmm'."""
    t = time.time()
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', lt)}.{ms:03d}"


def _summarize_spec(spec) -> str:
    """Compact one-line summary of an analyze spec, for 400-error logs.

    Captures the fields most likely to cause `build_game_state` failures
    (street/to_act/board sizes/pending sizes/fantasy tiers) without
    dumping potentially-large raw lists.
    """
    if not isinstance(spec, dict):
        return f"<non-dict: {type(spec).__name__}>"
    try:
        parts = [
            f"street={spec.get('street')!r}",
            f"to_act={spec.get('to_act')!r}",
            f"auto_fill_opp={spec.get('auto_fill_opponent')!r}",
            f"dead={len(spec.get('dead_cards') or [])}",
        ]
        players = spec.get("players") or []
        if not isinstance(players, list):
            parts.append(f"players=<{type(players).__name__}>")
        else:
            for i, p in enumerate(players):
                if not isinstance(p, dict):
                    parts.append(f"p{i}=<{type(p).__name__}>")
                    continue
                board = p.get("board") or {}
                t = len(board.get("top") or [])
                m = len(board.get("middle") or [])
                b = len(board.get("bottom") or [])
                d = len(board.get("discards") or [])
                pend = len(p.get("pending") or [])
                parts.append(
                    f"p{i}=(T{t}/M{m}/B{b}/X{d}, pend={pend}, "
                    f"ft={p.get('fantasy_tier')!r})"
                )
        return "{" + ", ".join(parts) + "}"
    except Exception as e:  # never let the logger raise
        return f"<summarize-failed: {type(e).__name__}: {e}>"


# ---------------------------------------------------------------------------
# UI server (static files + injected API base)
# ---------------------------------------------------------------------------
class _UIHandler(http.server.BaseHTTPRequestHandler):
    server_state: _ServerState

    def log_message(self, fmt: str, *args) -> None:
        print(f"{_ts()} [ui]  {self.address_string()} - {fmt % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_index()
            return
        if path.startswith("/static/"):
            sub = path[len("/static/"):]
            target = (STATIC_DIR / sub).resolve()
            sd = STATIC_DIR.resolve()
            if target != sd and sd not in target.parents:
                self.send_error(HTTPStatus.FORBIDDEN, "outside static dir")
                return
            if not target.exists() or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, f"file not found: {sub}")
                return
            ctype, _ = mimetypes.guess_type(str(target))
            _send_bytes(self, HTTPStatus.OK, target.read_bytes(),
                        ctype or "application/octet-stream")
            return
        self.send_error(HTTPStatus.NOT_FOUND, f"unknown path: {path}")

    def _serve_index(self) -> None:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "index.html missing")
            return
        html = index_path.read_text(encoding="utf-8")
        # Inject window.OFC_API_BASE right after <head>. The page reads
        # this constant to build all fetch() calls. JSON-encoding the
        # string keeps quoting safe across hostnames.
        injection = (
            "<script>window.OFC_API_BASE = "
            f"{json.dumps(self.server_state.api_base)};</script>\n"
        )
        if "<head>" in html:
            html = html.replace("<head>", "<head>\n" + injection, 1)
        else:
            html = injection + html
        _send_bytes(self, HTTPStatus.OK, html.encode("utf-8"),
                    "text/html; charset=utf-8")


# ---------------------------------------------------------------------------
# API server (JSON, CORS-enabled)
# ---------------------------------------------------------------------------
class _APIHandler(http.server.BaseHTTPRequestHandler):
    server_state: _ServerState

    def log_message(self, fmt: str, *args) -> None:
        print(f"{_ts()} [api] {self.address_string()} - {fmt % args}", flush=True)

    def do_OPTIONS(self) -> None:  # noqa: N802 (CORS preflight)
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            _send_json(self, HTTPStatus.OK, {
                "ok": True,
                "run_dir": self.server_state.run_dir,
                "tables_loaded": self.server_state.tables_loaded,
            }, cors=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND, f"unknown path: {path}")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/analyze":
            self._handle_analyze(mode="fast")
            return
        if path == "/api/analyze_accurate":
            self._handle_analyze(mode="accurate")
            return
        self.send_error(HTTPStatus.NOT_FOUND, f"unknown path: {path}")

    def _handle_analyze(self, mode: str = "fast") -> None:
        try:
            body = _read_json_body(self)
        except ValueError as e:
            print(f"{_ts()} [api-400] {self.address_string()} read_json: {e}", flush=True)
            _send_json(self, HTTPStatus.BAD_REQUEST, {"error": str(e)}, cors=True)
            return

        spec = body.get("spec")
        if not isinstance(spec, dict):
            keys = list(body.keys()) if isinstance(body, dict) else type(body).__name__
            print(f"{_ts()} [api-400] {self.address_string()} missing-spec: body_keys={keys}", flush=True)
            _send_json(self, HTTPStatus.BAD_REQUEST,
                       {"error": "missing 'spec'"}, cors=True)
            return
        # The fast endpoint defaults to a smaller top_k; the accurate
        # endpoint keeps the original default of 5.
        default_top_k = 3 if mode == "fast" else 5
        # Default rollouts: 20 (fast) / 40 (accurate). Hard cap enforced
        # server-side so a misconfigured client cannot ask for hours of
        # CPU. The interactive UI's "Rollouts" slider is the typical
        # source; the cap protects shared capacity from any single user.
        # Fast-path cap raised to 40 after Stack B optimizations cut
        # solo latency ~47% (was 634ms, now ~337ms at 20 rollouts).
        default_n_rollouts = 20 if mode == "fast" else 40
        max_n_rollouts = 40 if mode == "fast" else 200
        n_rollouts = int(body.get("n_rollouts", default_n_rollouts))
        top_k = int(body.get("top_k", default_top_k))
        future_hands = int(body.get("future_hands", 0))
        n_rollouts = max(0, min(n_rollouts, max_n_rollouts))
        top_k = max(1, min(top_k, 50))
        # ``future_hands`` is freely configurable. -1 means infinite horizon
        # (converged value-function bonuses); any non-negative integer is
        # accepted and clamped only by a generous upper bound to keep the
        # finite-horizon iteration cheap.
        if future_hands < 0:
            future_hands = -1
        else:
            future_hands = min(future_hands, 100_000)

        try:
            gs = build_game_state(spec)
        except Exception as e:
            # Log a compact spec summary so we can diagnose what the
            # client sent. We avoid dumping the whole spec at INFO
            # because it can include large `pending`/`board` arrays;
            # the summary captures the fields most likely to be wrong.
            summary = _summarize_spec(spec)
            print(
                f"{_ts()} [api-400] {self.address_string()} "
                f"build_game_state: {type(e).__name__}: {e} | spec={summary}",
                flush=True,
            )
            _send_json(self, HTTPStatus.BAD_REQUEST, {"error": str(e)}, cors=True)
            return

        analyzer = (
            self.server_state.fast_analyzer if mode == "fast"
            else self.server_state.analyzer
        )

        # Admission control:
        #   * Try to acquire a rollout slot without blocking.
        #   * If the analyzer is saturated, fall back to a heuristic-only
        #     answer (no rollouts, no worker pool). Heuristic scoring is
        #     ~milliseconds of pure-Python work, GIL-bound, and scales to
        #     unbounded concurrent users; nobody waits.
        #   * The fast endpoint uses this fallback; the accurate endpoint
        #     keeps the blocking behavior (callers explicitly asked for
        #     quality).
        pol = self.server_state.policy
        sem = self.server_state.analyze_semaphore
        used_fallback = False
        got_slot = sem.acquire(blocking=(mode != "fast"))
        try:
            try:
                if got_slot:
                    result = analyzer.analyze(
                        gs, int(spec.get("to_act", 0)),
                        n_rollouts=n_rollouts,
                        top_k=top_k,
                        future_hands=future_hands,
                    )
                else:
                    # Overflow: heuristic-only path. Available only on
                    # FastAnalyzer.
                    used_fallback = True
                    result = analyzer.analyze_heuristic_only(
                        gs, int(spec.get("to_act", 0)),
                        top_k=top_k,
                    )
                lookups = {
                    "transposition_hits": pol.n_transposition_hits,
                    "opening_hits": pol.n_opening_hits,
                    "fantasy_hits": pol.n_fantasy_hits,
                    "prior_hits": pol.n_prior_hits,
                    "fallback_calls": pol.n_fallback_calls,
                }
            except Exception as e:
                traceback.print_exc()
                _send_json(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(e).__name__}: {e}"},
                    cors=True,
                )
                return
        finally:
            if got_slot:
                sem.release()

        out = result.to_dict()
        out["lookups"] = lookups
        out["mode"] = "heuristic" if used_fallback else mode
        out["heuristic_fallback"] = used_fallback
        out["opp_was_synthesized"] = bool(
            getattr(gs, "opp_was_synthesized", False)
        )
        _send_json(self, HTTPStatus.OK, out, cors=True)


# ---------------------------------------------------------------------------
# Combined handler (UI + API on one port). Used in single-port mode so the
# page can fetch the API at a same-origin relative URL, eliminating the
# need to forward two separate ports across machines.
# ---------------------------------------------------------------------------
class _CombinedHandler(_APIHandler):
    """Routes /, /index.html, /static/* to UI; /api/* to API."""

    def log_message(self, fmt: str, *args) -> None:
        print(f"{_ts()} [srv] {self.address_string()} - {fmt % args}", flush=True)

    def _is_api(self, path: str) -> bool:
        return path.startswith("/api/") or path == "/api"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if self._is_api(path):
            _APIHandler.do_GET(self)
            return
        _UIHandler.do_GET(self)

    def do_POST(self) -> None:  # noqa: N802
        _APIHandler.do_POST(self)

    def do_OPTIONS(self) -> None:  # noqa: N802
        _APIHandler.do_OPTIONS(self)

    def _serve_index(self) -> None:
        _UIHandler._serve_index(self)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="OFC interactive UI server.")
    parser.add_argument("--run", default=None,
                        help="Path to artifacts/run_<TS>/. If omitted, the AI"
                             " uses heuristic-only (no precomputed tables).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=4040,
                        help="Port for the static page (default 4040).")
    parser.add_argument("--api-port", type=int, default=8180,
                        help="Port for the JSON API (default 8180).")
    parser.add_argument("--api-host-public", default=None,
                        help="Hostname embedded in the page for fetch() to "
                             "the API. Defaults to --host. Set to a public "
                             "hostname when serving over the network.")
    parser.add_argument("--prior-min-visits", type=int, default=1)
    parser.add_argument("--opening-min-visits", type=int, default=1)
    parser.add_argument("--rollout-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Number of worker processes for parallel "
                             "rollouts. Default: cpu_count - 1. Set to 1 "
                             "to disable multiprocessing.")
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="Max concurrent /api/analyze* calls in flight "
                             "doing rollouts. When saturated, the fast "
                             "endpoint falls back to a heuristic-only "
                             "answer (no rollouts) for overflow requests "
                             "so users never wait in a queue. Default 4.")
    parser.add_argument("--two-port", action="store_true",
                        help="Run UI and API on separate ports (legacy). "
                             "By default both run on --ui-port so only one "
                             "port needs forwarding for remote access.")
    args = parser.parse_args()

    api_host_public = args.api_host_public or args.host
    if args.two_port:
        api_base = f"http://{api_host_public}:{args.api_port}"
    else:
        # Single-port mode: page uses same-origin relative URLs.
        api_base = ""

    state = _ServerState(
        run_dir=args.run,
        min_visits={
            "prior": args.prior_min_visits,
            "opening": args.opening_min_visits,
        },
        api_base=api_base,
        rollout_workers=args.rollout_workers,
        max_concurrent_analyze=args.max_concurrent,
    )
    print(f"[server] tables loaded: {state.tables_loaded}")
    print(f"[server] rollout workers: {args.rollout_workers}")
    print(f"[server] max concurrent analyze: {args.max_concurrent}")

    _UIHandler.server_state = state
    _APIHandler.server_state = state
    _CombinedHandler.server_state = state

    if args.two_port:
        ui_httpd = http.server.ThreadingHTTPServer((args.host, args.ui_port), _UIHandler)
        api_httpd = http.server.ThreadingHTTPServer((args.host, args.api_port), _APIHandler)
        api_thread = threading.Thread(
            target=api_httpd.serve_forever, name="api-server", daemon=True
        )
        api_thread.start()
        print(f"[server] UI  listening on http://{args.host}:{args.ui_port}/")
        print(f"[server] API listening on http://{api_host_public}:{args.api_port}/api/...")
        print(f"[server] open http://{args.host}:{args.ui_port}/ in a browser")
        try:
            ui_httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[server] shutting down")
        finally:
            ui_httpd.server_close()
            api_httpd.shutdown()
            api_httpd.server_close()
    else:
        # Single-port mode: one HTTP server handles both UI and API.
        # ThreadingHTTPServer so a single in-flight analyze can't block
        # /api/health, /static/*, or other concurrent requests. The
        # analyzer itself is still serialized via state.lock.
        httpd = http.server.ThreadingHTTPServer((args.host, args.ui_port), _CombinedHandler)
        print(f"[server] UI+API listening on http://{args.host}:{args.ui_port}/")
        print(f"[server] open http://{args.host}:{args.ui_port}/ in a browser")
        print(f"[server] (forward only port {args.ui_port} for remote access)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[server] shutting down")
        finally:
            httpd.server_close()


if __name__ == "__main__":
    main()
