"""Test the heuristic-overflow fallback under burst load.

Sends N concurrent /api/analyze requests and reports per-request:
  - mode (fast vs heuristic)
  - client latency
  - server elapsed_s
  - whether heuristic_fallback flag set

Expected: with N > max-concurrent, the first ~4 take seconds (rollouts),
the rest return in milliseconds with heuristic_fallback=true.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request


SPEC = {
    "to_act": 0,
    "street": 1,
    "auto_fill_opponent": True,
    "players": [
        {
            "fantasy_tier": 0,
            "board": {"top": [], "middle": [], "bottom": [], "discards": []},
            "pending": ["As", "Kd", "Qh", "Jc", "Ts"],
        },
        {
            "fantasy_tier": 0,
            "board": {"top": [], "middle": [], "bottom": [], "discards": []},
            "pending": [],
        },
    ],
    "dead_cards": [],
}


def post(url: str, body: dict, timeout: float = 120.0):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def one(url: str, n_rollouts: int, top_k: int):
    body = {"spec": SPEC, "n_rollouts": n_rollouts, "top_k": top_k}
    t0 = time.perf_counter()
    try:
        status, resp = post(f"{url}/api/analyze", body)
        dt = time.perf_counter() - t0
        return {
            "ok": True,
            "client_ms": dt * 1000,
            "server_ms": (resp.get("elapsed_s") or 0) * 1000,
            "mode": resp.get("mode"),
            "fallback": resp.get("heuristic_fallback", False),
            "rec_ev": next((c["ev_mean"] for c in resp.get("candidates", []) if c.get("is_recommended")), None),
        }
    except Exception as e:
        return {"ok": False, "client_ms": (time.perf_counter() - t0) * 1000, "err": str(e)[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--n", type=int, default=10, help="concurrent clients")
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    print(f"Firing {args.n} concurrent requests (n_rollouts={args.rollouts}, top_k={args.top_k})\n")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as ex:
        futs = [ex.submit(one, args.url, args.rollouts, args.top_k) for _ in range(args.n)]
        rs = [f.result() for f in futs]
    wall = time.perf_counter() - t0

    print(f"{'#':>3}  {'mode':>10}  {'fallback':>8}  {'client_ms':>10}  {'server_ms':>10}  {'rec_ev':>8}")
    print("-" * 64)
    for i, r in enumerate(rs):
        if r["ok"]:
            print(f"{i:>3}  {r['mode']:>10}  {str(r['fallback']):>8}  "
                  f"{r['client_ms']:>10.0f}  {r['server_ms']:>10.0f}  "
                  f"{(r['rec_ev'] if r['rec_ev'] is not None else 0):>8.2f}")
        else:
            print(f"{i:>3}  ERROR: {r['err']} (client_ms={r['client_ms']:.0f})")

    n_ok = sum(1 for r in rs if r["ok"])
    n_fb = sum(1 for r in rs if r["ok"] and r["fallback"])
    n_real = sum(1 for r in rs if r["ok"] and not r["fallback"])
    print(f"\nwall: {wall * 1000:.0f} ms")
    print(f"ok={n_ok}/{args.n}  rollout-served={n_real}  heuristic-fallback={n_fb}")


if __name__ == "__main__":
    main()
