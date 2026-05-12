"""Concurrency stress test for /api/analyze.

Fires N concurrent requests against the running server and reports:
  - per-request latency (min / median / p95 / max)
  - total wall time
  - speedup vs. serial baseline (extrapolated from median)

Run:
    python -m scripts.bench_concurrency --url http://127.0.0.1:8081 --n 8 --rollouts 80
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


# A reasonable street-1 spec that does require rollouts (not just a smart-skip
# / opening-book hit), so we measure the heavy path.
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


def post(url: str, body: dict, timeout: float = 180.0):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def one_request(url: str, n_rollouts: int, top_k: int):
    body = {"spec": SPEC, "n_rollouts": n_rollouts, "top_k": top_k, "future_hands": 0}
    t0 = time.perf_counter()
    status, resp = post(f"{url}/api/analyze", body)
    dt = time.perf_counter() - t0
    return dt, status, resp.get("elapsed_s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--n", type=int, default=8, help="concurrent clients")
    ap.add_argument("--rollouts", type=int, default=80)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    # Warm-up call (loads any first-touch caches in the worker pool)
    print("[warm-up]", end=" ", flush=True)
    dt, st, srv = one_request(args.url, args.rollouts, args.top_k)
    print(f"status={st} client={dt*1000:.0f}ms server={(srv or 0)*1000:.0f}ms")

    print(f"\n[serial baseline] 1 call:")
    dt, st, srv = one_request(args.url, args.rollouts, args.top_k)
    baseline_client = dt
    baseline_server = srv or 0.0
    print(f"  client={dt*1000:.0f}ms server={baseline_server*1000:.0f}ms")

    print(f"\n[concurrent] {args.n} simultaneous clients:")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as ex:
        futs = [ex.submit(one_request, args.url, args.rollouts, args.top_k)
                for _ in range(args.n)]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]
    wall = time.perf_counter() - t0

    latencies = [r[0] for r in results]
    server_times = [r[2] or 0.0 for r in results]
    latencies.sort()
    print(f"  wall    = {wall*1000:.0f} ms")
    print(f"  per-req client latency (ms): "
          f"min={latencies[0]*1000:.0f} "
          f"med={statistics.median(latencies)*1000:.0f} "
          f"p95={latencies[max(0, int(0.95*len(latencies))-1)]*1000:.0f} "
          f"max={latencies[-1]*1000:.0f}")
    print(f"  per-req server elapsed_s (ms): "
          f"min={min(server_times)*1000:.0f} "
          f"med={statistics.median(server_times)*1000:.0f} "
          f"max={max(server_times)*1000:.0f}")

    expected_serial = args.n * baseline_client
    speedup = expected_serial / wall if wall > 0 else 0.0
    print(f"\n  expected wall if serial : {expected_serial*1000:.0f} ms")
    print(f"  observed wall           : {wall*1000:.0f} ms")
    print(f"  end-to-end speedup       : {speedup:.2f}x")


if __name__ == "__main__":
    main()
