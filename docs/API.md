# OFC Solver — HTTP API Reference

The OFC Solver exposes a small JSON-over-HTTP API that lets a remote agent ask
"given this Open-Face Chinese Poker position, what should the player to act do,
and how good is each option?".

This document covers everything an external agent needs to call the API
correctly: card encoding, request schema, response schema, error handling,
and ready-to-paste `curl` test commands.

---

## 1. Connection

| Item | Value |
|---|---|
| Protocol | HTTP/1.1, JSON bodies |
| API host (bind) | `0.0.0.0:8081` on the solver machine |
| Public base URL | `http://20.189.249.106:8081` |
| CORS | `Access-Control-Allow-Origin: *` on all `/api/*` responses |
| Auth | None (intended for trusted network) |
| Content type | `application/json; charset=utf-8` |

The solver is reachable at the public IP **`20.189.249.106`**. All examples
in this document use that address. If you are running on the solver machine
itself, `127.0.0.1` works equivalently.

There is also a separate UI on port `4040` (`http://20.189.249.106:4040/`)
which is a human-facing web page — it is not part of the API and agents do
not need to call it.

---

## 2. Endpoints overview

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/health` | Liveness + loaded tables |
| `POST` | `/api/analyze` | **Fast** analyzer (default) — interactive use |
| `POST` | `/api/analyze_accurate` | **Accurate** analyzer — higher quality, slower |
| `OPTIONS` | `/api/*` | CORS preflight (returns 204) |

Both analyze endpoints accept the **same request body** and return the
**same response shape**. They differ only in how rollouts are run:

| | `/api/analyze` (fast) | `/api/analyze_accurate` |
|---|---|---|
| Opponent in rollouts | cheap rank-based policy | full heuristic policy |
| Variance reduction | Common Random Numbers across candidates | independent draws |
| Smart-skip on confident prior | yes (returns prior top-K when visits ≥ 100) | no |
| Default `top_k` | 3 | 5 |
| Typical street-3 latency (2 CPU) | ~1–2 s | ~6 s |
| When to use | interactive UI, agent loops, fast iteration | offline analysis, final-answer decisions |

Unknown paths return `404`. The server never returns redirects.

---

## 3. Card encoding

Cards are 2-character strings:

```
<rank><suit>     example: "As", "Td", "2c", "9h"
```

- **Rank**: one of `2 3 4 5 6 7 8 9 T J Q K A` (uppercase `T` for ten).
- **Suit**: one of `c d h s` (lowercase: clubs, diamonds, hearts, spades).

All card lists in requests are arrays of these strings. The API also accepts
raw integer card ids `0..51` (rank*4 + suit, with `c=0, d=1, h=2, s=3`), but
strings are clearer and recommended.

A **slot** id used in responses:

| Id | Name | Meaning |
|---|---|---|
| 0 | `top` | top row (3-card row) |
| 1 | `middle` | middle row (5-card row) |
| 2 | `bottom` | bottom row (5-card row) |
| 3 | `discard` | the card thrown away on streets 2–5 |

---

## 4. `GET /api/health`

Returns immediately. Use this to verify connectivity and to learn which
precomputed tables are loaded on the server.

### Response 200

```json
{
  "ok": true,
  "run_dir": "artifacts/run_100k_20260508_103044/",
  "tables_loaded": [
    "fantasy_arrangement", "fantasy_ev", "fantasy_transitions",
    "foul_by_tier", "foul_prob", "match_summary", "metadata",
    "opening_book", "policy_prior", "royalty_by_row"
  ]
}
```

A missing or extra table changes the values reported as `table_*` in
`/api/analyze` responses (they become `null` if the corresponding table
is absent) but never breaks the request.

---

## 5. `POST /api/analyze` and `POST /api/analyze_accurate`

Both endpoints accept the same JSON body and return the same response
shape. The only behavioural difference is described in §2.

```jsonc
{
  "spec": { ... position description, see §5.1 ... },
  "n_rollouts": 80,        // optional, default 80,  clamped to 0..4000
  "top_k": 3,              // optional, default 3 for /api/analyze,
                           //                   5 for /api/analyze_accurate;
                           //           clamped to 1..50
  "future_hands": 0        // optional, default 0,   clamped to 0..200
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `spec` | object | **required** | The OFC position. See §5.1. |
| `n_rollouts` | int | 80 | Monte-Carlo rollouts per candidate. `0` skips rollouts and returns table/heuristic-only scores. |
| `top_k` | int | 5 | Maximum number of candidates returned. Always includes the recommended action. |
| `future_hands` | int | 0 | Discount horizon for fantasy-mode EV bonus. `0` evaluates only the current hand. |

### 5.1 `spec` — position description

```jsonc
{
  "street": 3,                       // 1..5, the street being played now
  "to_act": 0,                       // 0 or 1 — which player asks for advice
  "auto_fill_opponent": true,        // default true; see below
  "dead_cards": ["2c", "7d"],        // optional; cards seen elsewhere
  "players": [
    {
      "fantasy_tier": 0,             // 0, 14, 15, 16, or 17
      "board": {
        "top":      ["As", "Kc"],
        "middle":   ["Qh", "Jd", "Ts"],
        "bottom":   ["7s", "7c", "8h"],
        "discards": ["2c", "3d"]
      },
      "pending":   ["6c", "6d", "9s"]   // cards in hand on this street
    },
    { /* same shape for player 1 */ }
  ]
}
```

Fields:

- **`street`** (int, required): `1..5`. Determines pending size, capacity
  checks, and rollout depth.
- **`to_act`** (int, required): which player (`0` or `1`) is asking the
  solver for advice. Their `pending` must be non-empty.
- **`auto_fill_opponent`** (bool, default `true`): if the *other* player is
  fully empty (no board, no pending) and `street >= 2`, the solver
  synthesizes their prior streets by heuristic self-play, then deals them
  a current-street hand. Set `false` to disable.
- **`dead_cards`** (array, optional): cards known to be out of play (e.g.
  burned/seen elsewhere). They are excluded from rollouts' future draws.
- **`players[i].fantasy_tier`** (int): `0` = normal play; `14/15/16/17` =
  fantasy hand size. Only player `0` is allowed to be in fantasy mode in
  most realistic setups; the solver tolerates either seat.
- **`players[i].board.{top,middle,bottom}`**: cards already placed in each
  row. Row capacity limits: top ≤ 3, middle ≤ 5, bottom ≤ 5.
- **`players[i].board.discards`**: cards already discarded on prior streets.
- **`players[i].pending`**: cards in hand to be placed this street.

Pending size requirements for `to_act`:

| Situation | Required pending length |
|---|---|
| Normal play, street 1 | 5 |
| Normal play, streets 2–5 | 3 |
| Fantasy tier `t` | `t` cards (14/15/16/17) |

Placed-card count (for normal play, validated before the action):

| Street | Cards already on rows | Cards already discarded |
|---|---|---|
| 1 | 0 | 0 |
| 2 | 5 | 0 |
| 3 | 7 | 1 |
| 4 | 9 | 2 |
| 5 | 11 | 3 |

If these don't match, the API returns `400` with an explanatory error
message — see §7.

No card may appear twice across all known positions (both players' rows,
discards, pendings, plus `dead_cards`). Duplicates → `400`.

### 5.2 Response 200

The full response is a JSON object combining the analyzer output and a
small per-policy statistics block:

```jsonc
{
  "player": 0,
  "n_players": 2,
  "street": 3,
  "fantasy_tier": 0,
  "n_legal_actions": 42,
  "n_evaluated": 5,
  "n_rollouts_per_action": 80,
  "future_hands": 0,
  "elapsed_s": 1.92,
  "state_table_foul_prob": 0.073,    // null if FoulProbTable absent
  "state_table_prior_visits": 1238,  // 0 if PolicyPriorTable absent
  "tier_horizon_values": {},         // populated only when future_hands>0

  "candidates": [
    {
      "placements": [
        {"card": 19, "card_str": "7d", "slot": 2, "slot_str": "B"},
        {"card": 22, "card_str": "8h", "slot": 1, "slot_str": "M"},
        {"card": 41, "card_str": "Ks", "slot": 3, "slot_str": "X"}
      ],
      "heuristic_score": 12.4,
      "n_rollouts": 80,
      "ev_mean": 3.21,
      "ev_stderr": 0.62,
      "foul_rate": 0.0125,
      "fantasy_entry_rate": 0.075,
      "dest_tier_counts": {"0": 74, "14": 6},
      "horizon_ev": 0.0,
      "combined_ev": 3.21,
      "table_foul_prob": 0.011,
      "table_prior_visits": 187,
      "table_prior_mean_ev": 2.94,
      "is_recommended": true
    },
    { ... next-best candidate ... }
  ],

  "lookups": {
    "transposition_hits": 0,
    "opening_hits": 1,
    "fantasy_hits": 0,
    "prior_hits": 0,
    "fallback_calls": 0
  },
  "mode": "fast",
  "opp_was_synthesized": false
}
```

Top-level fields:

| Field | Meaning |
|---|---|
| `player` | Echo of `spec.to_act`. |
| `n_players` | Always `2` in the current rules. |
| `street`, `fantasy_tier` | Echo of input. |
| `n_legal_actions` | Total number of legal placements enumerated. |
| `n_evaluated` | `<= top_k`; how many made it into `candidates`. |
| `n_rollouts_per_action` | Echo of the rollouts setting. |
| `future_hands` | Echo. |
| `elapsed_s` | Wall-clock seconds spent in the analyzer. |
| `state_table_foul_prob` | Per-state foul probability for the *current* board (from `FoulProbTable`), or `null`. |
| `state_table_prior_visits` | Self-play support count for this state. |
| `tier_horizon_values` | Per-tier future-hand EV when `future_hands > 0`; otherwise `{}`. |

Each entry in `candidates` describes one specific placement of all pending
cards for the current street. Key fields:

| Field | Meaning |
|---|---|
| `placements[]` | Where each pending card goes. `slot_str` is `T`, `M`, `B`, or `X` (discard). |
| `heuristic_score` | Score from the heuristic policy (used to rank before rollouts). |
| `n_rollouts` | Actual rollouts completed for this candidate. |
| `ev_mean` / `ev_stderr` | Monte-Carlo EV (in points) and its standard error. |
| `foul_rate` | Probability this seat fouls by the end of the hand. |
| `fantasy_entry_rate` | Probability this seat enters fantasy next hand. |
| `dest_tier_counts` | Histogram of next-hand fantasy tier (keys `"0"`, `"14"`, …). |
| `horizon_ev` | Discounted future-hand bonus (only ≠ 0 if `future_hands > 0`). |
| `combined_ev` | `ev_mean + horizon_ev` — the headline number to maximize. |
| `table_foul_prob` | Per-action foul probability from `FoulProbTable`, or `null`. |
| `table_prior_visits` | Visit count for this action in `PolicyPriorTable`. |
| `table_prior_mean_ev` | Long-run mean EV for this action from the prior, or `null`. |
| `is_recommended` | `true` exactly once: the candidate the solver recommends. |

The recommended action is the one chosen by the table-aware policy. It is
**not always** the candidate with the highest `combined_ev` — for low
rollout budgets, the prior + opening-book / fantasy-cache lookups may
disagree with the small-sample MC estimate.

`lookups` is a small diagnostic block reporting which precomputed
caches got a hit while answering this request.

`mode` echoes which endpoint produced the response: `"fast"` for
`/api/analyze`, `"accurate"` for `/api/analyze_accurate`. Useful when
both are called from the same agent.

`opp_was_synthesized` is `true` iff the response was computed with a
heuristically-filled opponent (see `auto_fill_opponent`). Treat the
answer with proportionally more uncertainty when this flag is set.

When `n_rollouts_per_action == 0` in a fast-endpoint response, the
analyzer **smart-skipped** rollouts because the prior table was
confident at this state. In that case `ev_mean` comes from the prior's
recorded mean, `foul_rate` and `dest_tier_counts` will be zero/empty,
and `table_prior_visits` indicates how strong the underlying support is.

### 5.3 Error responses

All errors are JSON with a single `error` string and CORS headers set.

| Code | When |
|---|---|
| `400` | Invalid JSON body, missing `spec`, malformed card string, capacity violation, pending size mismatch, duplicate card, illegal `to_act`, etc. |
| `404` | Unknown path. |
| `500` | Unexpected exception in the analyzer (also printed to server stderr). |

Example 400:

```json
{ "error": "street 3 expects 7 placed cards before this action, but board has 5" }
```

---

## 6. Calling conventions

- Send `Content-Type: application/json`.
- One request per HTTP call; pipelining is not required.
- The analyzer is serialized server-side by a single lock; concurrent
  callers will queue. There is no per-IP rate limit, but expect
  per-request latency from ~0.5 s (`n_rollouts=0`) to several seconds
  for `n_rollouts >= 1000`.
- Keep request bodies under ~64 KB.
- Connections are short-lived. Open a new one per request, or reuse a
  keep-alive connection if your client supports it.

---

## 7. Test commands

All commands target the public host `20.189.249.106`. From a machine on the
same VM you can replace it with `127.0.0.1`. The server prints `[api]` log
lines for each request to stdout, so a successful call is easy to verify on
the server side too.

### 7.1 Health check

```bash
curl -s http://20.189.249.106:8081/api/health | python3 -m json.tool
```

Expected: `"ok": true` and a non-empty `tables_loaded` list.

### 7.2 Street 1 — fresh hand, no opponent, recommend top action

This is the simplest possible analyze call: player 0 has just been dealt
five cards and needs to place them.

```bash
curl -s -X POST http://20.189.249.106:8081/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "spec": {
      "street": 1,
      "to_act": 0,
      "auto_fill_opponent": true,
      "players": [
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": ["As", "Ah", "Kd", "Qc", "2s"]},
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": []}
      ]
    },
    "n_rollouts": 40,
    "top_k": 3
  }' | python3 -m json.tool | head -80
```

### 7.3 Mid-game — street 3 with both players partially built

Player 0 has 7 cards on the board + 1 discard, holding 3 pending. We let the
solver synthesize the opponent.

```bash
curl -s -X POST http://20.189.249.106:8081/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "spec": {
      "street": 3,
      "to_act": 0,
      "auto_fill_opponent": true,
      "players": [
        {"fantasy_tier": 0,
         "board": {
           "top":    ["As", "Kc"],
           "middle": ["Qh", "Jd", "Ts"],
           "bottom": ["7s", "7c"],
           "discards": ["2c"]
         },
         "pending": ["6c", "6d", "9s"]},
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": []}
      ]
    },
    "n_rollouts": 80,
    "top_k": 5
  }' | python3 -m json.tool
```

### 7.4 Street 5 — only legal placements left

Last street: 1 card discarded, 2 placed. Useful for sanity checks because
the analyzer should converge fast and `n_legal_actions` is small.

```bash
curl -s -X POST http://20.189.249.106:8081/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "spec": {
      "street": 5,
      "to_act": 0,
      "auto_fill_opponent": true,
      "players": [
        {"fantasy_tier": 0,
         "board": {
           "top":    ["As", "Kc", "9c"],
           "middle": ["Qh", "Jd", "Ts", "9d"],
           "bottom": ["7s", "7c", "8h", "8d"],
           "discards": ["2c", "3d", "4h"]
         },
         "pending": ["6c", "6d", "5s"]},
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": []}
      ]
    },
    "n_rollouts": 200,
    "top_k": 5
  }' | python3 -m json.tool
```

### 7.5 Fantasy hand (14-card tier)

Player 0 enters fantasy with 14 cards in hand. Pending must have exactly 14
cards; the boards start empty.

```bash
curl -s -X POST http://20.189.249.106:8081/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "spec": {
      "street": 1,
      "to_act": 0,
      "auto_fill_opponent": true,
      "players": [
        {"fantasy_tier": 14,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": ["As","Ah","Ad","Ks","Kh","Kd","Qs","Qh","Qd","Js","Jh","Jd","2c","3c"]},
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": []}
      ]
    },
    "n_rollouts": 0,
    "top_k": 3
  }' | python3 -m json.tool
```

Note `n_rollouts: 0` — fantasy candidates are produced by the exact-beam
solver and tables; rollouts add little for these.

### 7.6 With horizon (future-hand) discount

Same as 7.3 but asks the solver to value 5 future hands of fantasy income.

```bash
curl -s -X POST http://20.189.249.106:8081/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "spec": {
      "street": 3,
      "to_act": 0,
      "auto_fill_opponent": true,
      "players": [
        {"fantasy_tier": 0,
         "board": {
           "top":    ["Ah", "Ad"],
           "middle": ["Qh", "Jd", "Ts"],
           "bottom": ["7s", "7c"],
           "discards": ["2c"]
         },
         "pending": ["Kc", "Kd", "9s"]},
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": []}
      ]
    },
    "n_rollouts": 80,
    "top_k": 3,
    "future_hands": 5
  }' | python3 -m json.tool
```

Compare `ev_mean` vs. `combined_ev` to see how much the horizon bonus
moves each candidate.

### 7.7 Deliberate error — duplicate card

For confirming error handling:

```bash
curl -s -X POST http://20.189.249.106:8081/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "spec": {
      "street": 1, "to_act": 0,
      "players": [
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": ["As", "As", "Kd", "Qc", "2s"]},
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": []}
      ]
    }
  }'
```

Expected response: HTTP 400 with `{"error": "duplicate cards ..."}`.

### 7.8 CORS preflight

```bash
curl -s -i -X OPTIONS http://20.189.249.106:8081/api/analyze \
  -H 'Origin: http://example.com' \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: Content-Type'
```

Expected: `HTTP/1.0 204 No Content` plus `Access-Control-Allow-*` headers.

### 7.9 Accurate endpoint (same body, slower, higher quality)

```bash
curl -s -X POST http://20.189.249.106:8081/api/analyze_accurate \
  -H 'Content-Type: application/json' \
  -d '{
    "spec": {
      "street": 3,
      "to_act": 0,
      "auto_fill_opponent": true,
      "players": [
        {"fantasy_tier": 0,
         "board": {
           "top":    ["As", "Kc"],
           "middle": ["Qh", "Jd", "Ts"],
           "bottom": ["7s", "7c"],
           "discards": ["2c"]
         },
         "pending": ["6c", "6d", "9s"]},
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": []}
      ]
    },
    "n_rollouts": 240,
    "top_k": 5
  }' | python3 -m json.tool
```

Compare the recommended action and its `ev_mean`/`ev_stderr` against
the same spec hit through `/api/analyze`. They should agree on the top
recommendation in most cases; if they disagree, the accurate endpoint
is the tie-breaker.

The response includes `"mode": "accurate"` so you can confirm which
endpoint answered when scripting against both.

---

## 8. Minimal Python client

```python
import json
import urllib.request

API = "http://20.189.249.106:8081"

def analyze(spec, n_rollouts=80, top_k=5, future_hands=0):
    payload = json.dumps({
        "spec": spec,
        "n_rollouts": n_rollouts,
        "top_k": top_k,
        "future_hands": future_hands,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/api/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

result = analyze({
    "street": 1,
    "to_act": 0,
    "auto_fill_opponent": True,
    "players": [
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": ["As", "Ah", "Kd", "Qc", "2s"]},
        {"fantasy_tier": 0,
         "board": {"top": [], "middle": [], "bottom": [], "discards": []},
         "pending": []},
    ],
}, n_rollouts=40, top_k=3)

best = next(c for c in result["candidates"] if c["is_recommended"])
for p in best["placements"]:
    print(p["card_str"], "->", p["slot_str"])
print("EV:", best["combined_ev"], "+/-", best["ev_stderr"])
```

---

## 9. Operational notes for callers

- **Latency tuning**: `n_rollouts` is the main lever. `0` ≈ heuristic + tables
  only; `40–80` is good for interactive play; `400+` is for offline analysis.
- **Determinism**: rollouts share a fixed seed per server, so identical
  requests return identical numbers as long as the server hasn't been
  restarted with a different seed.
- **Synthesized opponents**: if you have any information about the
  opponent's board, pass it. `opp_was_synthesized: true` in the response
  means the solver had to guess.
- **State staleness**: the API has no notion of "session". Every call is a
  one-shot analysis of a fully described position.
- **Concurrency**: one request at a time will execute; additional callers
  block on a lock. Plan your worker count accordingly.
