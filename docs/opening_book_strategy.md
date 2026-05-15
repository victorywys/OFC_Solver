# OFC Opening Strategy — derived from the rich canonical opening book (v2)

**Source artefact.** [artifacts/opening_book_canonical_v2/opening_book_canonical.pkl](../artifacts/opening_book_canonical_v2/opening_book_canonical.pkl)
— 152,646 canonical street-1 orbits under suit (`S₄`) and joker (`S₂`) symmetry, each carrying the **top-5 candidate placements** from a heuristic prefilter, every candidate scored with **120 CRN rollouts** (`ev_mean`, `ev_se`, `foul_rate`, `fantasy_entry_rate`, full destination-tier histogram). Built in 8.09 h with 20 workers on commit `712587a`.

**Reproducer.**
```bash
python -m scripts.analyze_opening_book \
    --book artifacts/opening_book_canonical_v2/opening_book_canonical.pkl \
    --out  artifacts/opening_book_analysis.json
```
Generated stats are in [artifacts/opening_book_analysis.json](../artifacts/opening_book_analysis.json) (`~3.3 s` walk over 152k orbits).

---

## TL;DR (for humans)

1. **Bottom-row is the default.** In **85 %** of orbits the optimal action puts ≥ 2 cards on the bottom row, and in **38 %** of orbits the top row is left empty entirely.
2. **Three placement archetypes cover 91 % of opening hands**:
   `T=1 M=2 B=2` (50 %) · `T=0 M=2 B=3` (33 %) · `T=3 M=1 B=1` (8 %).
3. **Pairs partition cleanly by tier**:
   - **AA / KK / QQ** → typically go on **top** (T-T) in **85 / 82 / 29 %** of orbits, attempting fantasy entry.
   - **JJ down to 22** → almost always on **bottom** (B-B) in **88 – 91 %** of orbits.
4. **Trips → bottom**, near-universally (96 %); fantasy-17 by topping trips is only locally chosen ≤ 1 % of orbits per rank.
5. **Jokers are huge**: 1-joker hands average **+6.44 EV** vs **+1.21** for 0-joker hands.
6. **Decision is usually trivial.** Median EV-gap between best and 2nd-best stored candidate is **0.43**; the 95th-percentile is **3.85**. Most hands have one clearly best play.
7. **⚠ Discovered limitation.** For high-pair hands paired with high cards (e.g. `AA + K Q x`), the heuristic prefilter is biased toward AA-on-top and **never enumerates the AA-on-bottom alternative** which an unrestricted MC scan finds is **~10 EV better per hand**. The worst-EV orbits in the book are direct manifestations of this. See [§8 Known bug](#8-known-limitation--heuristic-prefilter-bias-toward-pair-on-top) for details and proposed fix.

---

## TL;DR (for the LLM downstream of this book)

Each `CanonicalOpeningBookTable` entry maps a canonical 5-card key to a sorted-by-`ev_mean` tuple of `CandidateRecord`s. Use:

- `book.lookup(hand)` → per-hand argmax (horizon = 0).
- `book.lookup_horizon(hand, tier_horizon_values=…)` → re-rank by `ev_mean + Σₜ P(next_tier=t | a) · horizon_value[t]`. The horizon values come from `FantasyEVTable.horizon_value_relative(H)`.
- `book.candidates(hand)` → real-suit-resolved list of all stored candidates (for explanations / "why" UI).

**Bias warning.** Only the top-5 by heuristic `score_action` are present. For hands where the heuristic ranking misorders alternatives (most prominently AA/KK + high cards), the book's best is locally optimal *within the prefilter set* but globally suboptimal. Do not trust the book as an oracle for those hands; flag if `n_candidates < ~20` and `foul_rate > 0.5`.

---

## 1. Global statistics

| Metric (best action per orbit) | min | p05 | p25 | median | mean | p75 | p95 | max | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ev_mean` (chips/hand) | -25.13 | -5.68 | -1.03 | **+1.76** | **+1.80** | +4.94 | +8.78 | +31.06 | 4.41 |
| `foul_rate` | 0 | 0.03 | 0.08 | 0.12 | 0.19 | 0.19 | 0.79 | 0.96 | 0.21 |
| `fantasy_entry_rate` | 0 | 0 | 0.01 | 0.03 | 0.05 | 0.08 | 0.18 | 0.59 | 0.06 |

- **102,497 / 152,646 (67 %)** orbits have **positive** best-action EV. **50,145 (33 %)** are negative-EV from the dealer's seat — losing-but-minimising hands.
- **Mean per-hand EV of +1.80** is the value of "playing first" with this book vs. the rollout-completion policy. A self-play `MC(book) vs MC(book)` head-to-head should average ≈ 0.

### Decision criticality (best-action EV gap)

| Gap (best − …) | median | mean | p95 | max |
|---|---:|---:|---:|---:|
| vs 2nd-best stored candidate | 0.43 | 0.90 | **3.85** | 26.02 |
| vs 5th-best (worst) candidate | 4.28 | 4.74 | 11.39 | 27.63 |

→ **Half of all openings**, the right placement matters by < 0.5 EV. **5 % of openings** lose ≥ 3.85 EV if mis-played (e.g. wrong row for QQ in awkward 5-card layouts).

---

## 2. Where do cards go?

### Slot histogram of the best action (full hand of 5 cards)

| T | M | B | X | Count | Share |
|---:|---:|---:|---:|---:|---:|
| **1** | **2** | **2** | 0 | 76,590 | **50.17 %** |
| 0 | 2 | 3 | 0 | 50,441 | **33.04 %** |
| 3 | 1 | 1 | 0 | 12,615 | 8.26 % |
| 0 | 1 | 4 | 0 | 5,002 | 3.28 % |
| 2 | 3 | 0 | 0 | 2,741 | 1.80 % |
| 0 | 3 | 2 | 0 | 1,724 | 1.13 % |
| (10 more layouts) | | | | < 3,500 | < 2.3 % |

**Reading.** A single dominant template — `T=1 / M=2 / B=2`. The `T=0 / M=2 / B=3` runner-up applies when no card is worth committing to top. `T=3 / M=1 / B=1` is the "premium pair commits to fantasy" template (mainly AA/KK).

### Cards placed on the top row

| # cards on top | Orbits | Share |
|---:|---:|---:|
| 0 | 58,100 | 38.1 % |
| 1 | 78,935 | 51.7 % |
| 2 | 2,992 | **2.0 %** |
| 3 | 12,619 | 8.3 % |

**Reading.** Two-on-top is essentially never chosen (mostly meaningful only when there's no pair to top, in which case stacking two highs on top loses the option to develop a higher pair). Three-on-top is reserved for committed-fantasy plays (almost always with AA/KK + sundries).

### Per-rank top-row usage — P[rank `R` on top │ rank `R` in hand]

| Rank | P(top) | Bar |
|---:|---:|:--|
| 2 | 17.0 % | `########` |
| 3 | 16.2 % | `########` |
| 4 | 15.2 % | `#######` |
| 5 | 14.4 % | `#######` |
| 6 | 16.7 % | `########` |
| 7 | 16.3 % | `########` |
| 8 | 15.9 % | `#######` |
| 9 | 15.1 % | `#######` |
| T | 13.8 % | `######` |
| J | 14.2 % | `#######` |
| Q | 17.1 % | `########` |
| **K** | **22.0 %** | `##########` |
| A | 16.1 % | `########` |

**Why is K most-topped?** Two reasons: (i) K is the most common "filler" on a partial top row in front of AA/KK pairs (e.g. `T=KK K | M=… | B=…`); (ii) K-singleton-on-top is the standard "ahead but not committed" placement when the player wants flexibility for a future Q pair (QQ→F14). A is less-topped because singleton-A on top blocks middle from later becoming AA pair.

---

## 3. Pair strategy

A pair is the single most decision-critical feature of a street-1 hand. The book makes the call cleanly along the QQ–JJ inflection point.

> **Note on the table below.** Counts are *orbits where rank R appears ≥ 2 times in the hand* (so trips are double-counted in the matching pair tier, but at < 1 % per tier this is noise). `TT` = both pair cards on top, `BB` = both on bottom, `MM` = both on middle, `TM`/`TB`/`MB` = split.

| Tier | Orbits | Avg best EV | Median | P[fantasy] | TT (top) | BB (bot) | MM | TM | TB | MB | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **AA** | 6,328 | **−3.34** | −5.04 | 15.4 % | **5,343 (84.4 %)** | 480 (7.6 %) | 113 | 0 | 0 | 0 | 392 |
| **KK** | 6,233 | **−2.79** | −4.38 | 18.6 % | **5,089 (81.6 %)** | 584 (9.4 %) | 164 | 0 | 0 | 6 | 390 |
| **QQ** | 6,138 | **+0.63** | +0.57 | 10.8 % | 1,796 (29.3 %) | **2,811 (45.8 %)** | 904 (14.7 %) | 101 | 0 | 138 | 388 |
| JJ | 6,043 | +3.98 | +3.88 | 7.1 % | 0 | **5,340 (88.4 %)** | 115 | 77 | 0 | 125 | 386 |
| TT | 5,948 | +3.84 | +3.72 | 7.0 % | 0 | **5,241 (88.1 %)** | 128 | 48 | 2 | 145 | 384 |
| 99–77 | 17,274 | +3.57 | +3.46 | 6.9 % | 0 | **15,308 (88.6 %)** | 417 (2.4 %) | 85 | 1 | 323 | 1,140 |
| 66–22 | 26,890 | +3.29 | +3.03 | 5.9 % | 0 | **24,423 (90.8 %)** | 289 | 53 | 2 | 263 | 1,860 |

### Key inflections
1. **AA, KK → top.** ~84 % of AA and ~82 % of KK orbits split the pair onto the top row as `T=PPx | M=… | B=…`, accepting a 16-19 % foul-entry rate to chase F15/F16. The mean EV is **negative** because the foul tail dominates — these pairs are *defensive* (or rather, fantasy-or-bust) in the dealer's seat.
2. **QQ is the only borderline pair**: ~29 % top, ~46 % bottom, ~15 % middle. Decision-gap p95 = **8.32 EV** — QQ in awkward layouts is the **most decision-critical pair in the game**.
3. **JJ and below → bottom**, near-deterministically. Pair-JJ on top is *never* the prefilter-best (0/6,043 orbits in the book). The EV is solidly positive (+3.8 to +4.0) because pair-JJ on bottom is a free pair-of-jacks worth in royalties + low foul risk.

### Pair-on-top fantasy economics

For QQ/KK/AA on top, the recorded `fantasy_entry_rate` reads ~11/19/15 %. The fantasy continuation bonus at horizon `H` (from `FantasyEVTable.horizon_value_relative`) is enormous for F16 (~+45 EV at H=∞), so when running with `tier_horizon_values` set the AA/KK plays become significantly more attractive — that's precisely what the rich book's `lookup_horizon` re-ranking is for. But see §8 — the book never enumerated AA-on-bottom for those orbits, so the comparison is flawed for many AA hands.

---

## 4. Trips strategy

Three-of-a-rank in 5 cards is rare (392 orbits per rank × 13 ranks = 5,096 orbits). The book's pattern is essentially uniform across ranks:

| Trip rank | BBB (all bot) | BBBB (quads) | Trips-on-top | Best EV mean |
|---|---:|---:|---:|---:|
| 2  | 378 | 12 | 1 | +7.29 |
| A  | 378 | 12 | 1 | **+8.12** |
| K  | 378 | 12 | 1 | +8.10 |
| (other ranks) | 378 | 12 | 1 | +7.1 — +8.1 |

**Insight.** Even trip-aces are essentially never topped (1/392 orbits). Reason: AAA on top → F17 entry, but bottom = single A leaves bottom developing without help; with AA on bottom and one A on top (the `TBB` 1-orbit pattern), you keep flexibility. The dominant `BBB` placement just dumps the natural trips into the bottom row immediately for a guaranteed strong baseline.

---

## 5. Jokers

Jokers behave as wildcards in scoring and tier-entry checks.

| # jokers in hand | Orbits | best `ev_mean` mean | median | P[fantasy] |
|---:|---:|---:|---:|---:|
| 0 | 134,459 | +1.21 | +1.26 | 5.1 % |
| **1** | 16,432 | **+6.44** | **+6.53** | 6.8 % |
| 2 | 1,755 | +3.54 | +2.12 | **15.1 %** |

**Reading.** A 1-joker hand is worth **+5.2 EV** more on average than a 0-joker hand — the joker can complete an almost-flush, an almost-straight, or pair with any singleton without commitment. **2-joker hands** trade mean EV down for fantasy-entry rate up — the player commits to fantasy more aggressively because the joker pair guarantees the top pair condition.

The top 10 highest-EV orbits in the entire book *all* contain 1–2 jokers + a near-straight-flush bottom (see §7).

---

## 6. Foul-risk hands

The book's best action has `foul_rate > 0.5` in **5,830 orbits (3.8 %)** — the foul tail. p95 of `foul_rate` is **0.79** and the maximum is **0.96**. All foul-heavy orbits are AA/KK-top plays. See §8 for the structural reason this is likely **over-fouled** by the book.

---

## 7. Best and worst openers

### Top 10 highest-EV openings (showing canonical hand → placement)

| Rank | EV | foul % | fent % | Hand (canonical) | Best action |
|---:|---:|---:|---:|---|---|
| 1 | **+31.06** | 5.0 | 2.5 | `Tc Jc Qc *1 *2` | `B=*2 *1 Qc Jc Tc` (joker-straight-flush bottom) |
| 2 | +26.79 | 14.2 | 2.5 | `Tc Jc Kc Ac *1` | `B=*1 Ac Kc Jc Tc` |
| 3 | +23.69 | 2.5 | 7.5 | `Tc Qc Kc *1 *2` | `B=*2 *1 Kc Qc Tc` |
| 4 | +21.95 | 12.5 | 0.0 | `Tc Jc Qc Kc *1` | `B=*1 Kc Qc Jc Tc` |
| 5 | +21.56 | 3.3 | 1.7 | `Tc Kc Ac *1 *2` | `B=*2 *1 Ac Kc Tc` |
| 6 | +21.41 | 15.8 | 0.8 | `Jc Qc Kc Ac *1` | `B=*1 Ac Kc Qc Jc` |
| 7 | **+20.68** | 18.3 | 0.0 | `Tc Jc Qc Kc Ac` | `B=Ac Kc Qc Jc Tc` (natural royal!) |
| 8 | +19.12 | 5.0 | 5.8 | `Jc Kc Ac *1 *2` | `B=*2 *1 Ac Kc Jc` |
| 9 | +18.18 | 0.0 | 5.0 | `2c 3c 4c *1 *2` | `B=*2 *1 4c 3c 2c` (straight-flush low) |
| 10 | +18.01 | 17.5 | 1.7 | `Tc Qc Kc Ac *1` | `B=*1 Ac Kc Qc Tc` |

**Pattern.** Premium openers are almost-made or made **straight-flush bottoms**. The action is "everything on bottom" — no top placement, no middle placement. The high foul rate for hands like #7 (royal flush bottom!) reflects the *terminal score variance* of fully-loaded straight-flush bottoms (high royalty, but blocked-board fouls if middle/top can't develop).

### Top 10 worst-EV openings

| Rank | EV | foul % | fent % | Hand | Best action (book) | Comment |
|---:|---:|---:|---:|---|---|---|
| 1 | **−25.13** | **89.2** | 10.8 | `5c Qd Kc Ah As` | `T=As Ah Kc \| M=5c \| B=Qd` | AA on top, 89 % foul — see §8 |
| 2 | −19.52 | 22.5 | 0.0 | `2c 3d 4h 6s Ac` | `T=6s \| M=4h 3d \| B=Ac 2c` | Disconnected high card, no pair |
| 3 | −18.89 | 25.8 | 0.0 | `2c 3c 6d 8h 9s` | `T=9s \| M=8h 6d \| B=3c 2c` | All-disconnected low-mid |
| 4 | −18.17 | 15.8 | 0.8 | `3c 4d 5c Td Kh` | `T=Td \| M=Kh 4d \| B=5c 3c` | Awkward mid + 5-pair fragments |
| 5 | −17.08 | 83.3 | 16.7 | `2c Td Kd Kh As` | `T=Kh Kd Td \| M=As \| B=2c` | KK top + ace on side — foul-locked |
| 6 | −16.43 | 89.2 | 10.8 | `3c 5d Kd Ac Ah` | `T=Ah Ac 5d \| M=3c \| B=Kd` | AA top + high-card outliers — see §8 |
| 7 | −15.90 | 43.3 | 1.7 | `3c 4c 8d Qc Ad` | `T=Ad \| M=Qc 8d \| B=4c 3c` | Single high card, no real shape |
| 8 | −15.62 | 7.5 | 0.8 | `5c 8c 8d Qc Kh` | `M=Kh Qc \| B=8d 8c 5c` | 8-pair OK on bot, but KQ56-high mid leaks |
| 9 | −15.52 | 90.8 | 9.2 | `6c Jd Qh Kc Kh` | `T=Kh Kc Qh \| M=6c \| B=Jd` | KK top + Q kicker — foul-locked |
| 10 | −15.31 | 6.7 | 0.8 | `7c 9d Th Kd Ad` | `T=7c \| M=Ad Th \| B=Kd 9d` | High broken cards, no good destination |

**Two classes of worst openers**:
- Class A: **High-disconnected no-pair** hands (entries 2, 3, 4, 7, 10) — true bad openings, ~20 % foul, near-zero royalty. These are genuinely tough.
- Class B: **AA/KK with high kickers** (entries 1, 5, 6, 9) — these are the **prefilter-bug victims** with 80-90 % foul rates. See next section.

### Top 5 highest fantasy-entry openers

| fent % | EV | foul % | Hand | Action |
|---:|---:|---:|---|---|
| 59.2 | −0.88 | 40.8 | `Qc Qd Kc Kh As` | `T=QQ \| M=KK \| B=A` (QQ-top, KK-mid → F14) |
| 55.0 | +1.49 | 45.0 | `8c Qd Qh Ks Ad` | `T=QQ 8 \| M=K \| B=A` |
| 55.0 | −0.73 | 45.0 | `Tc Qd Qh Kc As` | `T=QQ T \| M=K \| B=A` |
| 55.0 | −3.37 | 45.0 | `Qc Qd Kc Kd Ac` | `T=QQ \| M=KK \| B=A` |
| 55.0 | −1.22 | 45.0 | `Qc Qd Kc Kh Ah` | `T=QQ \| M=KK \| B=A` |

**Reading.** These are the QQ+KK / QQ-with-room hands. P(fantasy entry next hand) > 50 % is **massive** for a per-hand bonus, and even with -0.88 ev_per_hand the horizon-aware value (`combined_ev = ev_mean + 0.59 × ~+30`) is solidly net-positive once the F14/F15 continuation is folded in.

---

## 8. Known limitation — heuristic prefilter bias toward pair-on-top

### Symptom

Of the 6,328 AA orbits, **5,343 (84 %)** record AA-on-top as their best action. The **all 10 worst-EV orbits in the entire book** are AA-or-KK-on-top variants with 80-90 % foul rates.

### Probe

I ran an **unrestricted MC search** (`top_k=None`, all 232 legal placements scored under shared CRN, 60-120 rollouts each) against three contrasting AA hands and compared to the book's best:

| Hand | Book best (top-5 prefiltered) | Unrestricted MC best | EV gap |
|---|---|---|---:|
| `5c Qd Kc Ah As` (AA + high) | `T=AAK \| M=5 \| B=Q` → **EV=-25.1** (89 % foul) | `T=5 \| M=Q \| B=AAK` → **EV=+4.00** | **+29 EV** |
| `2c 3d 7c Ah As` (AA + low) | `T=AA2 \| M=3 \| B=7` → **EV=-10.6** (92 % foul) | `T=3 \| M=7 \| B=AA2` → **EV=+4.72** | **+15 EV** |
| `5c 6d 9h Ac Ad` (AA + mid) | `T=AA6 \| M=5 \| B=9` → **EV=-1.9** (86 % foul) | `T=6 \| M=95 \| B=AA` → **EV=+4.97** | **+7 EV** |

In **every** AA hand probed, the unrestricted MC finds **AA-on-bottom** to be strictly better — and never even surveyed by the book. The prefilter's `score_action` heuristic over-weights premium-pair-on-top placements (likely because the fantasy royalty bonus dominates the heuristic score), so the top-5 candidates per AA orbit are *all* AA-on-top variants and the genuine AA-on-bottom alternative is never scored by MC.

### Magnitude of the EV leakage

A rough estimate, assuming the +7 to +29 EV-per-hand gap holds across the 5,343 AA-on-top orbits and the ~5,089 KK-on-top orbits + ~1,796 QQ-on-top orbits in the book:

- AA orbits affected: 5,343 × E[gap ≈ +10 EV] = +53,430 chip-units across the orbit family.
- KK orbits affected: 5,089 × E[gap ≈ +6 EV] ≈ +30,500 chip-units.
- QQ orbits affected: smaller (QQ-top is correct in some cases) — needs further probing.

At a fleet/aggregate level: **most AA/KK opening hands are 5-15 EV-per-hand worse than they should be**. Once you factor in horizon (fantasy continuation) this *partially* offsets — AA top entering F16 yields ~+45 EV continuation × ~9 % entry rate ≈ +4 EV bonus, still nowhere near closing the gap.

### Why this slipped past the prior AA-on-top fix

Last session's fix to `HeuristicPolicy` corrected the *scoring* of AA-top so it does not auto-recommend it under user-facing analysis. But the **build-time prefilter inside `MonteCarloPolicy._top_k`** uses the same `score_action` and was never re-evaluated; it still prefers AA-top variants and prunes AA-bottom from MC consideration. The user-facing `Analyzer` then sees an opening-book pre-loaded with the wrong best action and surfaces it.

### Proposed fix

Three options, in order of effort:

1. **Bump prefilter `top_k`** from 5 → 20 or 30 in `scripts/build_full_opening_book.py`. At 20, the prefilter has high probability of including AA-bottom variants. Rebuild cost: **~32 h** at 20 workers (linear in `top_k`).
2. **Stratify the prefilter**: always include at least one candidate per (`#cards-on-top`) bucket. Probably the right fix. ~free at build time, but requires book builder change.
3. **Rebuild with a *better* heuristic** that doesn't over-reward pair-on-top with risky kickers (e.g. discount the top-pair royalty by `(1 − P[middle ≥ pair-tier])`). Requires heuristic redesign.

**Recommended.** Option 2 (stratified prefilter), since the cost is negligible and it cleanly addresses the missing-alternative problem. Validate post-rebuild by re-running the §8 probe — AA-on-bottom should be among the candidates of every AA-hand orbit, and `ev_mean` for AA hands should shift up by ~10 EV on average.

---

## 9. Strategy summary (cheat-sheet form)

For a human or LLM consuming this book:

```
Hand pattern                                    Best opening (book-aligned)
==================================================================
AAA / KKK / trips of any rank                   → all 3 to bottom (BBB)
AA + 3 random cards                             → AA-on-top (CAVEAT: see §8;
                                                  better play is often AA-bot)
KK + 3 random cards                             → KK-on-top (same caveat)
QQ + 3 cards                                    → depends on side cards:
                                                  - cluttered/risky: QQ-bot
                                                  - clean: QQ-top
                                                  - middling: QQ-mid
JJ / TT + 3 cards                               → JJ/TT to bottom (BB)
77-99 + 3 cards                                 → pair to bottom
22-66 + 3 cards                                 → pair to bottom (lowest royalty
                                                  but safest)
No pair, 5 high cards (AKQJT-ish)               → bottom-loading template
                                                  T=1 high or empty, M=2, B=2
No pair, mixed/disconnected                     → bottom-loading template,
                                                  highest cards to bottom
1 joker (no pair)                               → joker fills the most-promising
                                                  flush/straight on bottom
2 jokers                                        → both to bottom + complete pair
                                                  on top with highest single
                                                  rank (commits to fantasy)
3+ suited to bottom (with joker)                → all suited to bottom for
                                                  flush draw
T-J-Q-K + joker/wild                            → straight on bottom (T+ EV)
```

**Top-row entry rules**:
- Pair AA, KK → top (book default; verify against §8 caveat).
- Pair QQ → top only when middle and bottom have clear paths to ≥ QQ pair.
- Pair JJ and below → never top.
- Trips → never top (1/392 orbits per rank).

---

## 10. Reproducibility

This document is rendered from the JSON output of [scripts/analyze_opening_book.py](../scripts/analyze_opening_book.py) — re-run it with any future book version to refresh the numbers.

```bash
# Regenerate the analysis JSON
python -m scripts.analyze_opening_book \
    --book artifacts/opening_book_canonical_v2/opening_book_canonical.pkl \
    --out  artifacts/opening_book_analysis.json

# Targeted unrestricted-MC probe (§8 verification)
python -m scripts.probe_book_alternatives \
    --hands '5c Qd Kc Ah As' '2c 3d 7c Ah As' '5c 6d 9h Ac Ad'   # script not yet written
```

The book schema is described in [tables/canonical_opening.py](../tables/canonical_opening.py) (`CandidateRecord` and `CanonicalOpeningBookTable.SCHEMA_VERSION = 2`).
