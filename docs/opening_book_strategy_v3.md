# OFC Opening Strategy — derived from the v3 canonical opening book

> **Status.** This document supersedes [docs/opening_book_strategy.md](opening_book_strategy.md) (the v2 doc). The v2 book was built on top of a heuristic prefilter bug that systematically favoured *pair‑on‑top* placements for high pairs (AA/KK/QQ), inflating both the foul rate and the apparent value of fantasy‑entry gambits. The v3 book was rebuilt after the prefilter was fixed (commit `364feb3`). Whenever this doc says "v2 used to say … but v3 says …", the new advice is the correct one — the old number was the bug talking. The v2 file is kept on disk only for forensic comparison.

---

## Source artefacts

| Artefact | Location | Notes |
|---|---|---|
| Canonical opening book | [artifacts/opening_book_canonical_v3/opening_book_canonical.pkl](../artifacts/opening_book_canonical_v3/opening_book_canonical.pkl) | **152,646** orbits under `S_4 × S_2` symmetry, top‑5 candidates each, **120** CRN rollouts per candidate (rich schema). 83 MB. |
| Analyzer output | [artifacts/opening_book_analysis_v3.json](../artifacts/opening_book_analysis_v3.json) | Machine‑readable summary used to generate this doc. |
| Self‑play tables | [artifacts/run_250k_v3/](../artifacts/run_250k_v3) | 250 000 MC×MC games, `fantasy_rate=0.20`, seed 0. Contributes the fantasy economics, foul rates, and royalty distributions cited below. |

Reproduce the analysis:

```bash
python -m scripts.analyze_opening_book \
    --book artifacts/opening_book_canonical_v3/opening_book_canonical.pkl \
    --out  artifacts/opening_book_analysis_v3.json
```

---

## TL;DR — five rules of thumb

1. **Strength goes down, structure goes up.** High cards (Aces especially) and pairs go to the **bottom**. The middle row is for *draws* (suited connectors, flush draws, second pairs). The top row is the **last** thing you commit to — and only with a single low‑value card unless you have a premium pair plus a safety net.
2. **Empty top is fine.** **44 %** of all openings put **zero** cards on top; **55 %** put exactly one. Putting **2+** cards on top happens in **~1 %** of orbits and is almost always a deliberate fantasy gambit with QQ+KK or AA+KK two‑pair on a flat board.
3. **Pairs almost always live on the bottom.** From `22` through `AA`, the optimal placement classifies the pair as `B‑B` in **83 – 91 %** of orbits. The v2 doctrine of "AA goes on top for fantasy" was an artefact of a buggy prefilter; v3 puts AA on bottom 83 % of the time and AA on top **~0 %** of the time.
4. **Jokers always sink to the bottom.** A 1‑joker hand averages **+5.7 EV** (vs +0.3 EV for 0‑joker); a 2‑joker hand averages **+10.7 EV**. The book uses the joker to build the bottom row (flush / straight‑flush / quads / full house), not to lock a top pair.
5. **The book is conservative on fantasy.** The unconditional probability that the optimal opening enters fantasy in one street is only **4.1 %**. Fantasy is reached *over many streets* via a strong board, not by reckless top‑row gambits. The only opening hands with `fantasy_entry_rate > 0.30` are `QQ + KK` or `KK + AA` two‑pair, and even those average **negative** EV because of the foul risk.

---

## TL;DR — what an LLM/agent should do at street 1

1. **Look up the canonical orbit** of the dealt hand. Choose the row that maximises horizon‑aware EV.

   ```python
   from tables.canonical_opening import CanonicalOpeningBookTable
   from tables.fantasy_ev import FantasyEVTable

   book   = pickle.load(open(".../opening_book_canonical_v3/opening_book_canonical.pkl", "rb"))
   fev    = pickle.load(open(".../run_250k_v3/fantasy_ev.pkl", "rb"))

   # remaining_streets after street 1 is 4 (street 2..5), so use horizon 4.
   horizon_vals = fev.horizon_value_relative(4)            # {0:0, 14:21.95, 15:29.85, 16:33.29, 17:45.38}
   placements   = book.lookup_horizon(hand, horizon_vals)  # tuple[(card, slot), …]
   ```

2. **Never enter fantasy on street 1 with a marginal hand.** If the chosen candidate has `foul_rate > 0.30` *and* `fantasy_entry_rate > 0.30`, pick the safer #2 candidate instead — see §8.
3. **Fall back to live MC search** when (a) the orbit is missing (unlikely; the book is complete over 5‑card orbits), or (b) the best two candidates differ by `< 0.5 EV` *and* the foul gap differs by `> 0.05` — the prefilter occasionally orders close calls by EV alone, so the more conservative choice may be better.
4. **Don't use the book past street 1.** It only stores 5‑card orbits. Streets 2–5 still need MC search or the per‑signature `policy_prior`/`foul_prob` tables in [run_250k_v3](../artifacts/run_250k_v3).

---

## 1. Global statistics

`n_orbits = 152,646`, `120` CRN rollouts per stored candidate.

| Per‑orbit best action | min | p05 | p25 | median | mean | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ev_mean` (chips/game) | −19.77 | −6.00 | −0.93 | **+1.10** | **+1.02** | +3.07 | +7.89 | +27.60 |
| `foul_rate` | 0 | 0.025 | 0.05 | 0.10 | 0.12 | 0.17 | 0.27 | 0.76 |
| `fantasy_entry_rate` | 0 | 0 | 0.008 | 0.025 | 0.041 | 0.058 | 0.117 | 0.692 |

- **93,811 / 152,646 (61.5 %)** orbits have positive best‑action EV; **38.5 %** are negative.
- Mean dealer EV is **+1.02 chips/game** with the book against the rollout policy. This is **lower** than v2's +1.80 because the v3 book stops "buying" EV with fouled fantasy gambits.

### Decision criticality

| Gap (best minus …) | median | mean | p95 | max |
|---|---:|---:|---:|---:|
| 2nd‑best stored candidate | 0.43 | 0.76 | **2.93** | 19.79 |
| worst stored candidate    | 3.97 | 4.19 | 7.87 | 28.99 |

→ For half of all openings, the right placement matters by `< 0.5` EV. About **5 %** of openings lose `≥ 3` EV if mis‑played, almost always involving a pair, a flush draw, or two pair.

---

## 2. Where do cards go? — slot histogram

The full‑hand placement template — `(T, M, B, X)` = (# top, # middle, # bottom, # discard) — for the best action per orbit:

| T | M | B | X | Count | Share |
|---:|---:|---:|---:|---:|---:|
| **1** | **2** | **2** | 0 | 80,908 | **53.00 %** |
| 0 | 2 | 3 | 0 | 59,706 | **39.11 %** |
| 0 | 1 | 4 | 0 | 4,911 | 3.22 % |
| 1 | 1 | 3 | 0 | 1,729 | 1.13 % |
| 1 | 3 | 1 | 0 | 1,624 | 1.06 % |
| 0 | 3 | 2 | 0 | 1,517 | 0.99 % |
| 2 | 3 | 0 | 0 | 1,354 | 0.89 % |
| 2 | 2 | 1 | 0 | 231 | 0.15 % |
| 0 | 0 | 5 | 0 | 182 | 0.12 % |
| (5 more layouts) | | | | | < 0.4 % |

**Reading.** Two templates cover **92 %** of all openings:
- `T=1 M=2 B=2` — drop one card on top (kicker), put your two best non‑bottom cards on middle, and start the bottom row with your two strongest holdings.
- `T=0 M=2 B=3` — leave the top empty when nothing is sacrificial enough; commit three to the bottom.

The v2 archetype `T=3 / M=1 / B=1` (8.3 % of v2 orbits) is **gone** in v3 — it was a side‑effect of the prefilter bias toward stacking high cards on top.

### Cards placed on the top row

| # cards on top | Orbits | Share |
|---:|---:|---:|
| 0 | 66,448 | 43.5 % |
| 1 | 84,552 | **55.4 %** |
| 2 | 1,640 | 1.1 % |
| 3 | 6 | 0.004 % |

3 cards on top happens only six times in 150 k orbits (all are fantasy gambits with a premium pair and a clear bottom plan).

### Per‑rank top‑row usage — P[rank R on top │ rank R in hand]

| Rank | P(top) | Bar |
|---:|---:|:--|
| 2 | 16.3 % | `########` |
| 3 | 15.0 % | `#######` |
| 4 | 13.9 % | `#######` |
| 5 | 12.8 % | `######` |
| 6 | 15.1 % | `#######` |
| 7 | 14.5 % | `#######` |
| 8 | 13.9 % | `#######` |
| 9 | 13.3 % | `######` |
| T | 12.3 % | `######` |
| J | 13.0 % | `######` |
| Q | 13.0 % | `######` |
| K | 12.8 % | `######` |
| **A** | **5.6 %** | `###` |

**Reading.**
- Top‑row usage is *nearly flat* from 2–K (the chosen "kicker" rank is roughly uniform: whichever single card is least useful elsewhere goes up top).
- **Aces are dramatic outliers** — 5.6 % is less than half the rate of *any* other rank. The book actively avoids committing an ace to the top because (a) it can anchor a pair / two‑pair on the bottom, and (b) it has the highest expected upside if held back.
- v2 reported `P(A on top) = 64 %` due to the prefilter bug. The 11× collapse in P(A on top) is the single largest behavioural difference between v2 and v3.

---

## 3. Pair strategy

Out of 152,646 orbits, **76,854 (50.4 %)** contain at least one natural pair. The placement of the pair is highly stereotyped per tier:

| Tier | Orbits | `B‑B` | `M‑M` | `M‑B` (split) | `T‑M` | `T‑B` | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| **AA** | 6,328 | **83.0 %** | 2.2 % | 6.1 % | 2.4 % | 0.1 % | 6.2 % |
| **KK** | 6,233 | **86.8 %** | 3.2 % | 2.7 % | 0.9 % | <0.1 % | 6.3 % |
| **QQ** | 6,138 | **87.3 %** | 3.7 % | 2.4 % | 0.3 % | 0 | 6.3 % |
| **JJ** | 6,043 | **87.9 %** | 3.6 % | 2.0 % | 0.2 % | 0 | 6.4 % |
| **TT** | 5,948 | **87.8 %** | 3.5 % | 2.2 % | 0.1 % | <0.1 % | 6.5 % |
| **77 – 99** | 17,274 | **88.7 %** | 2.7 % | 1.8 % | 0.1 % | <0.1 % | 6.6 % |
| **22 – 66** | 26,890 | **90.5 %** | 1.5 % | 0.9 % | 0.1 % | <0.1 % | 6.9 % |

(`other` = paired with a joker, paired across (TT, MM), or absorbed into a fantasy two‑pair lock on the top row — handled in §6.)

| Tier | best EV (mean) | fantasy entry | 2nd‑best gap (mean) |
|---|---:|---:|---:|
| AA | **+2.68** | 5.8 % | +1.38 |
| KK | +2.49 | 5.4 % | +1.24 |
| QQ | +2.43 | 5.1 % | +1.19 |
| JJ | +2.30 | 7.2 % | +1.11 |
| TT | +2.18 | 7.0 % | +1.04 |
| 77‑99 | +2.06 | 6.9 % | +0.91 |
| 22‑66 | +1.64 | 5.8 % | +0.88 |

**Reading.**

- **The pair *always* belongs on the bottom.** No tier — not even AA — places the pair on top in more than 2.4 % of orbits.
- **EV is linear in pair rank** at ~+0.15 EV per rank step (66→AA). Premium pairs are not categorically different from mid pairs; they only earn more royalties.
- **AA has the lowest fantasy entry rate of any pair tier (5.8 %).** That seems counter‑intuitive until you remember that the book *doesn't try* to commit AA to the top (which would be the only way to enter fantasy on street 1). It plays for the long game — AA on bottom is a free pair plus draws elsewhere.

### Concrete recipes

| Hand pattern | Optimal placement | EV (mean) | Comment |
|---|---|---:|---|
| `AA + 3 low unsuited` (e.g. `Ac Ad 2c 3h 5s`) | M = high non‑pair · B = `AA` + best second card | ≈ −2 | One ace stays on bottom; the other goes to middle to pair‑split for a possible 2‑pair on M. The very worst openers in the book are `AA + low garbage` hands. |
| `AA + 3 suited connectors` (`Ac Ad 3c 4c 5c`) | M = `5‑4‑3 (suited)` · B = `A A` | **+2.13** | M is now a flush draw / straight draw; AA holds bottom. |
| `AA + Q‑J‑T` runs (`Ac Ad Qc Jc Tc`) | T = `T` · M = `Q J` · B = `A A` | +3.50 | A rare 3‑card‑on‑top template; safe because the bottom AA + suited Q/J/T middle have very low foul risk. |
| `KK + AQ unsuited` | M = `A‑Q` · B = `K K + low` | +2–3 | The ace becomes the middle's high card, not a top stay. |
| `77‑99 + 2 low + 1 high` | T = low non‑pair · M = high · B = `pair + low` | +2 | Standard "middle pair on bottom" line. |
| **Worst openers** (`2c 3d 4h 6s Ac`, `3c 4d 5c Td Kh`, etc.) | T = single mid‑rank · M = two odd cards · B = two odd cards | −19 to −13 | "Rainbow garbage" — no pair, no suit, no straight. Just minimise foul risk. |

---

## 4. Trips strategy

Trips are extremely rare in 5‑card openers (each rank has only `n=392` orbits, **0.26 %** of the book). The placement is deterministic.

| Tier | mean EV | `B‑B‑B` (trips on bot) | other |
|---|---:|---:|---|
| trips_2 | +5.87 | 96.4 % | 12 of `BBBB` (joker‑boosted quads on B), rare T/M cameos |
| trips_5 | +6.28 | 96.4 % | (same) |
| trips_T | +6.58 | 96.4 % | |
| trips_K | +6.89 | 96.4 % | |
| trips_A | **+6.97** | 96.4 % | |

**Reading.** *Trips always go on the bottom* (regardless of rank); the only ~3.5 % of orbits where they don't are joker‑augmented quads where the 4th card of the rank is already the joker. Topping trips for a fantasy‑17 lock is essentially never chosen — the bottom royalty (full house upside) plus middle development is worth more than +22 fantasy royalty plus catastrophic foul risk.

> **Practical agent rule.** If you see trips in your opening hand, stop searching: put them on the bottom, then place the remaining two cards by single‑card MC.

---

## 5. Suited connectors, flushes, and straights

The book's most lucrative non‑joker openers are *5‑card monochrome runs* that go directly on the bottom:

- `Tc Jc Qc Kc Ac` (royal flush dealt) → `B = AKQJT` — EV **+20.7** (foul 18.3 % from later‑street collapse).
- `2c 3c 4c 5c 6c` (small SF) → `B = 6c‑2c` — EV **+7.18** (foul 22.5 %).
- `9c Tc Jc Qc Kc` (high SF) → `B = K‑9c` — EV similarly high.

For 4‑suited‑plus‑1‑offsuit hands, the book typically:
1. Puts the four suited cards on the **bottom** as a flush draw, **and**
2. Puts the offsuit card on **top** if it is the lowest in the hand (becomes the kicker for free).

For 3‑suited‑plus‑two‑offsuit, the book splits — middle holds a 2‑card flush draw plus offsuit; bottom takes the third suited card plus the higher offsuit (sets up a *bottom flush draw* with one more catch).

---

## 6. Two‑pair openers — the only deliberate fantasy plays

Two‑pair (`pair_X + pair_Y`) appears in `6,864` orbits (4.5 % of the book) with **mean EV +0.11 — the lowest of any structural class**, primarily because two‑pair openers are sometimes pushed into fantasy gambits with high foul risk.

The book gambles on fantasy entry (`top = QQ` or `top = KK`) **only** when both of these are true:

| Condition | Why |
|---|---|
| The second pair is **KK or AA** (locks the middle row in front of the top pair, guaranteeing legality) | A `QQ‑top, JJ‑mid` line still fouls when middle catches a third J or two‑pair higher than Q later. |
| The bottom card is irrelevant (low offsuit) **or** the middle can be padded freely | Otherwise the kicker constraints make street 2–5 unwinnable. |

**The highest fantasy‑entry openers in the entire book**:

| Hand | Top | Mid | Bot | EV | Foul | F_ent |
|---|---|---|---|---:|---:|---:|
| `Qc Qd Kc Kd Ac` | `Qc Qd` | `Kc Kd` | `Ac` | **−1.54** | **30.8 %** | **69.2 %** |
| `Qc Qd Kc Kh Ad` | `Qc Qd` | `Kc Kh` | `Ad` | +1.70 | 32.5 % | 67.5 % |
| `Qc Qd Kc Kh Ah` | `Qc Qd` | `Kc Kh` | `Ah` | +0.18 | 34.2 % | 65.8 % |
| `8c Qc Qd Ac Ah` | `Qc Qd` | `Ah Ac` | `8c` | +2.82 | 40.8 % | 59.2 % |

→ **Even the best fantasy gambit averages only +2.82 EV with 41 % foul.** Most are negative‑EV. The book does the deal *because the alternative (QQ‑on‑bottom, KK‑on‑middle as two‑pair, ace kicker) is even worse* in this very specific structure — without QQ‑on‑top the second pair is dominated and ends up paying royalties to opponents on the middle.

> **Agent rule for two‑pair.** Default to the safe bottom‑heavy line. Trigger fantasy gambit only when both pairs are ≥ QQ. Otherwise commit the higher pair to the bottom.

---

## 7. Joker strategy

| #jokers | n_orbits | mean EV | mean F_entry |
|---:|---:|---:|---:|
| 0 | 134,459 | +0.32 | 3.8 % |
| **1** | 16,432 | **+5.74** | 6.4 % |
| **2** | 1,755 | **+10.66** | 7.4 % |

**Reading.**
- A single joker is worth on average **+5.42 chips** of EV (`+5.74 − +0.32`).
- A second joker is worth another **+4.92 chips**.
- Fantasy entry rate barely changes (+2.5 percentage points for 1J, +3.6 pp for 2J) — the joker is not used to "lock" a top‑row pair. **It is used to fuse the bottom row into a made hand**.

### Joker placement recipes

| Hand pattern | Best line | EV |
|---|---|---:|
| `1J + 4 suited connectors` (`Tc Jc Qc Kc *1`) | All five on bottom → royal flush | **+27.60** |
| `1J + 4 suited high cards` (`Jc Qc Kc Ac *1`) | All five on bottom (royal) | +21.43 |
| `2J + 3 suited connectors` (`Tc Jc Qc *1 *2`) | All five on bottom (SF/quads) | +17.23 |
| `2J + 2 low + 1 high` (`2c 3c Ad *1 *2`) | M = `Ad`; B = `*2 *1 3c 2c` → quads on B | +10.08 |
| `1J + pair_AA` (`2c 3c Ad Ah *1`) | M = `3c`; B = `*1 Ah Ad 2c` → AAA(+ low) + better catches | +9.65 |
| `2J + high_card` (`Jc Qc Kc *1 *2`) | B = all five → 5‑high straight flush | +17.23 |

The book never uses the joker to anchor a top‑row pair. Top‑row jokers occur only in **6 of 18,187 joker hands (0.03 %)**, all of which involve a confirmed 2‑pair or trips on lower rows that lock the legality.

> **Agent rule for jokers.** Reject any candidate that puts a joker above the middle row unless the rest of the board has already locked legality (rare in opener). Default: the joker joins the **largest** of the three rows below it (usually B).

---

## 8. Foul avoidance & risk budget

### From the canonical book (street 1, optimal action)

| Hand class | mean foul rate | comment |
|---|---:|---|
| **high_card** (no pair, no joker) | **15.8 %** | Rainbow garbage — fouls because mid/bot fill with random catches. |
| two_pair | 16.9 % | Driven by fantasy gambits (see §6). |
| pair_22 → pair_66 | 10–14 % | Low pairs on bottom; if middle catches a pair the top kicker may collapse. |
| pair_77 → pair_TT | 7–9 % | |
| pair_JJ → pair_AA | **6–7 %** | Most robust opener class. |
| high_card+1j | 7.3 % | Joker patches the bottom row; foul risk concentrated in catches that strand mid. |
| high_card+2j | **5.6 %** | Two jokers make the bottom flexible enough that foul risk is minimal. |
| trips_* | 3–5 % | Trips on bottom essentially never foul. |

### From 250 k self‑play (`run_250k_v3/foul_by_tier.pkl`)

| Phase | n_games | fouls | rate | notes |
|---|---:|---:|---:|---|
| NORMAL (no fantasy active) | 399,999 | 17,493 | **4.37 %** | The true per‑game foul rate of the MC×MC reference policy. |
| F14 | 54,181 | 37 | 0.07 % | ⚠ Inflated by the engine joker bug (commit `0613287`); see caveats. |
| F15 | 26,230 | 43 | 0.16 % | ⚠ Same. |
| F16 | 10,866 | 31 | 0.29 % | ⚠ Same. |
| F17 | 8,724 | 24 | 0.28 % | ⚠ Same. |

> **Caveat — joker engine fix landed after `run_250k_v3`.** The fantasy foul rates above are slight overestimates: the engine was using a too‑strict joker substitution rule in fantasy deal evaluation when the data was generated. The next data run will retire these artefacts; until then, treat any fantasy foul figure < 0.5 % as "essentially zero" rather than a learned signal. **NORMAL (4.37 %) is unaffected** because no jokers participate in plain placement legality the same way.

### Foul‑risk budget for the opener

A practical heuristic: **don't take an opener whose `foul_rate × 4` exceeds the EV gain over the alternative**.

- A 5 % foul opener loses ≈ `0.05 × 6 ≈ 0.3` EV to fouls (mean board score ≈ 6 chips for the winner).
- A 30 % foul opener (fantasy gambit) loses ≈ 1.8 EV before the upside is counted. If the alternative is ‑2 EV anyway, the gambit is correct; if the alternative is +1 EV, it isn't.

---

## 9. Fantasy economics — when (and whether) to gamble

Estimated from `run_250k_v3/fantasy_ev.pkl` (250 k MC×MC games):

| Tier | Definition | n_games | avg raw score | retention P(stay in fantasy) | steady‑state `V(tier)` |
|---|---|---:|---:|---:|---:|
| F14 | top = `QQ` (14‑card fantasy deal) | 54,181 | **+6.42** | 75.2 % | **+29.5** |
| F15 | top = `KK` (15‑card) | 26,230 | +9.10 | 76.4 % | **+42.4** |
| F16 | top = `AA` (16‑card) | 10,866 | +11.91 | 68.4 % | **+40.6** |
| F17 | top = trips on top (17‑card) | 8,724 | +14.48 | 77.1 % | **+67.2** |
| NORMAL | no fantasy | 399,999 | −2.11 | 96.9 % stay normal | 0 (reference) |

**Reading.**

- **First entry from NORMAL: only 3.14 %** of games transition NORMAL → any fantasy tier per game (`(4246 + 4442 + 3502 + 363) / 399,999`). Fantasy is reached by *long boards*, not by an opening gambit.
- **Once in, retention is ~75 % per game.** A fantasy player keeps the privilege for ≈ 4 hands on average (`1 / (1 − 0.75)`).
- **F15 is the best risk‑adjusted tier**: steady‑state value `+42.4` for only **76 %** retention. F17 has a higher absolute V (+67) but is harder to qualify and harder to keep without fouling.
- **F16 (AA‑on‑top fantasy) is surprisingly *worse* than F15** in steady state (40.6 vs 42.4) — the larger 16‑card deal forces awkward 5‑on‑middle decisions and retention drops by ~8 points. **Don't preferentially gun for the 16‑card variant over the 15‑card one.**
- **Continue‑bonus** (the marginal value of staying vs returning to normal) — used by `lookup_horizon` — equals `V(tier)` for `t > 0` because `V(NORMAL) = 0` by construction.

### Horizon‑adjusted picking

For the typical 4‑street horizon left after street 1, `FantasyEVTable.horizon_value_relative(4)` returns

```
{0: 0.0, F14: +21.95, F15: +29.85, F16: +33.29, F17: +45.38}
```

so `book.lookup_horizon(hand, horizon_4)` will value a `5 %` higher fantasy‑entry candidate by an additional **`0.05 × 29.85 ≈ +1.49`** chips. That is enough to flip 2nd‑best into 1st‑best in roughly 6 % of orbits (estimated from `gap2 < 1.5` count).

---

## 10. Best and worst opener catalogue

### Top 10 highest‑EV openers (joker‑heavy, suited royals)

| Rank | Hand | Type | Optimal | EV | Foul | F_ent |
|---:|---|---|---|---:|---:|---:|
| 1 | `Tc Jc Kc Ac *1` | high_card +1j | B = `*1 A K J T` (royal flush) | **+27.60** | 10.8 % | 0.8 % |
| 2 | `Tc Jc Qc *1 *2` | high_card +2j | B = `*2 *1 Q J T` | +24.31 | 5.0 % | 2.5 % |
| 3 | `Tc Qc Kc *1 *2` | high_card +2j | B = `*2 *1 K Q T` | +22.95 | 2.5 % | 8.3 % |
| 4 | `Jc Qc Kc Ac *1` | high_card +1j | B = `*1 A K Q J` | +21.43 | 13.3 % | 0.8 % |
| 5 | `Tc Kc Ac *1 *2` | high_card +2j | B = `*2 *1 A K T` | +21.18 | 3.3 % | 1.7 % |
| 6 | `Tc Jc Qc Kc Ac` | high_card | B = `A K Q J T` (royal, no joker) | +20.68 | 18.3 % | 0 % |
| 7 | `Jc Kc Ac *1 *2` | high_card +2j | B = `*2 *1 A K J` | +19.12 | 5.0 % | 5.8 % |
| 8 | `Tc Qc Kc Ac *1` | high_card +1j | B = `*1 A K Q T` | +17.93 | 16.7 % | 0 % |
| 9 | `2c 3c 4c *1 *2` | high_card +2j | B = `*2 *1 4 3 2` (low SF) | +17.91 | 0 % | 5.0 % |
| 10 | `Qc Kc Ac *1 *2` | high_card +2j | B = `*2 *1 A K Q` | +17.88 | 5.0 % | 3.3 % |

### Worst 10 openers (unavoidable −EV)

| Rank | Hand | Type | Optimal | EV | Foul |
|---:|---|---|---|---:|---:|
| 1 | `2c 3d 4h 6s Ac` | high_card | T = `6s` · M = `4h 3d` · B = `Ac 2c` | **−19.77** | 22.5 % |
| 2 | `3c 4d 5c Td Kh` | high_card | T = `Td` · M = `Kh 4d` · B = `5c 3c` | −19.10 | 15.8 % |
| 3 | `2c 3c 6d 8h 9s` | high_card | T = `9s` · M = `8h 6d` · B = `3c 2c` | −18.89 | 25.8 % |
| 4 | `5c Qd Kc Ah As` | pair_A | M = `Kc Qd` · B = `As Ah 5c` | **−18.23** | 1.7 % |
| 5 | `3c 4c 8d Qc Ad` | high_card | T = `Ad` · M = `Qc 8d` · B = `4c 3c` | −15.90 | 43.3 % |
| 6 | `5c 8c 8d Qc Kh` | pair_8 | M = `Kh Qc` · B = `8d 8c 5c` | −15.50 | 7.5 % |
| 7 | `7c 9d Th Kd Ad` | high_card | T = `Ad` · M = `Th 7c` · B = `Kd 9d` | −15.45 | 24.2 % |
| 8 | `2c 3c 6d 8c Jd` | high_card | T = `Jd` · M = `8c 6d` · B = `3c 2c` | −14.43 | 36.7 % |
| 9 | `2c 8c 9d Tc Qd` | high_card | T = `8c` · M = `Tc 2c` · B = `Qd 9d` | −14.37 | 28.3 % |
| 10 | `2c 8c 9c Jd Kh` | high_card | T = `2c` · M = `Kh Jd` · B = `9c 8c` | −14.25 | 17.5 % |

**Reading the worst list.** All ten are *rainbow non‑pair* hands except for `pair_A + Q‑K + low` and `pair_8 + Q‑K + low`. Even the pair_A entry (#4) keeps both aces on the bottom — the EV is bad because the high middle (Kc Qd) is exposed to over‑pairs on later streets. **The book does *not* "save" itself by re‑routing AA to the top**, which would only push the foul rate above 60 %.

---

## 11. v2 → v3 contrast (forensic summary)

| Metric | v2 (buggy) | v3 (fixed) | Δ |
|---|---:|---:|---:|
| Mean best‑action EV | +1.80 | +1.02 | −0.78 |
| Mean best‑action foul | 0.19 | 0.12 | −0.07 |
| `P(A on top │ A in hand)` | **64.4 %** | **5.6 %** | **−58.8 pp** |
| `P(AA pair on top │ AA in hand)` | 84.7 % | **0.0 %** (0 of 6,328) | −84.7 pp |
| Orbits with 3 cards on top | 8.3 % | 0.004 % | −8.3 pp |
| Orbits with 0 cards on top | 38.1 % | 43.5 % | +5.4 pp |
| `T=3 M=1 B=1` template share | 8.3 % | 0.0 % | −8.3 pp |
| Mean best‑action `fantasy_entry_rate` | 0.05 | 0.04 | −0.01 |

**Interpretation.** The buggy v2 prefilter scored AA‑on‑top placements as having high heuristic EV due to mis‑penalised ordering constraints, so the MC layer never even saw the AA‑on‑bottom alternative for those hands. v3 enumerates a sane candidate set, and the empirical winner is *almost always* AA‑on‑bottom. The drop in mean EV (+1.80 → +1.02) is the book *paying back* the EV it had been borrowing by counting wins on never‑actually‑legal fantasies — the v3 number is the honest one.

---

## 12. Concrete "agent recipes"

### Recipe A — Single‑street opener (no horizon)

```python
from pickle import load
from tables.canonical_opening import CanonicalOpeningBookTable

book = load(open("artifacts/opening_book_canonical_v3/opening_book_canonical.pkl", "rb"))
# Returns: tuple[(card, slot), …]  for the best CRN-EV action.
action = book.lookup(hand_5cards)
```

Use this when the agent is not currently in fantasy and there is no further table lookup planned beyond street 1.

### Recipe B — Horizon‑aware opener (recommended)

```python
fev          = load(open("artifacts/run_250k_v3/fantasy_ev.pkl", "rb"))
horizon_vals = fev.horizon_value_relative(4)     # 4 streets left after street 1
action       = book.lookup_horizon(hand_5cards, horizon_vals)
```

Use this when the agent will play multiple streets (the default for any real opponent).

### Recipe C — Safety override

After consulting the book, before committing the action:

```python
best, second = book.candidates(hand_5cards)[:2]
if best.foul_rate > 0.30 and best.fantasy_entry_rate > 0.30 and \
   second.ev_mean > best.ev_mean - 1.5:
    chosen = second   # avoid the fantasy gambit when 2nd is comparable
else:
    chosen = best
```

This catches the rare two‑pair fantasy gambits where the EV margin is small but the foul risk is huge. Empirically activates on `< 0.5 %` of orbits.

### Recipe D — Fall‑through to live MC

If `book.lookup` returns `None` (orbit missing — should not happen for legal 5‑card hands) or `best.n_rollouts < 60` (book was incompletely re‑run), call:

```python
from ai.monte_carlo_policy import MonteCarloPolicy
mcs = MonteCarloPolicy(rollouts=200, ...).choose(state, hand=hand_5cards)
```

### Recipe E — Beyond street 1

For streets 2–5, do not use the opening book. Use the per‑signature tables under `run_250k_v3/`:

- `policy_prior.pkl` (2 M entries) for action priors.
- `foul_prob.pkl` (2 M entries) for legality awareness.
- `fantasy_arrangement.pkl` (100 k entries) for fantasy hand layouts.

---

## 13. Royalty distribution — what good play earns

From `run_250k_v3/royalty_by_row.pkl` (`n_boards = 482,372`, both seats, fouled boards excluded):

| Row | Royalty value | n_boards | Share |
|---|---:|---:|---:|
| **Top** | 1 (`high_card` / 22‑55 pair) | 368,199 | **76.3 %** |
| | 2 (66+ pair) | 110,755 | 22.9 % |
| | 4 (trips) | 3,418 | 0.7 % |
| **Middle** | 1 (`hi`/pair/two_pair) | 376,987 | **78.2 %** |
| | 2 (trips/straight) | 61,100 | 12.7 % |
| | 4 (flush) | 23,560 | 4.9 % |
| | 8 (full house) | 18,778 | 3.9 % |
| | 12 (quads) | 1,684 | 0.3 % |
| | 20 (SF/RF) | 263 | 0.05 % |
| **Bottom** | 0 (≤ trips) | 204,066 | **42.3 %** |
| | 1 (straight) | 18,923 | 3.9 % |
| | 2 (flush) | 36,946 | 7.7 % |
| | 4 (full house) | 109,843 | **22.8 %** |
| | 8 (quads) | 84,692 | **17.6 %** |
| | 12 (SF) | 21,481 | 4.5 % |
| | 25 (royal flush) | 6,421 | **1.3 %** |

**Reading.**

- **The bottom is where 96 % of all royalty chips come from.** Roughly **half** the boards score `≥ 4` on the bottom row (full house or better). Roughly **24 %** score `≥ 8` (quads or better). **1.3 %** of boards are royal flushes — this is inflated by fantasy hands, which see ≥ 14 cards.
- The top almost always pays only 1 chip; 22.9 % of the time it pays 2 (66+ pair).
- The middle is *mostly trash* (78 % score just 1).

So the strategic principle "build the bottom" is not just about avoiding fouls — it is *where the chips are*.

---

## 14. Match‑level signal — what playing this book actually looks like

From `run_250k_v3/match_summary.pkl`:

| Quantity | Value |
|---|---:|
| n_games | 250,000 |
| seat 0 wins / seat 1 wins / ties | 120,804 / 122,475 / 6,721 |
| seat 0 foul rate | 3.54 % |
| seat 1 foul rate | 3.51 % |
| seat 0 scoop rate | 21.9 % |
| seat 1 scoop rate | 23.0 % |
| seat 0 royalty / game | 6.00 |
| seat 1 royalty / game | 6.01 |
| Sum EV (chips) over all games | −19,622 |
| **Per‑game EV difference (seat 1 − seat 0)** | **+0.078** chips |

→ Seat 1 (the second to act) has a tiny edge of **0.078 EV/game** under `MC × MC` with the v3 book. The game is essentially balanced; the foul rate per board is ≈ 3.5 %; scoops happen ≈ 22 % of the time per seat.

---

## 15. Caveats

1. **Fantasy foul rates are inflated.** The `run_250k_v3` self‑play was generated **before** the engine joker‑substitution fix (commit `0613287`). Fantasy‑phase foul rates (F14 0.07 %, F15 0.16 %, F16 0.29 %, F17 0.28 %) are *engine artefacts*, not solver mistakes; the next data refresh will retire them. **NORMAL foul rate (4.37 %) is correct.**
2. **The book is sound for street 1 only.** Streets 2–5 still need MC search or the per‑signature lookup tables.
3. **`n_rollouts = 120` per candidate.** Standard errors on `ev_mean` are typically `0.6–0.9` chips. The book is *not* precise enough to distinguish two candidates whose mean‑EV differs by `< 0.4` chips; treat such ties as ties.
4. **Joker symmetry.** The book canonicalises `(*1, *2)` to a single orbit. When using `book.candidates(hand)` to display real‑suit placements, both joker IDs are populated; do not display "*1" / "*2" as distinct kinds.
5. **Royalty distribution is inflated by fantasy hands.** ~35 % of the 482 k boards in `royalty_by_row` are fantasy deals (14–17 cards), which biases "bottom = quads/full house" rates upward. For a non‑fantasy seat the bottom is more like 30 % full‑or‑better, not 50 %.

---

## 16. Open questions / future work

- Re‑run `run_250k_v3` with the joker‑fixed engine to retire the fantasy foul artefact (a 6‑hour job).
- The book stores only `top_k = 5` candidates per orbit. For ~5 % of orbits the 2nd‑best is within `0.1` EV of the best; widening to `top_k = 10` would let the horizon‑aware re‑ranking flip a few more orbits. Cost: ≈ 2× build time, 1.6× disk.
- The current MC×MC self‑play uses a uniform `fantasy_rate = 0.20`. The empirical `NORMAL → fantasy` transition rate from real play is only `3.14 %`. The injection weights `(F14 0.55, F15 0.25, F16 0.12, F17 0.08)` over‑represent the rarer F16/F17 tiers; if we re‑weighted toward the empirical mix the value function would shift downward by ~10–15 % for F16/F17.
