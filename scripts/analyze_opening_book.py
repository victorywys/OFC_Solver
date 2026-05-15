"""Analyze the rich (v2) canonical opening book and emit summary stats.

Reads the per-orbit ``CandidateRecord`` lists from
``artifacts/opening_book_canonical_v2/opening_book_canonical.pkl`` and
prints structured statistics covering:

  * Global EV distribution + best-vs-worst gap (decision criticality).
  * Top-row usage by best action (overall + per-rank).
  * Pair strategy: where pairs go by pair rank.
  * Fantasy entry / foul / trip handling.
  * Joker handling (1 vs 2 jokers).
  * Best/worst openers by EV.

Usage::

    python -m scripts.analyze_opening_book \\
        --book artifacts/opening_book_canonical_v2/opening_book_canonical.pkl \\
        --out artifacts/opening_book_analysis.json

The JSON output is machine-readable and is also used to render the
human/LLM-facing strategy document.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from engine.cards import (
    JOKER_1,
    JOKER_2,
    NUM_STD_CARDS,
    RANK_CHARS,
    card_rank,
    card_str,
    is_joker,
)
from engine.fantasy import FantasyTier
from state.board import SLOT_BOTTOM, SLOT_DISCARD, SLOT_MIDDLE, SLOT_NAMES, SLOT_TOP
from tables.canonical_opening import CanonicalOpeningBookTable, CandidateRecord


# ---------------------------------------------------------------------------
# Hand-feature helpers
# ---------------------------------------------------------------------------
def hand_ranks(hand: tuple[int, ...]) -> tuple[int, ...]:
    """Return non-joker ranks in the hand (sorted ascending). Jokers omitted."""
    return tuple(sorted(card_rank(c) for c in hand if not is_joker(c)))


def n_jokers(hand: tuple[int, ...]) -> int:
    return sum(1 for c in hand if is_joker(c))


def rank_multiplicities(hand: tuple[int, ...]) -> dict[int, int]:
    """Map rank -> count (jokers excluded)."""
    out: dict[int, int] = defaultdict(int)
    for c in hand:
        if not is_joker(c):
            out[card_rank(c)] += 1
    return dict(out)


def pair_ranks(hand: tuple[int, ...]) -> list[int]:
    """Ranks that appear at least twice (jokers excluded)."""
    mults = rank_multiplicities(hand)
    return sorted([r for r, c in mults.items() if c >= 2], reverse=True)


def trip_ranks(hand: tuple[int, ...]) -> list[int]:
    """Ranks that appear at least three times."""
    mults = rank_multiplicities(hand)
    return sorted([r for r, c in mults.items() if c >= 3], reverse=True)


def hand_type_label(hand: tuple[int, ...]) -> str:
    """Coarse hand category label for grouping.

    Categories (priority order):
      "trips"       — three or more of one rank (rare)
      "two_pair"    — at least two distinct pairs
      "pair_<R>"    — exactly one pair, of rank R (chars 2..A)
      "high_card"   — no pairs
    Joker counts are appended as ``+1j`` / ``+2j`` when present.
    """
    n_jok = n_jokers(hand)
    trips = trip_ranks(hand)
    pairs = pair_ranks(hand)
    base: str
    if trips:
        base = f"trips_{RANK_CHARS[trips[0]]}"
    elif len(pairs) >= 2:
        base = "two_pair"
    elif len(pairs) == 1:
        base = f"pair_{RANK_CHARS[pairs[0]]}"
    else:
        base = "high_card"
    if n_jok == 1:
        base += "+1j"
    elif n_jok == 2:
        base += "+2j"
    return base


def pair_tier_label(rank: int) -> str:
    """Bucket a pair rank into a tier label (premium/high/mid/low)."""
    if rank >= 12:  # AA
        return "AA"
    if rank == 11:
        return "KK"
    if rank == 10:
        return "QQ"
    if rank == 9:
        return "JJ"
    if rank == 8:
        return "TT"
    if rank >= 5:  # 77-99
        return "mid_77-99"
    return "low_22-66"


# ---------------------------------------------------------------------------
# Action-feature helpers
# ---------------------------------------------------------------------------
def slot_breakdown(placements: tuple[tuple[int, int], ...]) -> dict[str, int]:
    """Return {'T': n_top, 'M': n_mid, 'B': n_bot, 'X': n_discard}."""
    out = {"T": 0, "M": 0, "B": 0, "X": 0}
    for _, s in placements:
        out[SLOT_NAMES[s]] += 1
    return out


def cards_in_slot(
    placements: tuple[tuple[int, int], ...],
    slot: int,
) -> list[int]:
    return [c for c, s in placements if s == slot]


def is_pair_in_slot(
    placements: tuple[tuple[int, int], ...],
    slot: int,
) -> bool:
    """Whether at least two cards of the same non-joker rank are in `slot`.

    Jokers in the slot also count toward forming a pair (joker is wild).
    """
    cs = cards_in_slot(placements, slot)
    ranks = [card_rank(c) for c in cs if not is_joker(c)]
    n_jok_in_slot = sum(1 for c in cs if is_joker(c))
    counts: dict[int, int] = defaultdict(int)
    for r in ranks:
        counts[r] += 1
    max_natural = max(counts.values(), default=0)
    # A joker can pair with any singleton in the slot.
    return (max_natural + n_jok_in_slot) >= 2


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": len(s),
        "min": s[0],
        "p05": percentile(s, 0.05),
        "p25": percentile(s, 0.25),
        "median": percentile(s, 0.5),
        "mean": statistics.fmean(s),
        "p75": percentile(s, 0.75),
        "p95": percentile(s, 0.95),
        "max": s[-1],
        "stdev": statistics.pstdev(s) if len(s) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-orbit walk
# ---------------------------------------------------------------------------
def iter_orbits(
    tbl: CanonicalOpeningBookTable,
) -> Iterable[tuple[tuple[int, ...], tuple[CandidateRecord, ...]]]:
    for key, val in tbl.entries.items():
        if isinstance(val, (tuple, list)) and val and isinstance(val[0], CandidateRecord):
            yield key, tuple(val)


def hand_to_str(hand: tuple[int, ...]) -> str:
    parts = []
    for c in hand:
        if c == JOKER_1:
            parts.append("*1")
        elif c == JOKER_2:
            parts.append("*2")
        else:
            parts.append(card_str(c))
    return " ".join(parts)


def placements_to_str(placements: tuple[tuple[int, int], ...]) -> str:
    by_slot: dict[int, list[int]] = defaultdict(list)
    for c, s in placements:
        by_slot[s].append(c)
    out = []
    for s in (SLOT_TOP, SLOT_MIDDLE, SLOT_BOTTOM, SLOT_DISCARD):
        if s in by_slot:
            cs = " ".join(
                "*1" if c == JOKER_1 else "*2" if c == JOKER_2 else card_str(c)
                for c in sorted(by_slot[s], reverse=True)
            )
            out.append(f"{SLOT_NAMES[s]}={cs}")
    return " | ".join(out)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyze(tbl: CanonicalOpeningBookTable) -> dict:
    if not tbl.is_rich():
        raise SystemExit("Book is legacy (v1). Re-run with a v2 rich book.")

    n_orbits = len(tbl)
    print(f"Analyzing {n_orbits:,} canonical orbits...")

    # ---- global accumulators ----
    best_evs: list[float] = []
    best_foul_rates: list[float] = []
    best_fent_rates: list[float] = []
    ev_gap_best_minus_worst: list[float] = []
    ev_gap_best_minus_2nd: list[float] = []
    best_n_rollouts: list[int] = []

    # ---- slot histograms (best action only) ----
    slot_histogram: Counter[tuple[int, int, int, int]] = Counter()
    n_top_histogram: Counter[int] = Counter()
    rank_on_top_count: Counter[int] = Counter()  # rank -> count of orbits where best action puts >=1 of this rank on top
    rank_in_hand_count: Counter[int] = Counter()  # rank -> count of orbits where hand contains rank

    # ---- hand-type rollups ----
    by_type_best_ev: dict[str, list[float]] = defaultdict(list)
    by_type_fent: dict[str, list[float]] = defaultdict(list)
    by_type_foul: dict[str, list[float]] = defaultdict(list)
    by_type_count: Counter[str] = Counter()

    # ---- pair strategy ----
    # For each pair_tier ("AA", "KK", ...): how often does the best action put the pair on top vs split vs middle vs bottom?
    pair_placement: dict[str, Counter[str]] = defaultdict(Counter)
    pair_tier_ev: dict[str, list[float]] = defaultdict(list)
    pair_tier_fent: dict[str, list[float]] = defaultdict(list)

    # ---- trip strategy ----
    trip_placement: dict[str, Counter[str]] = defaultdict(Counter)  # tier ("trips_A" etc) -> Counter
    trip_orbit_ev: dict[str, list[float]] = defaultdict(list)

    # ---- joker rollup ----
    by_jok_best_ev: dict[int, list[float]] = defaultdict(list)
    by_jok_fent: dict[int, list[float]] = defaultdict(list)
    by_jok_count: Counter[int] = Counter()

    # ---- top-N records ----
    top_ev_records: list[tuple[float, tuple[int, ...], CandidateRecord]] = []
    bot_ev_records: list[tuple[float, tuple[int, ...], CandidateRecord]] = []
    top_fent_records: list[tuple[float, tuple[int, ...], CandidateRecord]] = []

    # ---- pair-tier action gap (does the right placement matter?) ----
    pair_tier_gap: dict[str, list[float]] = defaultdict(list)
    by_type_gap: dict[str, list[float]] = defaultdict(list)

    t0 = time.time()
    for hand, recs in iter_orbits(tbl):
        best = recs[0]
        worst = recs[-1]
        ev_b = best.ev_mean
        ev_w = worst.ev_mean
        gap = ev_b - ev_w
        gap2 = ev_b - (recs[1].ev_mean if len(recs) > 1 else ev_b)

        best_evs.append(ev_b)
        best_foul_rates.append(best.foul_rate)
        best_fent_rates.append(best.fantasy_entry_rate)
        ev_gap_best_minus_worst.append(gap)
        ev_gap_best_minus_2nd.append(gap2)
        best_n_rollouts.append(best.n_rollouts)

        # slot breakdown
        sb = slot_breakdown(best.placements)
        slot_histogram[(sb["T"], sb["M"], sb["B"], sb["X"])] += 1
        n_top_histogram[sb["T"]] += 1

        # per-rank top placement
        ranks_in_hand = set(card_rank(c) for c in hand if not is_joker(c))
        for r in ranks_in_hand:
            rank_in_hand_count[r] += 1
        ranks_on_top = set(
            card_rank(c) for c, s in best.placements
            if s == SLOT_TOP and not is_joker(c)
        )
        for r in ranks_on_top:
            rank_on_top_count[r] += 1

        # hand type
        ht = hand_type_label(hand)
        by_type_best_ev[ht].append(ev_b)
        by_type_fent[ht].append(best.fantasy_entry_rate)
        by_type_foul[ht].append(best.foul_rate)
        by_type_count[ht] += 1
        by_type_gap[ht].append(gap2)

        # pair strategy (handles natural pairs only — joker wild-pairs analyzed elsewhere)
        prs = pair_ranks(hand)
        if prs:
            top_pair_rank = prs[0]
            tier = pair_tier_label(top_pair_rank)
            # find where the two (+ possibly more) pair cards landed
            pair_card_slots = [
                s for c, s in best.placements
                if not is_joker(c) and card_rank(c) == top_pair_rank
            ]
            placement_class = _classify_pair_placement(pair_card_slots)
            pair_placement[tier][placement_class] += 1
            pair_tier_ev[tier].append(ev_b)
            pair_tier_fent[tier].append(best.fantasy_entry_rate)
            pair_tier_gap[tier].append(gap2)

        # trip strategy
        trs = trip_ranks(hand)
        if trs:
            top_trip = trs[0]
            tier = f"trips_{RANK_CHARS[top_trip]}"
            trip_card_slots = sorted(
                s for c, s in best.placements
                if not is_joker(c) and card_rank(c) == top_trip
            )
            placement_class = _classify_trip_placement(trip_card_slots)
            trip_placement[tier][placement_class] += 1
            trip_orbit_ev[tier].append(ev_b)

        # joker rollup
        nj = n_jokers(hand)
        by_jok_best_ev[nj].append(ev_b)
        by_jok_fent[nj].append(best.fantasy_entry_rate)
        by_jok_count[nj] += 1

        # collect top/bottom-K records
        top_ev_records = _push_topk(top_ev_records, (ev_b, hand, best), k=25, reverse=True)
        bot_ev_records = _push_topk(bot_ev_records, (ev_b, hand, best), k=25, reverse=False)
        top_fent_records = _push_topk(
            top_fent_records,
            (best.fantasy_entry_rate, hand, best),
            k=25,
            reverse=True,
        )

    dt = time.time() - t0
    print(f"Walked {n_orbits:,} orbits in {dt:.2f}s.")

    # ---- assemble output ----
    out: dict = {
        "n_orbits": n_orbits,
        "best_ev": summarize(best_evs),
        "best_foul_rate": summarize(best_foul_rates),
        "best_fantasy_entry_rate": summarize(best_fent_rates),
        "ev_gap_best_minus_worst": summarize(ev_gap_best_minus_worst),
        "ev_gap_best_minus_2nd": summarize(ev_gap_best_minus_2nd),
        "best_n_rollouts": summarize([float(x) for x in best_n_rollouts]),
        "n_orbits_positive_ev": sum(1 for x in best_evs if x > 0),
        "n_orbits_zero_ev": sum(1 for x in best_evs if -0.05 < x < 0.05),
        "n_orbits_negative_ev": sum(1 for x in best_evs if x < 0),
        "slot_histogram": [
            {"T": k[0], "M": k[1], "B": k[2], "X": k[3], "count": v}
            for k, v in slot_histogram.most_common()
        ],
        "n_top_histogram": dict(sorted(n_top_histogram.items())),
        "rank_top_usage": [
            {
                "rank": RANK_CHARS[r],
                "in_hand": rank_in_hand_count[r],
                "on_top": rank_on_top_count[r],
                "p_top_given_in_hand": (
                    rank_on_top_count[r] / rank_in_hand_count[r]
                    if rank_in_hand_count[r] > 0 else 0.0
                ),
            }
            for r in range(13)
        ],
        "by_hand_type": {
            ht: {
                "count": by_type_count[ht],
                "best_ev": summarize(by_type_best_ev[ht]),
                "fantasy_entry_rate": summarize(by_type_fent[ht]),
                "foul_rate": summarize(by_type_foul[ht]),
                "ev_gap_best_minus_2nd": summarize(by_type_gap[ht]),
            }
            for ht in sorted(by_type_count.keys(), key=lambda k: -by_type_count[k])
        },
        "pair_tier_summary": {
            tier: {
                "count": sum(pair_placement[tier].values()),
                "placement": dict(pair_placement[tier]),
                "best_ev": summarize(pair_tier_ev[tier]),
                "fantasy_entry_rate": summarize(pair_tier_fent[tier]),
                "ev_gap_best_minus_2nd": summarize(pair_tier_gap[tier]),
            }
            for tier in [
                "AA", "KK", "QQ", "JJ", "TT", "mid_77-99", "low_22-66",
            ]
            if tier in pair_placement
        },
        "trip_tier_summary": {
            tier: {
                "count": sum(trip_placement[tier].values()),
                "placement": dict(trip_placement[tier]),
                "best_ev": summarize(trip_orbit_ev[tier]),
            }
            for tier in sorted(trip_placement.keys())
        },
        "by_joker_count": {
            nj: {
                "count": by_jok_count[nj],
                "best_ev": summarize(by_jok_best_ev[nj]),
                "fantasy_entry_rate": summarize(by_jok_fent[nj]),
            }
            for nj in sorted(by_jok_count.keys())
        },
        "top_ev_records": [_serialize_record(ev, h, r) for ev, h, r in top_ev_records],
        "worst_ev_records": [_serialize_record(ev, h, r) for ev, h, r in bot_ev_records],
        "top_fantasy_entry_records": [
            _serialize_record(ev, h, r) for ev, h, r in top_fent_records
        ],
    }
    return out


def _classify_pair_placement(slots: list[int]) -> str:
    """slots: list of slot ids where pair cards landed (typically 2 entries)."""
    s = sorted(slots)
    if s == [SLOT_TOP, SLOT_TOP]:
        return "TT"  # pair on top
    if s == [SLOT_MIDDLE, SLOT_MIDDLE]:
        return "MM"
    if s == [SLOT_BOTTOM, SLOT_BOTTOM]:
        return "BB"
    if s == [SLOT_TOP, SLOT_MIDDLE]:
        return "TM"
    if s == [SLOT_TOP, SLOT_BOTTOM]:
        return "TB"
    if s == [SLOT_MIDDLE, SLOT_BOTTOM]:
        return "MB"
    if SLOT_DISCARD in s:
        return "split_with_discard"
    return "other"


def _classify_trip_placement(slots: list[int]) -> str:
    """slots: list of slot ids where trip cards landed (typically 3 entries)."""
    s = tuple(sorted(slots))
    return "".join(SLOT_NAMES[x] for x in s)


def _push_topk(
    lst: list[tuple[float, tuple[int, ...], CandidateRecord]],
    item: tuple[float, tuple[int, ...], CandidateRecord],
    *,
    k: int,
    reverse: bool,
) -> list[tuple[float, tuple[int, ...], CandidateRecord]]:
    lst.append(item)
    lst.sort(key=lambda x: -x[0] if reverse else x[0])
    return lst[:k]


def _serialize_record(ev: float, hand: tuple[int, ...], rec: CandidateRecord) -> dict:
    return {
        "ev_mean": rec.ev_mean,
        "ev_se": rec.ev_se,
        "n_rollouts": rec.n_rollouts,
        "foul_rate": rec.foul_rate,
        "fantasy_entry_rate": rec.fantasy_entry_rate,
        "dest_tier_distribution": rec.dest_tier_distribution,
        "hand_canonical": hand_to_str(hand),
        "hand_type": hand_type_label(hand),
        "placements_canonical": placements_to_str(rec.placements),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--book",
        type=str,
        default="artifacts/opening_book_canonical_v2/opening_book_canonical.pkl",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/opening_book_analysis.json",
    )
    args = parser.parse_args()

    print(f"Loading {args.book}...")
    t0 = time.time()
    with open(args.book, "rb") as f:
        tbl = pickle.load(f)
    print(f"Loaded {tbl!r} in {time.time()-t0:.2f}s.")

    stats = analyze(tbl)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Wrote {out_path}.")


if __name__ == "__main__":
    main()
