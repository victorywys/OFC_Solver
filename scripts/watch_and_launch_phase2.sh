#!/usr/bin/env bash
# Watch the v3 opening book build, then auto-launch Phase 2 self-play.
#
# Stages:
#   1. Wait for build_full_opening_book (PID in artifacts/opening_book_canonical_v3/build.pid)
#      to exit.
#   2. Sanity-check the output pickle: file exists, loadable, schema=rich,
#      n_entries == 152,646.
#   3. Launch build_all_tables (Phase 2) at
#      artifacts/run_250k_v3/.
#   4. Record everything to artifacts/watcher.log.
#
# Run with: nohup bash scripts/watch_and_launch_phase2.sh > /dev/null 2>&1 &

set -u  # treat unset vars as error

REPO="/home/yansenwang/code/OFC_solver"
cd "$REPO" || exit 1

WATCH_LOG="$REPO/artifacts/watcher.log"
V3_PID_FILE="$REPO/artifacts/opening_book_canonical_v3/build.pid"
V3_BOOK_PKL="$REPO/artifacts/opening_book_canonical_v3/opening_book_canonical.pkl"
V3_BOOK_META="$REPO/artifacts/opening_book_canonical_v3/opening_book_canonical.meta.json"
PHASE2_OUT="$REPO/artifacts/run_250k_v3"
PHASE2_LOG="$REPO/artifacts/run_250k_v3.log"
PHASE2_PID_FILE="$REPO/artifacts/run_250k_v3.pid"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$WATCH_LOG"
}

log "watcher started (pid=$$)"

# ---------------------------------------------------------------------------
# Stage 1: wait for the v3 build to exit
# ---------------------------------------------------------------------------
if [[ ! -f "$V3_PID_FILE" ]]; then
    log "FATAL: $V3_PID_FILE not found; aborting"
    exit 1
fi

V3_PID=$(cat "$V3_PID_FILE")
log "v3 build PID: $V3_PID — waiting for it to exit…"

# Poll every 5 minutes. /proc/$pid disappears when the process exits.
while [[ -d "/proc/$V3_PID" ]]; do
    sleep 300
done

log "v3 build PID $V3_PID has exited"

# ---------------------------------------------------------------------------
# Stage 2: sanity-check the output
# ---------------------------------------------------------------------------
if [[ ! -f "$V3_BOOK_PKL" ]]; then
    log "FATAL: $V3_BOOK_PKL not found after build exit; aborting Phase 2 launch"
    exit 1
fi

log "verifying $V3_BOOK_PKL …"
python - <<PYEOF >> "$WATCH_LOG" 2>&1
import pickle, sys
from tables.canonical_opening import CanonicalOpeningBookTable
with open("$V3_BOOK_PKL", "rb") as f:
    tbl = pickle.load(f)
assert isinstance(tbl, CanonicalOpeningBookTable), f"bad type: {type(tbl)}"
assert tbl.is_rich(), "v3 book is not rich schema (expected rich)"
assert len(tbl) == 152646, f"expected 152,646 entries, got {len(tbl):,}"
print(f"sanity OK: {tbl!r}")
PYEOF
SANITY_RC=$?
if [[ $SANITY_RC -ne 0 ]]; then
    log "FATAL: sanity check failed (rc=$SANITY_RC); aborting Phase 2 launch"
    exit 1
fi
log "sanity check PASSED"

# Spot-check the previously-buggy hand: AA+high should record positive EV
# (rather than -25 EV with 89% foul) for the top candidate.
log "spot-checking AA+high regression…"
python - <<PYEOF >> "$WATCH_LOG" 2>&1
import pickle
from engine.cards import parse_cards, is_joker
from state.board import SLOT_BOTTOM
with open("$V3_BOOK_PKL", "rb") as f:
    tbl = pickle.load(f)
hand = parse_cards("5c Qd Kc Ah As")
recs = tbl.candidates(hand)
print(f"AA+high (5c Qd Kc Ah As): {len(recs)} candidates stored")
for i, r in enumerate(recs):
    slot_of = {c: s for c, s in r.placements}
    ace_slots = [slot_of[c] for c in hand if not is_joker(c) and (c >> 2) == 12]
    on_bot = ace_slots == [SLOT_BOTTOM, SLOT_BOTTOM]
    marker = " <-- AA-on-bottom" if on_bot else ""
    print(f"  #{i+1}: ev_mean={r.ev_mean:+6.2f} foul={r.foul_rate*100:5.1f}% fent={r.fantasy_entry_rate*100:5.1f}%{marker}")
best = recs[0]
slot_of = {c: s for c, s in best.placements}
ace_slots = [slot_of[c] for c in hand if not is_joker(c) and (c >> 2) == 12]
if ace_slots == [SLOT_BOTTOM, SLOT_BOTTOM] and best.ev_mean > 0:
    print("REGRESSION FIX CONFIRMED: AA+high best is AA-on-bottom with positive EV")
elif best.ev_mean > 0:
    print(f"WARNING: AA+high best is positive EV but not AA-on-bottom (ace_slots={ace_slots})")
else:
    print(f"WARNING: AA+high best ev_mean is {best.ev_mean:+.2f} (regression may not be fully fixed)")
PYEOF

# ---------------------------------------------------------------------------
# Stage 3: launch Phase 2 self-play
# ---------------------------------------------------------------------------
log "launching Phase 2 self-play (build_all_tables → $PHASE2_OUT)…"

nohup python -m scripts.build_all_tables \
    --n-games 250000 --n-workers 20 --p0 mc --p1 mc --fantasy-rate 0.20 \
    --out "$PHASE2_OUT" \
    > "$PHASE2_LOG" 2>&1 &

# Capture the real python parent PID (not the shell wrapper).
sleep 10
PHASE2_PID=$(pgrep -f 'python.*scripts.build_all_tables' | head -1)
if [[ -z "$PHASE2_PID" ]]; then
    log "FATAL: failed to find Phase 2 python parent after launch"
    exit 1
fi
echo "$PHASE2_PID" > "$PHASE2_PID_FILE"
log "Phase 2 launched: PID=$PHASE2_PID, log=$PHASE2_LOG"

# Confirm 20 workers spawned (give them another 30s to come up).
sleep 30
WORKER_COUNT=$(pgrep -P "$PHASE2_PID" -fc spawn_main || echo 0)
log "Phase 2 worker count: $WORKER_COUNT"
if [[ "$WORKER_COUNT" -lt 18 ]]; then
    log "WARNING: only $WORKER_COUNT workers spawned (expected 20)"
fi

log "watcher done — Phase 2 will run unattended; check $PHASE2_LOG"
