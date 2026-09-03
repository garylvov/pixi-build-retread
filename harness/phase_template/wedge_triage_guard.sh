#!/usr/bin/env bash
# Guard for wedge_triage.sh (p6l-1, 2026-09-03).  usage: wedge_triage_guard.sh [<path to wedge_triage.sh>]
#
# Three properties, each of which has ALREADY been got wrong by a human on this
# campaign, in both directions:
#
#   A. An open lockd outage must win over everything else. On 2026-09-03 a
#      44m53s NLM outage on node2347 was read as a backend hang and a healthy
#      60-minute install came within minutes of being SIGTERMed.
#   B. A pipe_read whose write end is held by a LIVE relative must NOT be called
#      WEDGED. That is the pixi->backend RPC stdin with no request in flight, and
#      it is what a correct idle backend looks like.
#   C. The script must still be able to say WEDGED. A triage that can only ever
#      exonerate is as useless as one that always convicts -- on 2026-09-02 two
#      real wedges were correctly killed, and that ability must survive.
#
# It drives the REAL wedge_triage.sh against REAL processes (a real coproc pipe,
# a real idle sleep) with only the dmesg source and the sample interval
# substituted, because a fake /proc would test the fixture and not the script.
# Nothing is signalled except the guard's own fixture pids, which it started.
#
# Falsification (proved by hand, see LANE-SPEED-LOG "p6l"): delete the (a) lockd
# block and case A fails; make (b) fire only on "other" holders (or drop the
# early exit) and case B fails; make the (d) verdict unconditionally ADVANCING
# and case C fails.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SUT=${1:-$HERE/wedge_triage.sh}
W=$(mktemp -d)
FIXPIDS=""
cleanup() { for p in $FIXPIDS; do kill "$p" 2>/dev/null; done; rm -rf "$W"; }
trap cleanup EXIT
FAIL=0
say() { printf '%s\n' "$*"; }
ok()   { say "PASS  $*"; }
bad()  { say "FAIL  $*"; FAIL=1; }

say "guard: script under test = $SUT"
[ -x "$SUT" ] || { say "FAIL  not executable: $SUT"; exit 1; }
bash -n "$SUT" || { say "FAIL  bash -n"; exit 1; }

# ---- dmesg fixtures ---------------------------------------------------------
cat > "$W/dmesg_outage.txt" <<'EOF'
[Thu Sep  3 12:10:29 2026] lockd: server hpcnfs.ccv.brown.edu not responding, still trying
[Thu Sep  3 12:10:35 2026] lockd: server hpcnfs.ccv.brown.edu OK
[Thu Sep  3 12:54:22 2026] lockd: server hpcnfs.ccv.brown.edu not responding, still trying
EOF
cat > "$W/dmesg_clean.txt" <<'EOF'
[Thu Sep  3 12:54:22 2026] lockd: server hpcnfs.ccv.brown.edu not responding, still trying
[Thu Sep  3 13:39:15 2026] lockd: server hpcnfs.ccv.brown.edu OK
EOF
mkdir -p "$W/artifacts"; printf 'frozen\n' > "$W/artifacts/cert-install.log"

run() { # dmesgfile pid -> stdout in $OUT, exit in $RC
  OUT=$(WT_DMESG_CMD="cat $1" WT_SAMPLE_SECS=3 WT_ARTIFACT_DIR="$W/artifacts" \
        "$SUT" "$2" 2>&1); RC=$?
}

# ---- case A: open lockd outage ---------------------------------------------
sleep 120 </dev/null >/dev/null 2>&1 & A_PID=$!; FIXPIDS="$FIXPIDS $A_PID"
run "$W/dmesg_outage.txt" "$A_PID"
printf '%s\n' "$OUT" > "$W/caseA.log"
if printf '%s' "$OUT" | grep -q '^NFS-LOCK-OUTAGE (since ' && [ "$RC" = 10 ]; then
  ok "A  dmesg ending in 'not responding' -> $(printf '%s' "$OUT" | grep '^NFS-LOCK-OUTAGE' ) (rc=10)"
else
  bad "A  expected NFS-LOCK-OUTAGE + rc=10, got rc=$RC verdict='$(printf '%s' "$OUT" | tail -12 | grep -E '^(NFS-LOCK-OUTAGE|IDLE-ON-RPC-CHANNEL|WEDGED|ADVANCING)' | head -1)'"
fi
if printf '%s' "$OUT" | grep -qi 'WEDGED'; then bad "A  a lockd outage must never print WEDGED"; fi

# ---- case B: pipe_read, write end held by a live parent ---------------------
bash -c 'coproc RD { exec cat; }; echo $RD_PID > "'"$W"'/catpid"; while :; do sleep 1; done' >/dev/null 2>&1 &
B_PARENT=$!; FIXPIDS="$FIXPIDS $B_PARENT"
for i in $(seq 1 40); do [ -s "$W/catpid" ] && break; sleep 0.25; done
B_PID=$(cat "$W/catpid" 2>/dev/null)
if [ -z "$B_PID" ] || [ ! -d "/proc/$B_PID" ]; then
  bad "B  fixture did not start (no blocked pipe reader)"
else
  FIXPIDS="$FIXPIDS $B_PID"
  sleep 1
  WCH=$(cat "/proc/$B_PID/task/$B_PID/wchan" 2>/dev/null)
  [ "$WCH" = "pipe_read" ] || say "note  B fixture wchan is '$WCH' (expected pipe_read)"
  run "$W/dmesg_clean.txt" "$B_PID"
  printf '%s\n' "$OUT" > "$W/caseB.log"
  if printf '%s' "$OUT" | grep -q '^IDLE-ON-RPC-CHANNEL (write end held by ' && [ "$RC" = 11 ]; then
    ok "B  pipe_read w/ write end on a live parent -> $(printf '%s' "$OUT" | grep '^IDLE-ON-RPC') (rc=11)"
  else
    bad "B  expected IDLE-ON-RPC-CHANNEL + rc=11, got rc=$RC verdict='$(printf '%s' "$OUT" | grep -E '^(NFS-LOCK-OUTAGE|IDLE-ON-RPC-CHANNEL|WEDGED|ADVANCING)' | head -1)'"
  fi
  if printf '%s' "$OUT" | grep -q '^WEDGED'; then bad "B  an idle RPC channel must never be called WEDGED"; fi
fi

# ---- case C: flat counters, no lockd and no pipe cause ----------------------
sleep 120 </dev/null >/dev/null 2>&1 & C_PID=$!; FIXPIDS="$FIXPIDS $C_PID"
sleep 0.5
run "$W/dmesg_clean.txt" "$C_PID"
printf '%s\n' "$OUT" > "$W/caseC.log"
if printf '%s' "$OUT" | grep -q '^WEDGED (flat 3s; wchan=' && [ "$RC" = 12 ]; then
  ok "C  flat counters, lock service up, no pipe -> $(printf '%s' "$OUT" | grep '^WEDGED') (rc=12)"
else
  bad "C  expected WEDGED + rc=12, got rc=$RC verdict='$(printf '%s' "$OUT" | grep -E '^(NFS-LOCK-OUTAGE|IDLE-ON-RPC-CHANNEL|WEDGED|ADVANCING)' | head -1)'"
fi
if ! printf '%s' "$OUT" | grep -q "^    kill -TERM $C_PID$"; then
  bad "C  a WEDGED verdict must print the exact kill line for the operator"
fi

say ""
if [ $FAIL -eq 0 ]; then say "wedge_triage_guard: ALL PASS"; else say "wedge_triage_guard: FAILURES ABOVE"; fi
exit $FAIL
