#!/usr/bin/env bash
# Guard for the leftover-token self-check, both directions.
#
# The check used to pipe "FILENAME:LNO: line" into grep, so it matched its OWN
# FILENAME: a harness derived into a directory named after a previous batch
# (p6b-c3b, here) failed against itself on every line, with the token nowhere in
# its body. That is a scan that can match itself, which is never allowed to be
# the thing deciding an exit code. The match now runs inside awk, on the LINE.
#
#   A. a harness whose PATH contains a leftover token, but whose BODY does not,
#      must pass.
#   B. a leftover token in the BODY, outside the three marked regions, must
#      still fail with exit 9 and name the line -- the check must not have been
#      weakened into uselessness.
#   C. a token inside a marked region (EVIDENCE / SUBSTITUTE / LEFTOVER-CHECK)
#      must still be exempt.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
FAIL=0
say() { printf '%s\n' "$*"; }

TOKEN=bfinal            # already in every template's LEFTOVER_RE
mkdir -p "$W/$TOKEN-batch"
run() { SLURM_JOB_ID=999999 bash "$1" 2>&1; }

for TPL in phaseN_relock.sh phaseN_cert.sh; do
  say "== $TPL =="
  # A. the token is in the PATH only.
  P="$W/$TOKEN-batch/$TPL"; cp "$HERE/$TPL" "$P"
  OUT=$(run "$P"); RC=$?
  if printf '%s' "$OUT" | grep -q 'leftover-token self-check: clean'; then
    say "  PASS  a path containing '$TOKEN' does not trip the check"
  else
    say "  FAIL  the check matched its own FILENAME (rc=$RC)"; printf '%s\n' "$OUT" | head -4; FAIL=1
  fi

  # B. the token in the BODY, outside every marked region, must still fail 9.
  P2="$W/$TOKEN-batch/body-$TPL"
  awk -v t="# $TOKEN leftover planted by the guard" \
      '{print} /^### LEFTOVER-CHECK END/{print t}' "$HERE/$TPL" > "$P2"
  OUT=$(run "$P2"); RC=$?
  if [ "$RC" = 9 ] && printf '%s' "$OUT" | grep -q 'leftover planted by the guard'; then
    say "  PASS  a token in the body still fails 9 and names the line"
  else
    say "  FAIL  a planted leftover was NOT caught (rc=$RC)"; FAIL=1
  fi

  # C. the same token inside the EVIDENCE region stays exempt.
  P3="$W/$TOKEN-batch/evid-$TPL"
  awk -v t="# $TOKEN cited deliberately in EVIDENCE" \
      '/^### EVIDENCE BEGIN/{print; print t; next} {print}' "$HERE/$TPL" > "$P3"
  OUT=$(run "$P3")
  if printf '%s' "$OUT" | grep -q 'leftover-token self-check: clean'; then
    say "  PASS  a deliberate citation inside EVIDENCE stays exempt"
  else
    say "  FAIL  the EVIDENCE region is no longer exempt"; FAIL=1
  fi
done


# =============================================================================
# MERGE-N-4 (2026-09-06, HARNESS-FIX-2). THE DEFECT, AND WHY THE ARMS ABOVE
# COULD NOT SEE IT.
#
# The versioned `arms/mh1_relock.sh` FAILED ITS OWN self-check and exited 9 for
# every lane that re-derived from it -- which READERS-1-1 makes the REQUIRED way
# to build a merge lane's relock script, so this blocked the next merge lane
# outright. The hit was ONE line, a comment READERS-1 added under MERGE-M-1:
#
#     # stale on 2026-09-03 15:35, when p6i merged and that file RE-ENABLED
#
# `p6i` is a token in that harness's OWN `LEFTOVER_RE`. The citation is correct
# and load-bearing; the check was right to see it and wrong to refuse.
#
# THE FIX IS A FOURTH STRIPPED REGION, `### CITATION BEGIN` / `### CITATION END`,
# and NOT a blanket comment exemption. Exempting comments would have deleted the
# check's whole purpose -- its own header says a stale path in a COMMENT is the
# thing it was written for. A citation pair is a deliberate, per-site opt-out
# that a botched derivation never carries, so arm E below still catches a real
# leftover, in a comment, with the region in place.
#
# ARMS. Every file under `harness/` that carries the check is covered, not just
# the two templates -- the defect was in an ARM, and the arms had no guard.
#   D  the VERSIONED file, unmodified, must pass its own check clean. THIS IS
#      MERGE-N-4: `arms/mh1_relock.sh` was rc=9 here before the fix.
#   E  a `bfinal` leftover planted in the BODY, outside every region, must still
#      exit 9 and name the line. The check must not have been weakened.
#   E2 mh1 only: the REAL token, a `p6i` body literal outside every region, must
#      still be caught -- the exact string the region now exempts in ONE place.
#   F  the same token inside a `### CITATION BEGIN`/`### CITATION END` pair is
#      exempt.
#   G  MUTATION, PINNED TO A COMMIT CONSTANT: `$N4_OLD:harness/arms/mh1_relock.sh`
#      -- the file as it stood when MERGE-N-4 was boarded -- fixture-derived and
#      run, must exit 9 on its own `p6i` comment. Without this arm D proves
#      nothing.
#   G2 MUTATION: the pinned OLD file with a CITATION-wrapped token must STILL
#      fail, because that build knows no such region. Without it arm F could
#      pass on a check that exempts everything.
#
# HOW A RELOCK SCRIPT IS RUN WITHOUT RUNNING A RELOCK. Every fixture is the file
# with `exit 0` inserted immediately after `### LEFTOVER-CHECK END`. The awk
# scans `"$0"` -- the WHOLE file on disk -- so the check still sees every line
# below the exit, including the line 649 comment that is the defect. Nothing
# above the check does anything but assign variables. A relock is never started.
N4_OLD=${N4_OLD:-6279978}   # the HARNESS_COMMIT that carried the defect
# WHERE THE REPO IS -- same rule as cleanup_gate_env_derivation_guard.sh, and it
# is load-bearing here: `arms/`, `instrumented/` and `proof/` exist ONLY in the
# repo, so from the task copy at `<T>/tools/phase_template/` the relative guess
# is wrong and HARNESS_REPO has to win.
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
if [ ! -d "$REPO/harness/tools" ] || ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  REPO=/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
fi
H=$REPO/harness
if [ ! -d "$H/arms" ]; then
  say "FAIL  MERGE-N-4 arms cannot run: no harness tree at $H"; FAIL=1
else
CHECKED_FILES="arms/mh1_relock.sh arms/c29_relock.sh instrumented/p6b_relock.sh instrumented/p6b_relock.b2.sh phase_template/phaseN_cert.sh phase_template/phaseN_relock.sh proof/hlgd_relock.sh"

# fixture <src> <dst>: the file, plus `exit 0` right after the check.
fixture() { awk '{print} /^### LEFTOVER-CHECK END/{print "exit 0"}' "$1" > "$2"; }

say "== MERGE-N-4: the leftover check over every file that carries it =="
for rel in $CHECKED_FILES; do
  SRC=$H/$rel
  [ -f "$SRC" ] || { say "FAIL  D $rel: no such file at $SRC"; FAIL=1; continue; }
  TAGN=$(printf '%s' "$rel" | tr / _)

  # ---- D: the versioned file passes its own check --------------------------
  fixture "$SRC" "$W/D-$TAGN"
  OUT=$(bash "$W/D-$TAGN" 2>&1); RC=$?
  if [ "$RC" = 0 ] && printf '%s' "$OUT" | grep -q 'leftover-token self-check: clean'; then
    say "  PASS  D $rel passes its own self-check (rc=0)"
  else
    say "  FAIL  D $rel FAILS its own self-check (rc=$RC) -- a lane re-deriving from it exits 9"
    printf '%s\n' "$OUT" | grep -v '^### leftover-token' | head -4; FAIL=1
  fi

  # ---- E: a real leftover in the body is still caught -----------------------
  awk -v t="# bfinal leftover planted by the MERGE-N-4 guard" \
      '{print} /^### LEFTOVER-CHECK END/{print t; print "exit 0"}' "$SRC" > "$W/E-$TAGN"
  OUT=$(bash "$W/E-$TAGN" 2>&1); RC=$?
  if [ "$RC" = 9 ] && printf '%s' "$OUT" | grep -q 'leftover planted by the MERGE-N-4 guard'; then
    say "  PASS  E $rel still catches a planted body leftover (rc=9, line named)"
  else
    say "  FAIL  E $rel did NOT catch a planted leftover (rc=$RC) -- the check is weakened"; FAIL=1
  fi

  # ---- F: the same token inside a CITATION pair is exempt -------------------
  awk -v a="### CITATION BEGIN" -v t="# bfinal cited deliberately beside its code" -v z="### CITATION END" \
      '{print} /^### LEFTOVER-CHECK END/{print a; print t; print z; print "exit 0"}' "$SRC" > "$W/F-$TAGN"
  OUT=$(bash "$W/F-$TAGN" 2>&1); RC=$?
  if [ "$RC" = 0 ] && printf '%s' "$OUT" | grep -q 'leftover-token self-check: clean'; then
    say "  PASS  F $rel exempts a token inside a CITATION pair"
  else
    say "  FAIL  F $rel does not honour the CITATION region (rc=$RC)"; FAIL=1
  fi
done

# ---- E2: mh1's REAL token, the one the region exempts in one place ----------
awk -v t="# p6i planted raw by the MERGE-N-4 guard, outside every region" \
    '{print} /^### LEFTOVER-CHECK END/{print t; print "exit 0"}' "$H/arms/mh1_relock.sh" > "$W/E2"
OUT=$(bash "$W/E2" 2>&1); RC=$?
if [ "$RC" = 9 ] && printf '%s' "$OUT" | grep -q 'planted raw by the MERGE-N-4 guard'; then
  say "  PASS  E2 arms/mh1_relock.sh still catches a RAW p6i body literal (rc=9)"
else
  say "  FAIL  E2 the CITATION region exempted p6i everywhere, not just at its one site (rc=$RC)"; FAIL=1
fi

# ---- G / G2: THE MUTATION, pinned to a commit constant ----------------------
OLDF=$W/mh1_relock.$N4_OLD.sh
if git -C "$REPO" show "$N4_OLD:harness/arms/mh1_relock.sh" > "$OLDF" 2>/dev/null && [ -s "$OLDF" ]; then
  fixture "$OLDF" "$W/G"
  OUT=$(bash "$W/G" 2>&1); RC=$?
  if [ "$RC" = 9 ] && printf '%s' "$OUT" | grep -q 'p6i merged'; then
    say "  PASS  G the pinned $N4_OLD arm exits 9 on its own p6i comment -- MERGE-N-4, reproduced"
  else
    say "  FAIL  G $N4_OLD did not reproduce MERGE-N-4 (rc=$RC) -- WRONG PIN, the mutation is not the defect"; FAIL=1
  fi
  awk -v a="### CITATION BEGIN" -v t="# bfinal cited deliberately beside its code" -v z="### CITATION END" \
      '{print} /^### LEFTOVER-CHECK END/{print a; print t; print z; print "exit 0"}' "$OLDF" > "$W/G2"
  OUT=$(bash "$W/G2" 2>&1); RC=$?
  if [ "$RC" = 9 ]; then
    say "  PASS  G2 the pinned $N4_OLD arm knows no CITATION region -- arm F is not vacuous"
  else
    say "  FAIL  G2 $N4_OLD already honoured a CITATION pair (rc=$RC) -- arm F proves nothing"; FAIL=1
  fi
else
  say "  FAIL  G could not extract $N4_OLD:harness/arms/mh1_relock.sh -- THE MUTATION ARMS DID NOT RUN"; FAIL=1
fi
fi

[ "$FAIL" = 0 ] && { say "leftover-check guard (with MERGE-N-4 arms): ALL PASS"; exit 0; }
say "leftover-check guard (with MERGE-N-4 arms): FAILED"; exit 1
