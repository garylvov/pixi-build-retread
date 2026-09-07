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

# HARNESS-CONSOL-15, 2026-09-07: ARMS A-C WERE THE LAST TEMPLATE-EXECUTING ARMS
# IN THIS FILE, AND THEY BECAME LIVE THE DAY THE TEMPLATES WENT CLEAN.
#
# `run()` was `SLURM_JOB_ID=999999 bash "$1"` over an UNMODIFIED copy of the
# template. That is only ever safe while the copy EXITS 9 EARLY, and arms A and
# C are the two arms that assert the check comes back CLEAN -- so on a clean
# template `bash` keeps going: past the gates, into stage_build_mirror, against
# the SHARED mirror under STAGE_MIRROR_ROOT, with the job id forced to 999999.
# Arm H's own note already records what that costs, measured on hc14-guard
# 6023585: eleven minutes building a mirror and the leftover
# `85db7fdbbf51206a0cb57fa0d55e0e74.building.999999-MH1-2247202` in the mirror
# root. H was rewritten to cut the check out; A-C were not, and they aimed at
# phaseN_relock.sh rather than mh1 only by luck of which gate refused first.
# The templates are clean as of this commit, so the luck has run out.
#
# THE FIX IS THE ONE ARMS D-G ALREADY USE: every file these arms execute gets
# `exit 0` spliced immediately after `### LEFTOVER-CHECK END`, so the check runs
# over the WHOLE file on disk (the awk scans `"$0"`, not a truncated copy) and
# the process stops the instant the check has spoken. Nothing above the check
# does anything but assign variables.
#
# AND THE SPLICE IS VERIFIED, NOT ASSUMED. `arm_fixture` refuses -- loudly, and
# without running anything -- if the marker was not found, because a silent
# splice failure is the same runaway with a green row over it. That refusal is
# the guard rail; the `exit 0` is only the mechanism.
#
# The path still carries $TOKEN, because arm A is ABOUT the filename: the check
# must scan the LINE, never "FILENAME:LNO: line". Splicing in place keeps that.
arm_fixture() {   # arm_fixture <file, rewritten in place>
  awk '{print} /^### LEFTOVER-CHECK END/{print "exit 0"}' "$1" > "$1.fix" || return 1
  if ! grep -qx 'exit 0' "$1.fix" || [ "$(grep -c '^### LEFTOVER-CHECK END' "$1")" != 1 ]; then
    rm -f "$1.fix"; return 1
  fi
  mv -f "$1.fix" "$1"
}
run() {
  if ! arm_fixture "$1"; then
    say "  FAIL  could not splice 'exit 0' after the check in $1 -- REFUSING to execute a live template"
    FAIL=1; return 99
  fi
  bash "$1" 2>&1
}

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

  # A2. HARNESS-CONSOL-15's own reader: the arms above must not be able to run a
  # template body. After A-C, the three fixtures must each carry the spliced
  # `exit 0`, and NONE of them may have reached a stage-mirror row -- the first
  # thing a runaway prints.
  for f in "$P" "$P2" "$P3"; do
    grep -qx 'exit 0' "$f" || { say "  FAIL  A2 $f was executed without the splice"; FAIL=1; }
  done
  if printf '%s' "$OUT" | grep -q 'stage(mirror)\|stage: building\|PRE-LOCK mirror verify'; then
    say "  FAIL  A2 a template body RAN -- a stage-mirror row reached the guard's stdout"; FAIL=1
  else
    say "  PASS  A2 all three fixtures carry the splice and no template body ran"
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


# ── ARM H: MERGE-V-2-1. A JOB ID IS A PREVIOUS BATCH'S TOKEN ────────────────
# The tip's sdist-scoping paragraph cited `mCA job 6000717` as a plain comment,
# outside every marked region. The check greps comments ON PURPOSE, so the
# moment a derived batch put that id into LEFTOVER_RE the shipped template would
# exit 9 AGAINST ITSELF -- for a deliberate citation, not a botched derivation.
#
# THE ARM RUNS THE CHECK, NOT THE HARNESS, AND THAT DISTINCTION COST A JOB.
# The first cut of this arm did what arms A-C do -- `bash <template>` -- and
# that is only safe while the check EXITS 9 early. Once the template is clean,
# `bash arms/mh1_relock.sh` keeps going: past the gates, into stage_build_mirror,
# against the SHARED stage mirror under STAGE_MIRROR_ROOT, with SLURM_JOB_ID
# forced to 999999. Guard job 6023585 did exactly that and sat there for minutes
# building a mirror; the Aug-19 leftovers `…building.999999-MH1-…` in the mirror
# root are older evidence of the same shape. A guard that can write production
# state is not fixture-only however green it prints. So H CUTS the check --
# the LEFTOVER_RE line and the marked LEFTOVER-CHECK region, verbatim, from the
# template -- and runs THAT over the template as data. One substitution is made
# and it is asserted below: the region scans `"$0"`, and the probe must scan the
# TEMPLATE instead of itself.
#
# THE TOKEN SET IS THE JOB ID MERGE-V-2-1 NAMED, and nothing more. A first cut
# spliced in `b30|B30|mCA|mCB` as well and arm H went RED on both templates --
# correctly: those strings are all over the prose that EXPLAINS the merge
# campaign, and no real derived batch would put a lane name its own commentary
# uses in every paragraph into its regex. Widening a guard's regex past what the
# defect was is how a guard starts reporting its own fixture.
say "== H: MERGE-V-2-1, a previous batch's JOB ID in LEFTOVER_RE =="
B30_TOKENS='6000717'

mk_leftover_probe () {   # $1 = template, $2 = out probe, $3 = extra regex alternatives
  { echo '#!/usr/bin/env bash'
    echo 'set -uo pipefail'
    echo 'TARGET=${1:?the file to scan}'
    grep -m1 "^LEFTOVER_RE='" "$1" | sed "s@^LEFTOVER_RE='@LEFTOVER_RE='$3|@"
    awk '/^### LEFTOVER-CHECK BEGIN/{p=1; next} /^### LEFTOVER-CHECK END/{p=0} p' "$1" \
      | sed 's@"\$0")@"$TARGET")@'
  } > "$2"
}

for REL in phaseN_relock.sh ../arms/mh1_relock.sh; do
  SRCF=$HERE/$REL
  [ -f "$SRCF" ] || { say "  FAIL  H no file at $SRCF"; FAIL=1; continue; }
  B=$(basename "$REL")
  mkdir -p "$W/h"
  HP=$W/h/probe-$B
  mk_leftover_probe "$SRCF" "$HP" "$B30_TOKENS"
  if ! grep -q "LEFTOVER_RE='$B30_TOKENS|" "$HP" \
     || ! grep -q 'leftover-token self-check' "$HP" \
     || ! grep -q '"\$TARGET")' "$HP" \
     || grep -q '"\$0")' "$HP" \
     || ! bash -n "$HP" 2>/dev/null; then
    say "  FAIL  H could not cut the check out of $B -- the arm did not run"; FAIL=1; continue
  fi
  say "  PASS  H the check was CUT from $B ($(wc -l < "$HP") lines), and the probe scans the template, never itself"
  OUT=$(bash "$HP" "$SRCF" 2>&1); RC=$?
  if [ "$RC" = 0 ] && printf '%s' "$OUT" | grep -q 'leftover-token self-check: clean'; then
    say "  PASS  H $B is clean with $B30_TOKENS in LEFTOVER_RE"
  else
    say "  FAIL  H $B trips its own leftover check on a deliberate citation (rc=$RC):"
    printf '%s\n' "$OUT" | grep -m4 -E '6000717|leftover-token' | sed 's/^/          /'
    FAIL=1
  fi
  # H2. THE MUTATION: the citation pair removed from the SCANNED FILE.
  HM=$W/h/mut-$B
  grep -v '^### CITATION BEGIN' "$SRCF" | grep -v '^### CITATION END' > "$HM"
  if [ "$(grep -c '^### CITATION' "$HM")" != 0 ] || [ "$(grep -c '^### CITATION' "$SRCF")" = 0 ]; then
    say "  FAIL  H2 could not build the citation-stripped copy of $B"; FAIL=1
  else
    OUT=$(bash "$HP" "$HM" 2>&1); RC=$?
    if [ "$RC" = 9 ] && printf '%s' "$OUT" | grep -q '6000717'; then
      say "  PASS  H2 MUTATION -- with the CITATION pair removed $B exits 9 naming 6000717, so H is a real exemption"
    else
      say "  FAIL  H2 the citation-stripped $B did NOT exit 9 (rc=$RC) -- arm H proves nothing"; FAIL=1
    fi
  fi
done
[ "$FAIL" = 0 ] && { say "leftover-check guard (with MERGE-N-4 arms): ALL PASS"; exit 0; }
say "leftover-check guard (with MERGE-N-4 arms): FAILED"; exit 1
