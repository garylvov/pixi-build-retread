#!/usr/bin/env bash
# GUARD for cleanup_gated.sh ROOT FIX (d), 2026-09-05, p6ad-4.
#
# THE DEFECT IT GUARDS. `D`, `TAG` and `RJ` reached the gate only through
# `--export`, so an sbatch `--wrap` without that clause died on bash's own
# `${D:?...}` before the gate printed a row, before the roots were parsed, and
# with a message naming the variable rather than the fix. Two independent lanes
# hit it on one night -- B-cert-4's cleanup 5841112 (C31-2) and p6ad-4's 5879244,
# which stranded certP6AD4-5879243-{A,B} and ws.P6AD4-5879243-{A,B}. The three
# values are already carried by the root arguments, so they are now derived.
#
# NOTHING IS EVER DELETED BY THIS GUARD. Every arm points the gate at roots that
# DO NOT EXIST, so the run stops at the evidence or containment conditions long
# before `cleanup.sh` is reached. The guard asserts on what the gate PRINTS and
# on the values it derives, never on a deletion.
#
# ARMS
#   A  no D/TAG/RJ at all, over a fixture harness whose artifacts/ holds the
#      evidence -> the gate DERIVES all three and prints them. This is the fix.
#   B  the OLD file, byte-for-byte from git, same arguments -> dies on the bash
#      parameter expansion with no `### DERIVED` row anywhere. THE MUTATION:
#      without it arm A proves nothing, because a gate that always printed
#      `### DERIVED` would pass arm A for free.
#   C  explicit D/TAG/RJ WIN over the derivation: a TAG the roots do not name is
#      honoured, so the fix is additive and cannot override an operator.
#   D  underivable -- a root basename with no job-id token -> exit 2 and the
#      printed `--export=ALL,D=...` line, never a bash error.
#   E  derivable tag+job but NO harness directory holds the evidence -> exit 2
#      with the export line, and D reported unset.
#   F  SWEEP-3-1: the harness is nested TWO deep under the task root
#      (`<T>/c2-merged/a/artifacts/`), the shape every merge lane writes -> the
#      gate still derives D and still runs its conditions.
#   FR SWEEP-3-1 MUTATION: the SAME nested fixture against the PINNED previous
#      file ($SWEEP3_OLD), whose `derive_harness_dir` searched at `-maxdepth 3`
#      -> D is never derived and the gate refuses with exit 2. This is the defect
#      that refused four lanes whose evidence was complete.
#   G  TWO harness directories hold the same TAG-RJ evidence -> the gate refuses
#      with exit 2 and NAMES BOTH candidates. Widening the depth widens what can
#      collide, so the uniqueness rule is now guarded, not merely retained.
#
# MUTATION ARMS ARE PINNED TO COMMIT CONSTANTS, NEVER TO `HEAD`. Arm B used to
# read `HEAD:harness/phase_template/cleanup_gated.sh`; the moment d2ba3fd
# committed the p6ad-4 fix, HEAD carried the FIXED file and the arm began
# asserting the fix against itself. It was signed off at 15/15 and measured at
# 13/2 (job 5891315, node2315) with two arm-B failures and nothing else changed.
# A mutation arm must name the commit that carries the defect.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
# WHERE THE REPO IS. `$HERE/../..` is the repo when this guard runs from the
# harness worktree -- but the task-dir copy the drift check syncs lives at
# `<task>/tools/`, where `$HERE/../..` is `/oscar/data/stellex/glvov/agrescap/tasks`
# and every arm died `FATAL: no gate at .../tasks/harness/...`. A synced copy that
# can only ever FATAL is a file with no reader. `HARNESS_REPO` wins (same variable
# harness_drift_check.sh uses for the same thing); otherwise the relative guess is
# TESTED and the campaign worktree is the fallback. The guard always exercises the
# VERSIONED file -- the drift check is what proves the task copy equals it.
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
if [ ! -d "$REPO/harness/tools" ] || ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  REPO=/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
fi
[ -d "$REPO/harness/tools" ] || { echo "FATAL: no harness repo at $REPO"; exit 3; }
GATE=$REPO/harness/phase_template/cleanup_gated.sh
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
WORK=${TMPDIR:-/tmp}/cleanup-gate-env-guard-$$
# The two PINNED mutation sources. Each names the last commit that carries the
# defect its arm reproduces; neither is `HEAD`, and neither may be changed to
# `HEAD` (see the note above the arm list).
DERIV_OLD=${DERIV_OLD:-ececead}   # d2ba3fd^ -- the gate before p6ad-4's derivation
SWEEP3_OLD=${SWEEP3_OLD:-efe74a0} # the gate while derive_harness_dir was -maxdepth 3
mkdir -p "$WORK"
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }

[ -f "$GATE" ] || { echo "FATAL: no gate at $GATE"; exit 3; }
echo "### guard for $GATE"
echo "### work $WORK"

# A fixture harness directory under the TASK ROOT, because `derive_harness_dir`
# searches there by construction. Its tag is unique to this run so it can never
# collide with a live lane's artifacts.
TAG=GUARDENVDERIV$$
RJ=999$$
HDIR=$T/guard-envderiv-$$   # ONE level under the task root -- the FLAT shape.
mk_evidence () {            # mk_evidence <harness dir> <tag> <rj>
  mkdir -p "$1/artifacts"
  printf '0\n'   > "$1/artifacts/$2-$3.rc"
  printf '123\n' > "$1/artifacts/$2-$3.wall"
  printf 'x\n'   > "$1/artifacts/$2-$3.lock.log"
  printf 'x\n'   > "$1/artifacts/$2-$3.pixi.lock.cert"
}
mk_evidence "$HDIR" "$TAG" "$RJ"
ROOT=/oscar/data/stellex/glvov/retread/cert$TAG-$RJ-A
[ -e "$ROOT" ] && { echo "FATAL: fixture root $ROOT exists on disk -- refusing to run"; exit 3; }

# SWEEP-3-1 fixture: the NESTED shape, `<T>/c2-merged/a/artifacts/`. This is not
# a hypothetical layout -- it is what a merge lane writes when a batch directory
# carries one subdirectory per candidate, and four lanes with complete evidence
# were refused by the gate because `-maxdepth 3` could not see one level further
# down. Its tag is distinct from the flat fixture's so the two can never collide
# and make arm G pass for the wrong reason.
NTAG=GUARDNESTED$$
NRJ=888$$
NBASE=$T/guard-nested-$$
NDIR=$NBASE/a
mk_evidence "$NDIR" "$NTAG" "$NRJ"
NROOT=/oscar/data/stellex/glvov/retread/cert$NTAG-$NRJ-A
[ -e "$NROOT" ] && { echo "FATAL: fixture root $NROOT exists on disk -- refusing to run"; exit 3; }

# ARM G fixture: the SAME tag+job written under TWO harness directories, one
# flat and one nested. Widening the depth widens what can collide, so uniqueness
# has to be asserted, not assumed.
DTAG=GUARDDUP$$
DRJ=777$$
DDIR1=$T/guard-dup1-$$
DDIR2=$T/guard-dup2-$$/a
mk_evidence "$DDIR1" "$DTAG" "$DRJ"
mk_evidence "$DDIR2" "$DTAG" "$DRJ"
DROOT=/oscar/data/stellex/glvov/retread/cert$DTAG-$DRJ-A
[ -e "$DROOT" ] && { echo "FATAL: fixture root $DROOT exists on disk -- refusing to run"; exit 3; }

cleanup_fixture() { rm -rf "$HDIR" "$NBASE" "$DDIR1" "$T/guard-dup2-$$" "$WORK"; }
trap cleanup_fixture EXIT

run_gate() {   # run_gate <logfile> <script> [VAR=VAL ...] -- <roots...>
  local log=$1 script=$2; shift 2
  local -a envs=()
  while [ "${1:-}" != "--" ]; do envs+=("$1"); shift; done
  shift
  ( env -u D -u TAG -u RJ "${envs[@]}" bash "$script" "$@" ) > "$log" 2>&1
  echo $?
}

# ---- ARM A: derivation, with nothing exported -------------------------------
rcA=$(run_gate "$WORK/A.log" "$GATE" -- "$ROOT")
grep -q "### DERIVED TAG=$TAG " "$WORK/A.log" \
  && ok "A: TAG derived from the root basename" || bad "A: TAG not derived"
grep -q "### DERIVED RJ=$RJ " "$WORK/A.log" \
  && ok "A: RJ derived from the root basename" || bad "A: RJ not derived"
grep -q "### DERIVED D=$HDIR " "$WORK/A.log" \
  && ok "A: D derived from the artifacts that hold the evidence" || bad "A: D not derived"
grep -q "### CLEANUP GATE tag=$TAG relock_job=$RJ" "$WORK/A.log" \
  && ok "A: the gate actually ran its conditions (rc=$rcA)" || bad "A: the gate never reached its conditions"
grep -qi "set D to the harness directory" "$WORK/A.log" \
  && bad "A: still died on the bash parameter expansion" || ok "A: no bash parameter-expansion death"

# ---- ARM B: THE MUTATION -- the previous version of the same file ------------
OLD=$WORK/cleanup_gated.OLD.sh
if git -C "$REPO" show "$DERIV_OLD:harness/phase_template/cleanup_gated.sh" > "$OLD" 2>/dev/null && [ -s "$OLD" ]; then
  rcB=$(run_gate "$WORK/B.log" "$OLD" -- "$ROOT")
  if grep -q "### DERIVED" "$WORK/B.log"; then
    bad "B: the OLD file already derived -- this guard cannot fail and is worthless"
  else
    ok "B: the OLD file prints no ### DERIVED row (the guard can fail)"
  fi
  if grep -qi "set D to the harness directory" "$WORK/B.log"; then
    ok "B: the OLD file dies on the bash parameter expansion, rc=$rcB -- the defect, reproduced"
  else
    bad "B: the OLD file did not reproduce the defect (rc=$rcB) -- read $WORK/B.log"
  fi
else
  bad "B: could not extract $DERIV_OLD:harness/phase_template/cleanup_gated.sh -- MUTATION ARM DID NOT RUN"
fi

# ---- ARM C: an explicit export WINS over the derivation ----------------------
rcC=$(run_gate "$WORK/C.log" "$GATE" "TAG=EXPLICITTAG" "RJ=$RJ" "D=$HDIR" -- "$ROOT")
grep -q "### CLEANUP GATE tag=EXPLICITTAG relock_job=$RJ" "$WORK/C.log" \
  && ok "C: an exported TAG overrides the derivation (rc=$rcC)" || bad "C: the derivation overrode an explicit export"
grep -q "### DERIVED" "$WORK/C.log" \
  && bad "C: derived something that was explicitly given" || ok "C: nothing derived when everything was given"

# ---- ARM D: underivable -- no job-id token in the basename -------------------
rcD=$(run_gate "$WORK/D.log" "$GATE" -- /oscar/data/stellex/glvov/retread/certNOJOBIDHERE)
[ "$rcD" = 2 ] && ok "D: an underivable root refuses with exit 2" || bad "D: exit $rcD, want 2"
grep -q -- "--export=D=" "$WORK/D.log" \
  && ok "D: the refusal prints the exact --export clause to add" || bad "D: the refusal does not print the export line"
# OWNER-EXPORT-1. The clause this refusal prints is the one an operator COPIES,
# so it is a submit path like any other: a printed `--export=ALL,...` is how the
# defect propagates into the next driver. `ALL` may appear in the PROSE that
# explains why it is banned, but never inside an `--export=` clause.
if grep -oE -- '--export=[^ "]*' "$WORK/D.log" | grep -q 'ALL'; then
  bad "D: the printed --export clause still carries ALL -- that clause is what held 6013350/6013351/6014485/5841188"
else
  ok "D: the printed --export clause carries NO ALL (OWNER-EXPORT-1)"
fi
for v in D= TAG= RJ= DRY_RUN= PATH= HOME=; do
  grep -oE -- '--export=[^ "]*' "$WORK/D.log" | grep -q -- "$v" \
    && ok "D: the printed clause names $v" || bad "D: the printed clause omits $v -- an owner submitted from it runs with half its contract"
done
grep -qi "set D to the harness directory" "$WORK/D.log" \
  && bad "D: refused with a bash parameter-expansion error" || ok "D: refused with a message, not a bash error"

# ---- ARM E: derivable tag+job, but no harness holds the evidence -------------
rcE=$(run_gate "$WORK/E.log" "$GATE" -- "/oscar/data/stellex/glvov/retread/certNOSUCHHARNESS$$-888$$-A")
[ "$rcE" = 2 ] && ok "E: no harness for the evidence refuses with exit 2" || bad "E: exit $rcE, want 2"
grep -q "D='<unset>'" "$WORK/E.log" \
  && ok "E: the refusal names D as the value it could not derive" || bad "E: the refusal does not name D"

# ---- ARM F: SWEEP-3-1 -- the harness is nested two deep ----------------------
rcF=$(run_gate "$WORK/F.log" "$GATE" -- "$NROOT")
grep -q "### DERIVED D=$NDIR " "$WORK/F.log" \
  && ok "F: D derived for a harness nested two deep ($NDIR)" || bad "F: D not derived for the nested harness"
grep -q "### CLEANUP GATE tag=$NTAG relock_job=$NRJ" "$WORK/F.log" \
  && ok "F: the gate ran its conditions on the nested harness (rc=$rcF)" || bad "F: the gate never reached its conditions"

# ---- ARM FR: THE SWEEP-3-1 MUTATION -- the same fixture, the -maxdepth 3 file --
OLD3=$WORK/cleanup_gated.DEPTH3.sh
if git -C "$REPO" show "$SWEEP3_OLD:harness/phase_template/cleanup_gated.sh" > "$OLD3" 2>/dev/null && [ -s "$OLD3" ]; then
  grep -q -- "-maxdepth 3 -type f -path" "$OLD3" \
    && ok "FR: the pinned $SWEEP3_OLD file really does search at -maxdepth 3" \
    || bad "FR: $SWEEP3_OLD does not carry the -maxdepth 3 search -- WRONG PIN, the mutation is not the defect"
  rcFR=$(run_gate "$WORK/FR.log" "$OLD3" -- "$NROOT")
  grep -q "### DERIVED D=" "$WORK/FR.log" \
    && bad "FR: the -maxdepth 3 file derived D for a nested harness -- this arm cannot fail" \
    || ok "FR: the -maxdepth 3 file never derives D for a nested harness (the guard can fail)"
  [ "$rcFR" = 2 ] \
    && ok "FR: it refuses with exit 2 -- the defect that stranded four lanes, reproduced" \
    || bad "FR: exit $rcFR, want 2 -- read $WORK/FR.log"
else
  bad "FR: could not extract $SWEEP3_OLD:harness/phase_template/cleanup_gated.sh -- MUTATION ARM DID NOT RUN"
fi

# ---- ARM G: two harnesses hold the same evidence -> refuse, naming both ------
rcG=$(run_gate "$WORK/G.log" "$GATE" -- "$DROOT")
[ "$rcG" = 2 ] && ok "G: two candidate harness dirs refuse with exit 2" || bad "G: exit $rcG, want 2"
grep -q "### AMBIGUOUS D: 2 directories" "$WORK/G.log" \
  && ok "G: the refusal says the derivation was ambiguous" || bad "G: the refusal does not say it was ambiguous"
grep -q "candidate: $DDIR1$" "$WORK/G.log" && grep -q "candidate: $DDIR2$" "$WORK/G.log" \
  && ok "G: it names BOTH candidates and picks neither" || bad "G: it did not name both candidates"
grep -q "### DERIVED D=" "$WORK/G.log" \
  && bad "G: it guessed a D from an ambiguous set" || ok "G: no D was guessed"

# ---- nothing was deleted ----------------------------------------------------
[ -s "$HDIR/artifacts/$TAG-$RJ.rc" ] && [ -s "$NDIR/artifacts/$NTAG-$NRJ.rc" ] \
  && [ -s "$DDIR1/artifacts/$DTAG-$DRJ.rc" ] && [ -s "$DDIR2/artifacts/$DTAG-$DRJ.rc" ] \
  && ok "no fixture evidence was removed by any arm" || bad "the guard deleted its own fixture"

echo "### cleanup_gate_env_derivation_guard: pass=$pass fail=$fail"
[ "$fail" -eq 0 ] || exit 1
exit 0
