#!/usr/bin/env bash
# env_seed_export_guard.sh -- the reader for tools/env_seed.sh's `env_seed_export`,
# and for the phaseN_relock.sh call site that is its production consumer.
#
# WHAT IT IS FOR. DET-1-4-1 (job 6001140) locked one manifest three times on one
# node with one binary and varied nothing but PYTHONHASHSEED in the LAUNCHING
# shell: the two seeded arms emitted a byte-identical gym `requires_dist` block
# (md5 e569ebf5..., cmp rc=0) and the arm with the seed absent from
# /proc/<pixi.real>/environ emitted a different ORDER of the same rows (28 raw /
# 0 sorted). The backend cannot fix this from inside -- the process that builds
# that block is an in-process PEP 517 child of the pixi FRONTEND, which the
# backend never execs -- so the wrapper's export is the only channel, and this
# guard is what keeps it from silently becoming a no-op.
#
# THE THREE FAILURES IT WOULD HAVE CAUGHT, and the third is the one that is not
# obvious:
#
#   A. the verb answers `0`      -> exported, and a CHILD PROCESS sees it. The
#                                   assertion is on the child, not on the row:
#                                   a wrapper that printed the row and forgot
#                                   the `export` would pass a row-only check and
#                                   still lock under a random seed.
#   B. the verb answers NOTHING  -> REFUSE, non-zero, and the refusal names an
#      or exits non-zero            actuator. CPython treats an empty
#                                   PYTHONHASHSEED as UNSET, i.e. random, so
#                                   exporting "" would look like a pin and
#                                   behave like the defect.
#   C. the binary does NOT carry  -> REFUSE, and refuse WITHOUT RUNNING IT.
#      the verb                     `main.rs` matches no verb, falls through to
#                                   the automatic preflight and starts the
#                                   JSON-RPC transport, so `<old bin> env-seed`
#                                   inside `$(...)` does not fail -- it BLOCKS.
#                                   This arm's stub never returns if executed,
#                                   and the arm is run under `timeout`, so a
#                                   wrapper that probed by running would show up
#                                   here as a HANG and not as a pass.
#
#   MUTATION. Cut `export PYTHONHASHSEED=$seed` out of the extracted function and
#   arm A must go RED -- the child stops seeing the seed. If A still passed, A
#   would be reading the wrapper's intent instead of its effect.
#
#   O1/O2 (DET-1-6-1). The function now has TWO modes, because the SMOKE calls it
#   too and every known-good control binary in the tree predates the verb. So
#   `optional` must (O1) let a verb-less binary through with a not-applicable row
#   and NOTHING exported -- and still not run it, so the anti-hang assertion
#   survives -- while (O2) STILL refusing a binary that carries the verb and then
#   answers empty. Arm C is O1's non-vacuity pair: the SAME stub under STRICT is
#   refused, so `optional` is a named mode and not a hole.
#   G. And the one-authority arm: the template must SOURCE the library rather
#   than carry its own copy of the function, which is the defect DET-1-6-1 fixed.
#
# The function is lifted VERBATIM out of the shipped LIBRARY (same awk extractor
# wheel_store_census_guard.sh uses), so this guard tests the code that runs, not
# a copy of it.
#
# WHY THE LIBRARY AND NOT THE TEMPLATE (DET-1-6-1). 58717bd defined
# `env_seed_export` inside `phaseN_relock.sh`, where only the relock arms reached
# it. `tools/proof_smoke.sh` launches its own `pixi lock` and did not, so
# det16-proof 6013332 refused its own FIX binary before arm 1 -- the backend log's
# first line was `preflight: $PYTHONHASHSEED must be `0` in the environment that
# launched pixi`, eight times in three seconds. The function moved to
# `tools/env_seed.sh` and both callers source it; this guard follows it there and
# arm E keeps checking the template's CALL SITE and its reader.
#
# Usage: env_seed_export_guard.sh [<template>] [<library>]
#        (self-contained, needs only $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TPL=${1:-$HERE/phaseN_relock.sh}
LIB=${2:-$HERE/../tools/env_seed.sh}
[ -f "$TPL" ] || { echo "GUARD FATAL: no template at $TPL"; exit 2; }
[ -f "$LIB" ] || { echo "GUARD FATAL: no env_seed.sh library at $LIB -- nothing defines the export"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/env-seed-export-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

extract () {  # $1=file $2=function name -> the function text, verbatim
  awk -v fn="$2" '$0 ~ "^"fn" \\(\\) \\{" {p=1} p {print} p && /^\}$/ {exit}' "$1"
}

FN=$(extract "$LIB" env_seed_export)
MARKER_LINE=$(grep -m1 '^ENV_SEED_MARKER=' "$LIB")
[ -n "$FN" ]          || { echo "GUARD FATAL: $LIB has no env_seed_export function -- nothing exports the seed"; exit 2; }
[ -n "$MARKER_LINE" ] || { echo "GUARD FATAL: $LIB has no ENV_SEED_MARKER -- the verb-absent check cannot be static"; exit 2; }

# The marker VALUE, as the template spells it. The stubs below embed it (or not)
# so the static detection is exercised on real bytes rather than on a flag.
MARKER=$(printf '%s\n' "$MARKER_LINE" | sed "s/^ENV_SEED_MARKER='//; s/'$//")
case $MARKER in *env-seed*) ok "the marker names the verb: '$MARKER'";;
  *) fail "ENV_SEED_MARKER does not mention env-seed: '$MARKER'";; esac

########## stub backends ######################################################
# A stub is a shell script, and `grep -a -F` over it finds the marker exactly as
# it finds it in an ELF binary -- the detection is a byte scan either way.
mk_stub () {   # $1=path  $2=carries-marker(1/0)  $3=body
  { echo '#!/usr/bin/env bash'
    [ "$2" = 1 ] && echo "# $MARKER"
    printf '%s\n' "$3"; } > "$1"
  chmod +x "$1"
}
mk_stub "$W/good"      1 'case ${1:-} in env-seed) printf 0 ;; preflight) exit 0 ;; *) exit 9 ;; esac'
mk_stub "$W/silent"    1 'case ${1:-} in env-seed) exit 0 ;; *) exit 9 ;; esac'
mk_stub "$W/broken"    1 'case ${1:-} in env-seed) echo "boom" >&2; exit 3 ;; *) exit 9 ;; esac'
# NO marker, and it NEVER RETURNS if run -- the observable shape of an old
# binary answering an unknown verb by starting its stdin transport.
mk_stub "$W/noverb"    0 'exec sleep 300'
mk_stub "$W/notexec"   1 'true'; chmod -x "$W/notexec"

########## the driver: the real function, plus a CHILD that reads the export ###
mk_driver () {   # $1=path  $2=function text
  { echo 'set -u'
    printf '%s\n' "$MARKER_LINE"
    printf '%s\n' "$2"
    echo 'env_seed_export "$1" "${2:-strict}"; rc=$?'
    # the CHILD. `env` is a separate process, so this is the export's effect and
    # not the assignment's appearance.
    echo 'echo "CHILD_SEES=$(env | sed -n "s/^PYTHONHASHSEED=//p")"'
    echo 'exit $rc'; } > "$1"
}
mk_driver "$W/drv.sh" "$FN"
MUT=$(printf '%s\n' "$FN" | grep -v '^  export PYTHONHASHSEED=\$seed$')
mk_driver "$W/drv.mut.sh" "$MUT"
if [ "$(printf '%s\n' "$FN" | wc -l)" -eq "$(printf '%s\n' "$MUT" | wc -l)" ]; then
  fail "the mutation removed NOTHING -- the export assignment is not on its own line any more, so the mutation arm is vacuous"
fi

run () {   # $1=driver $2=stub -> sets OUT, RC. `timeout` is the anti-hang assertion.
  OUT=$(timeout 20 bash "$1" "$2" "${3:-strict}" 2>&1); RC=$?
}
child_sees () { printf '%s\n' "$1" | sed -n 's/^CHILD_SEES=//p' | head -1; }

########## A. the verb answers 0 ##############################################
run "$W/drv.sh" "$W/good"
if [ "$RC" = 0 ] \
   && printf '%s\n' "$OUT" | grep -q "^### ENV SEED exported PYTHONHASHSEED=0 source=$W/good env-seed$" \
   && [ "$(child_sees "$OUT")" = 0 ]; then
  ok "A: verb prints 0 -> rc 0, the row is greppable verbatim, and a CHILD PROCESS sees PYTHONHASHSEED=0"
else
  fail "A: rc=$RC child_sees='$(child_sees "$OUT")'; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

########## A-mut. the export cut -> A must go RED #############################
run "$W/drv.mut.sh" "$W/good"
if [ -z "$(child_sees "$OUT")" ]; then
  ok "A-mut: with the export assignment removed the child sees NOTHING -- arm A is reading the effect, not the row"
else
  fail "A-mut: the child still sees '$(child_sees "$OUT")' after the export was cut -- arm A is vacuous"
fi

########## B1. the verb prints nothing ########################################
run "$W/drv.sh" "$W/silent"
if [ "$RC" != 0 ] && [ -z "$(child_sees "$OUT")" ] \
   && printf '%s\n' "$OUT" | grep -q 'FATAL ENV SEED' \
   && printf '%s\n' "$OUT" | grep -q 'ACTUATOR:' \
   && printf '%s\n' "$OUT" | grep -q 'CPython'; then
  ok "B1: verb prints nothing -> rc=$RC, nothing exported, and the refusal names an ACTUATOR and says why empty != pinned"
else
  fail "B1: rc=$RC child_sees='$(child_sees "$OUT")'; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

########## B2. the verb exits non-zero ########################################
run "$W/drv.sh" "$W/broken"
if [ "$RC" != 0 ] && [ -z "$(child_sees "$OUT")" ] \
   && printf '%s\n' "$OUT" | grep -q 'FATAL ENV SEED' \
   && printf '%s\n' "$OUT" | grep -q 'ACTUATOR:'; then
  ok "B2: verb exits non-zero -> rc=$RC, nothing exported, refusal names an ACTUATOR"
else
  fail "B2: rc=$RC child_sees='$(child_sees "$OUT")'; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

########## C. a binary WITHOUT the verb: refuse, and do NOT hang ##############
S=$(date +%s)
run "$W/drv.sh" "$W/noverb"
E=$(( $(date +%s) - S ))
if [ "$RC" = 124 ]; then
  fail "C: the wrapper HUNG on a verb-less binary (timeout at ${E}s) -- it probed by running instead of by the static marker"
elif [ "$RC" != 0 ] && [ -z "$(child_sees "$OUT")" ] \
     && printf '%s\n' "$OUT" | grep -q 'does NOT carry' \
     && printf '%s\n' "$OUT" | grep -q 'ACTUATOR:' && [ "$E" -lt 15 ]; then
  ok "C: a verb-less binary is REFUSED statically in ${E}s (rc=$RC), never executed -- the stub would never have returned"
else
  fail "C: rc=$RC elapsed=${E}s child_sees='$(child_sees "$OUT")'; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

########## C-neg. the non-vacuity control for C ###############################
# C would pass for the wrong reason if the refusal fired on something other than
# the marker. Same stub body, marker ADDED: it must now get past the static gate
# and be RUN -- which, with `exec sleep 300`, is the hang. So the marker is what
# decides, and C is not an accident of the stub's shape.
mk_stub "$W/noverb-marked" 1 'exec sleep 300'
S=$(date +%s); run "$W/drv.sh" "$W/noverb-marked"; E=$(( $(date +%s) - S ))
if [ "$RC" = 124 ]; then
  ok "C-neg: the SAME stub carrying the marker is executed and hangs (timeout at ${E}s) -- the marker, not the stub's shape, is what C refuses on"
else
  fail "C-neg: marker-carrying stub returned rc=$RC in ${E}s -- C's refusal is not keyed on the marker"
fi

########## D. a backend that is not executable ################################
run "$W/drv.sh" "$W/notexec"
if [ "$RC" != 0 ] && printf '%s\n' "$OUT" | grep -q 'not an executable binary'; then
  ok "D: a non-executable backend is refused before anything else"
else
  fail "D: rc=$RC; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

########## O1. OPTIONAL mode: a verb-less binary passes, unseeded, and SAYS SO #
# DET-1-6-1. The smoke calls the function in this mode because every known-good
# control binary in the tree predates the verb. The stub is the SAME `exec sleep
# 300` arm C uses, so this arm is also still an anti-hang assertion: `optional`
# must relax the VERDICT, never the rule that a verb-less binary is not run.
S=$(date +%s)
run "$W/drv.sh" "$W/noverb" optional
E=$(( $(date +%s) - S ))
if [ "$RC" = 124 ]; then
  fail "O1: optional mode HUNG on a verb-less binary (timeout at ${E}s) -- it probed by running instead of by the static marker"
elif [ "$RC" = 0 ] && [ -z "$(child_sees "$OUT")" ] \
     && printf '%s\n' "$OUT" | grep -q '^### ENV SEED not-applicable binary lacks verb' \
     && [ "$E" -lt 15 ]; then
  ok "O1: optional mode PASSES a verb-less binary in ${E}s (rc=0), exports NOTHING, prints the not-applicable row, and still never runs it"
else
  fail "O1: rc=$RC elapsed=${E}s child_sees='$(child_sees "$OUT")'; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

########## O2. OPTIONAL is a named case, not a hole ###########################
# The one case `optional` relaxes is the verb's ABSENCE. A binary that CARRIES
# the verb and then answers empty is a defect in the binary, and an empty
# PYTHONHASHSEED is random, so it must be refused in BOTH modes -- otherwise
# `optional` would be "never refuse", and the smoke would launch pixi under a
# random seed while printing a row that looked like a pin.
run "$W/drv.sh" "$W/silent" optional
if [ "$RC" != 0 ] && [ -z "$(child_sees "$OUT")" ] \
   && printf '%s\n' "$OUT" | grep -q 'FATAL ENV SEED'; then
  ok "O2: optional mode STILL refuses a marker-carrying binary whose verb prints nothing (rc=$RC) -- it relaxes absence, not badness"
else
  fail "O2: rc=$RC child_sees='$(child_sees "$OUT")'; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

########## G. ONE AUTHORITY: the template SOURCES, it does not re-define ######
# The DET-1-6-1 defect in one line. While the function lived only in the
# template, `tools/proof_smoke.sh` launched pixi without it and refused the
# template's own FIX binary before arm 1. A template that grew its own copy back
# would restore exactly that split, and both copies would look right.
if grep -q '^\. "\$ENV_SEED_LIB"$' "$TPL" \
   && ! grep -q '^env_seed_export () {' "$TPL" \
   && ! grep -q '^ENV_SEED_MARKER=' "$TPL"; then
  ok "G: the template SOURCES $LIB and defines neither the function nor the marker itself -- one authority"
else
  fail "G: the template does not source the library, or has grown its own copy of env_seed_export/ENV_SEED_MARKER (the DET-1-6-1 split)"
fi

########## E. the export has a READER in the template #########################
# Law 2: a wrapper that exported a variable and never checked that anything
# consumes it is a stamped-but-unread directive. The reader is the backend's own
# `preflight`, called right after the export.
if grep -q '^env_seed_export "\$BACKEND" || exit 15$' "$TPL" \
   && grep -q '^if ! "\$BACKEND" preflight; then$' "$TPL"; then
  ok "E: the template calls env_seed_export and then \`\$BACKEND preflight\` -- the export has a reader"
else
  fail "E: the template does not call env_seed_export followed by \`\$BACKEND preflight\`"
fi

########## F. the seed reaches the evidence packet ############################
if grep -q "^env | grep -E '\^(HOME|PIXI_|PYTHONHASHSEED|" "$TPL"; then
  ok "F: PYTHONHASHSEED is in the template's env census, so the exported seed lands in the evidence packet"
else
  fail "F: PYTHONHASHSEED is not in the template's env census line"
fi


# ---- ARM S: THE BLOCK IS IN ALL SIX SHIPPED RELOCK TEMPLATES ----------------
# HARNESS-CONSOL-12 (2026-09-07). Arms A-E above measure the FUNCTION and ONE
# call site -- `$TPL`, which defaults to phaseN_relock.sh. That is how the
# capability came to be 1/6: measured at 8f1dd88, across the six shipped relock
# templates, `retread_relock_frontend_log` appeared in 6/6 and
# `retread_relock_scope_and_verify` in 6/6, while `env_seed_export` appeared in
# ONE -- phaseN, this guard's default argument. `git cat-file blob
# 8f1dd88:harness/arms/mh1_relock.sh | grep -c seed` returned 0. The two
# capabilities that had a READER (tools/relock_capability_check.sh) travelled;
# the one with no reader did not, and nothing failed while it did not. Law 2
# from the reader's side, and this arm is the reader.
#
# WHAT IS ASSERTED IS THE PORTABLE HALF ONLY. phaseN additionally runs
# `"$BACKEND" preflight` and an `env | grep` row; those are phaseN's, they are
# arms E2/E3 above, and this arm does not demand them of every template. What
# every relock template must carry is: the ONE authority SOURCED, the export
# called STRICT over its backend, and a REFUSAL when it fails.
SEED_TARGETS=
for t in "$HERE/phaseN_relock.sh" "$HERE/../arms/mh1_relock.sh" \
         "$HERE/../arms/c29_relock.sh" "$HERE/../proof/hlgd_relock.sh" \
         "$HERE/../instrumented/p6b_relock.sh" "$HERE/../instrumented/p6b_relock.b2.sh"; do
  [ -f "$t" ] && SEED_TARGETS="$SEED_TARGETS $t"
done
SEED_N=$(printf '%s\n' $SEED_TARGETS | grep -c .)
if [ "$SEED_N" != 6 ]; then
  fail "S: found $SEED_N of the 6 shipped relock templates -- this arm would be green over whatever happened to be present"
else
  ok "S: 6 shipped relock templates found"
  for t in $SEED_TARGETS; do
    tn=$(basename "$t")
    if ! grep -qE '^[[:space:]]*\.[[:space:]]+"\$ENV_SEED_LIB"$' "$t"; then
      fail "S: $tn does not SOURCE \$ENV_SEED_LIB -- tools/env_seed.sh is the one authority (DET-1-6-1) and a wrapper that does not source it either has no seed or has a second one"
    elif ! grep -qE '^[[:space:]]*env_seed_export "\$BACKEND"[[:space:]]*\|\|[[:space:]]*exit 15$' "$t"; then
      fail "S: $tn does not call \`env_seed_export \"\$BACKEND\" || exit 15\` -- STRICT, over its backend, refusing on failure. $(grep -nE '^[[:space:]]*env_seed_export' "$t" | head -1 | sed 's/^/found: /')"
    elif grep -q '^env_seed_export () {' "$t" || grep -q '^ENV_SEED_MARKER=' "$t"; then
      fail "S: $tn carries its OWN copy of the function or the marker -- that is 58717bd's defect growing back, and det16-proof 6013332 is what it cost"
    elif grep -qE '^[[:space:]]*(export[[:space:]]+)?PYTHONHASHSEED=[0-9]' "$t"; then
      fail "S: $tn ASSIGNS a literal PYTHONHASHSEED -- a second authority that drifts from the backend's constant; the value is asked of the binary"
    else
      ok "S: $tn sources the one authority and calls env_seed_export \"\$BACKEND\" || exit 15 (STRICT, refusing)"
    fi
  done
  # THE MUTATION, so arm S can fail. mh1 is the template the block was
  # back-ported INTO; cut it back out of a COPY and S's own check must reject it.
  MH1F=$HERE/../arms/mh1_relock.sh
  if [ -f "$MH1F" ]; then
    MH1FCUT=$W/mh1_F_seedcut.sh
    sed -e '/^env_seed_export "\$BACKEND" || exit 15$/d' -e '/^\. "\$ENV_SEED_LIB"$/d' "$MH1F" > "$MH1FCUT"
    if cmp -s "$MH1F" "$MH1FCUT"; then
      fail "S-mut: the mutation removed NOTHING from mh1_relock.sh -- arm S is asserting against an unmutated file and cannot fail"
    elif grep -qE '^[[:space:]]*\.[[:space:]]+"\$ENV_SEED_LIB"$' "$MH1FCUT" \
      || grep -qE '^[[:space:]]*env_seed_export "\$BACKEND"[[:space:]]*\|\|[[:space:]]*exit 15$' "$MH1FCUT"; then
      fail "S-mut: the cut copy still satisfies arm S's check -- the check is not reading what it claims to read"
    else
      ok "S-mut: with the back-ported block cut back out, mh1_relock.sh FAILS arm S's check -- F can fail, and this is the 1/6 state 8f1dd88 shipped"
    fi
  fi
fi
echo "### env_seed_export_guard: $( [ "$FAIL" = 0 ] && echo PASS || echo FAIL )"
exit "$FAIL"
