#!/usr/bin/env bash
# env_seed_export_guard.sh -- the reader for phaseN_relock.sh's `env_seed_export`.
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
# The function is lifted VERBATIM out of the shipped template (same awk extractor
# wheel_store_census_guard.sh uses), so this guard tests the code that runs, not
# a copy of it.
#
# Usage: env_seed_export_guard.sh [<template>]   (self-contained, needs only $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TPL=${1:-$HERE/phaseN_relock.sh}
[ -f "$TPL" ] || { echo "GUARD FATAL: no template at $TPL"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/env-seed-export-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

extract () {  # $1=file $2=function name -> the function text, verbatim
  awk -v fn="$2" '$0 ~ "^"fn" \\(\\) \\{" {p=1} p {print} p && /^\}$/ {exit}' "$1"
}

FN=$(extract "$TPL" env_seed_export)
MARKER_LINE=$(grep -m1 '^ENV_SEED_MARKER=' "$TPL")
[ -n "$FN" ]          || { echo "GUARD FATAL: $TPL has no env_seed_export function -- nothing exports the seed"; exit 2; }
[ -n "$MARKER_LINE" ] || { echo "GUARD FATAL: $TPL has no ENV_SEED_MARKER -- the verb-absent check cannot be static"; exit 2; }

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
    echo 'env_seed_export "$1"; rc=$?'
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
  OUT=$(timeout 20 bash "$1" "$2" 2>&1); RC=$?
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

echo "### env_seed_export_guard: $( [ "$FAIL" = 0 ] && echo PASS || echo FAIL )"
exit "$FAIL"
