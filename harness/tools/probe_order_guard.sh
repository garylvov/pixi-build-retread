#!/usr/bin/env bash
# probe_order_guard.sh -- MERGE-V-2-2. THE BACKEND VERSION PROBE MUST RUN AFTER
# THE SEED IS EXPORTED, in every shipped relock template.
#
#   usage: probe_order_guard.sh        (HARNESS_REPO=<repo> to point it)
#   Self-contained: fixtures only, no cluster resource, no real backend.
#
# ── WHAT WENT WRONG, AND WHY IT LOOKED LIKE NOTHING ─────────────────────────
# `ls -l "$SNAP"; "$SNAP" --version` sat in the snapshot gate, ~580 lines above
# the `env_seed_export "$BACKEND"` that exports PYTHONHASHSEED. A backend at or
# after fix/det1-env-seed carries a preflight that REFUSES an unset seed, so
# every relock log opened with that refusal around line 21. Ungated -- the
# probe's rc is discarded by the pipe -- and harmless, and therefore never
# fixed; but the first thing a reader saw in a HEALTHY log was a FATAL-shaped
# line about the one variable this harness exists to pin, which is how a
# diagnostician spends an hour on a run that was fine.
#
# ── THE ARMS ────────────────────────────────────────────────────────────────
#   A  STATIC, EVERY TEMPLATE: the `--version` probe's line number is GREATER
#      than the `env_seed_export` line's, in all of them. A file with one and
#      not the other is reported, not skipped.
#   B  EXECUTED, ON THE TEMPLATE'S OWN TWO LINES: the probe line and the export
#      line are CUT from the real template (never retyped) and run in file order
#      against a verb-carrying stub backend whose `--version` refuses an unset
#      seed. The refusal must not appear at all, and the ENV SEED row must.
#   C  MUTATION: the same two cut lines in the OLD order. The refusal must
#      appear, and BEFORE the ENV SEED row -- reproducing the log everyone read.
#      Without C, arm B only says a script ran.
#   D  NON-VACUITY on the stub: with PYTHONHASHSEED forced out of the
#      environment, the stub's `--version` really does refuse, so C is measuring
#      the preflight and not a typo.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
H=$REPO/harness
[ -d "$H/tools" ] || { echo "GUARD FATAL: no harness tree at $H"; exit 2; }
W=$(mktemp -d "${TMPDIR:-/tmp}/probeorder.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
pass=0; fail=0
ok()  { echo "GUARD  ok : $*"; pass=$((pass+1)); }
bad() { echo "GUARD FAIL: $*"; fail=$((fail+1)); }

TEMPLATES="arms/mh1_relock.sh arms/c29_relock.sh instrumented/p6b_relock.sh instrumented/p6b_relock.b2.sh proof/hlgd_relock.sh phase_template/phaseN_relock.sh"

########## A. STATIC, EVERY TEMPLATE ##########################################
echo "GUARD: === A: the probe runs after the export, in every shipped template ==="
NT=0
for rel in $TEMPLATES; do
  f=$H/$rel
  [ -f "$f" ] || { bad "A: no template at $rel"; continue; }
  NT=$((NT+1))
  # ANCHORED AT LINE START, and that is not cosmetic: the comment this fix added
  # QUOTES the probe, so an unanchored grep found the explanation and measured
  # its line number instead of the code's. Measured on job 6023496: arm B cut
  # `# FIX. \`"$SNAP" --version\` used to run beside the \`ls -l\`...` and ran a
  # comment. A scan that can match prose about the code is the same class of
  # defect as a scan that can match itself (CLAUDE.md law 14).
  PL=$(grep -n '^[[:space:]]*"\$SNAP" --version' "$f" | head -1 | cut -d: -f1)
  SL=$(grep -n '^[[:space:]]*env_seed_export "\$BACKEND"' "$f" | head -1 | cut -d: -f1)
  if [ -z "$PL" ] || [ -z "$SL" ]; then
    bad "A: $rel has probe=${PL:-none} export=${SL:-none} -- a template with one and not the other cannot be ordered, and is not silently skipped"
    continue
  fi
  if [ "$PL" -gt "$SL" ]; then
    ok "A: $rel probe at $PL runs AFTER the export at $SL"
  else
    bad "A: $rel probes the backend at line $PL, $((SL-PL)) lines BEFORE the seed export at $SL -- its log opens with the preflight refusal"
  fi
  # And the old combined form must not come back in any of them.
  grep -q 'ls -l "\$SNAP"; "\$SNAP" --version' "$f" \
    && bad "A: $rel still carries the combined \`ls -l; --version\` line the probe was cut out of" \
    || ok "A: $rel no longer carries the combined ls/probe line"
done
[ "$NT" -ge 6 ] && ok "A read $NT templates (not green against nothing)" \
                || bad "A found only $NT templates -- expected 6"

########## the stub backend: carries the verb, refuses an unset seed ##########
# The marker string is the one env_seed.sh greps for, so this stub is a
# verb-carrying backend by exactly the test the real code applies.
STUB=$W/backend
cat > "$STUB" <<'EOS'
#!/usr/bin/env bash
# retread env-seed: writing stdout: (marker, so env_seed.sh sees the verb)
case "${1:-}" in
  env-seed) echo 424242; exit 0;;
  --version)
    if [ -z "${PYTHONHASHSEED:-}" ]; then
      echo "FATAL preflight: PYTHONHASHSEED is not set; refusing to start" >&2
      exit 1
    fi
    echo "retread 0.0-fixture (PYTHONHASHSEED=$PYTHONHASHSEED)"; exit 0;;
esac
echo "fixture backend: unknown verb ${1:-}"; exit 3
EOS
chmod +x "$STUB"

########## D. NON-VACUITY on the stub #########################################
DOUT=$(env -u PYTHONHASHSEED "$STUB" --version 2>&1); DRC=$?
if [ "$DRC" != 0 ] && printf '%s\n' "$DOUT" | grep -q 'PYTHONHASHSEED is not set'; then
  ok "D: the stub's --version really refuses an unset seed (rc=$DRC), so C measures a preflight"
else
  bad "D: the stub did not refuse an unset seed (rc=$DRC) -- arms B and C measure nothing"
fi

########## B and C: the template's OWN two lines, both orders #################
SRC=$H/arms/mh1_relock.sh
[ -f "$SRC" ] || SRC=$H/phase_template/phaseN_relock.sh
PROBE=$(grep -m1 '^[[:space:]]*"\$SNAP" --version' "$SRC" | sed 's/^[[:space:]]*//')
EXPORTL=$(grep -m1 '^[[:space:]]*env_seed_export "\$BACKEND"' "$SRC" | sed 's/^[[:space:]]*//')
if [ -z "$PROBE" ] || [ -z "$EXPORTL" ]; then
  bad "B/C: could not cut both lines out of $SRC -- the executed arms did not run"
else
  ok "B/C: both lines were CUT from $(basename "$SRC"), not retyped: '$PROBE' and '$EXPORTL'"
  mkorder () {   # $1 = out, $2 = first line, $3 = second line
    { echo 'set -uo pipefail'
      echo "SNAP=$STUB"
      echo "BACKEND=\$SNAP"
      echo ". $H/tools/env_seed.sh"
      printf '%s\n' "$2"
      printf '%s\n' "$3"
    } > "$1"
  }
  # B: file order -- export first, probe second.
  mkorder "$W/fixed.sh" "$EXPORTL" "$PROBE"
  BOUT=$(env -u PYTHONHASHSEED bash "$W/fixed.sh" 2>&1); BRC=$?
  if [ "$BRC" = 0 ] \
     && ! printf '%s\n' "$BOUT" | grep -q 'PYTHONHASHSEED is not set' \
     && printf '%s\n' "$BOUT" | grep -q '### ENV SEED exported PYTHONHASHSEED='; then
    ok "B: with the export first the probe prints no refusal at all, and the ENV SEED row is there (rc=$BRC)"
  else
    bad "B: rc=$BRC; out: $(printf '%s' "$BOUT" | tr '\n' '|')"
  fi
  # C: THE MUTATION -- the old order.
  mkorder "$W/old.sh" "$PROBE" "$EXPORTL"
  COUT=$(env -u PYTHONHASHSEED bash "$W/old.sh" 2>&1)
  RL=$(printf '%s\n' "$COUT" | grep -n 'PYTHONHASHSEED is not set' | head -1 | cut -d: -f1)
  SL2=$(printf '%s\n' "$COUT" | grep -n '### ENV SEED exported' | head -1 | cut -d: -f1)
  if [ -n "$RL" ] && [ -n "$SL2" ] && [ "$RL" -lt "$SL2" ]; then
    ok "C: MUTATION -- in the OLD order the preflight refusal prints on line $RL, BEFORE the ENV SEED row on line $SL2, exactly the log everyone read"
  else
    bad "C: the old order did not reproduce the misleading refusal (refusal=${RL:-none} seedrow=${SL2:-none}) -- arm B proves nothing"
  fi
fi

echo "### probe_order_guard: pass=$pass fail=$fail -- $( [ "$fail" = 0 ] && echo PASS || echo FAIL )"
[ "$fail" = 0 ]
