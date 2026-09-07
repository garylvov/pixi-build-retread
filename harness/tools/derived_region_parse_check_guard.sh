#!/usr/bin/env bash
# derived_region_parse_check_guard.sh -- the reader for
# tools/derived_region_parse_check.sh.
#
#   usage: derived_region_parse_check_guard.sh [<check>] [<template>]
#   Self-contained: fixtures only, no cluster resource, no binary.
#
# THE ARMS, and D is the one that makes the rest mean anything:
#   A  the SHIPPED template passes -- the check does not just always refuse
#   B  a fixture whose region carries an apostrophe inside `${2:?...}` is
#      REFUSED rc 3, and the refusal NAMES THE OPENING LINE and an actuator
#   C  NON-VACUITY: the SAME fixture with the apostrophe removed PASSES, so B
#      fires on the quote and not on the fixture's shape
#   D  THE CONTROL THAT EXPLAINS WHY THIS TOOL EXISTS: `bash -n` on B's fixture
#      IN FULL returns 0. The defect is invisible to the gate every driver
#      already ran, and visible to this one.
#   E  a file with no end marker is REFUSED rc 4, loudly -- a check that passes
#      a file it cannot inspect is decorative
#   F  an unbalanced DOUBLE quote is caught too, so the check is about the
#      shell's quoting and not about apostrophes
#   G  DET-1-6-3b: a REAL copy of the template whose SUBSTITUTE region carries
#      the dead `../retread_fast_env.sh` hop is REFUSED rc 5 -- with G1, the
#      control that the stripped diff the drivers run sees 0 lines on it
#   H  NON-VACUITY: the same copy with the TIP's hop passes rc 0, hop=ok
#   I  MUTATION: with the hop block deleted from a copy of the check, the SAME
#      stale fixture passes -- so G is a refusal, not a constant
#   J  an absent tip template is rc 4 FATAL, never a silent pass
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHK=${1:-$HERE/derived_region_parse_check.sh}
TPL=${2:-$HERE/../phase_template/phaseN_relock.sh}
[ -f "$CHK" ] || { echo "GUARD FATAL: no check at $CHK"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/derived-region-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

# The fixture is the SHAPE of a derived wrapper: a region between the markers,
# and ordinary script below it that must not be swallowed.
mk_fixture () {   # $1=path  $2=the region's assignment line
  cat > "$1" <<FIXEOF
#!/usr/bin/env bash
set -u
### SUBSTITUTE: BEGIN -- edit ONLY between these markers
TAG=\${1:?arg 1: the arm tag}
$2
### SUBSTITUTE: END
# a comment mentioning --export=ALL, whose backticked \`ALL\` must stay inert
echo "### fixture reached the body with D=\$D"
echo 'left|^Name' | head -1
# THE RE-BALANCING APOSTROPHE, AND IT IS WHAT MAKES ARM D POSSIBLE. In the real
# wrapper the stray quote opened at the region and CLOSED 109 lines later on the
# next apostrophe in the file, so the file as a whole parsed and \`bash -n\` was
# green. A fixture with no later apostrophe would fail full-file parse and arm D
# would be measuring a fixture that does not reproduce 6014197.
# closes the swallowed quote AND the brace, exactly as the real file did: dont' }
FIXEOF
}

# The fixtures above carry NO FAST_ENV hop, so they are compared against a tip
# that carries none either -- otherwise every one of them would be refused rc 5
# by the hop check for a reason that has nothing to do with the arm. The hop
# arms (G/H/I) use the REAL template and REAL derived copies of it.
EM='### SUBSTITUTE: END'
mk_fixture "$W/tpl_nohop.sh" 'D=${2:-a job root}'
NOHOP=$W/tpl_nohop.sh

########## A. the shipped template passes #####################################
if [ -f "$TPL" ]; then
  OUT=$(bash "$CHK" "$TPL" 2>&1); RC=$?
  if [ "$RC" = 0 ] && printf '%s\n' "$OUT" | grep -q '^### DERIVED REGION PARSE: clean'; then
    ok "A: the shipped template $TPL passes (rc 0) -- the check does not always refuse"
  else
    fail "A: the shipped template was refused rc=$RC; out: $(printf '%s' "$OUT" | tr '\n' '|')"
  fi
else
  fail "A: no template at $TPL to check against"
fi

########## B. the 6014197 defect, reproduced ##################################
mk_fixture "$W/bad.sh" 'D=${2:?arg 2: this arm'"'"'s job root, which must hold HARNESS_COMMIT}'
OUT=$(bash "$CHK" "$W/bad.sh" "$EM" "$NOHOP" 2>&1); RC=$?
if [ "$RC" = 3 ] \
   && printf '%s\n' "$OUT" | grep -q 'DERIVED REGION PARSE REFUSED' \
   && printf '%s\n' "$OUT" | grep -q 'line 5' \
   && printf '%s\n' "$OUT" | grep -q 'ACTUATOR:'; then
  ok "B: an apostrophe inside \${2:?...} in the region is REFUSED rc 3, naming line 5 and an ACTUATOR"
else
  fail "B: rc=$RC; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

########## C. NON-VACUITY: the same fixture, apostrophe removed ###############
mk_fixture "$W/good.sh" 'D=${2:?arg 2: this arm job root, which must hold HARNESS_COMMIT}'
OUT=$(bash "$CHK" "$W/good.sh" "$EM" "$NOHOP" 2>&1); RC=$?
if [ "$RC" = 0 ]; then
  ok "C: the SAME fixture with the apostrophe removed PASSES -- B fires on the quote, not on the shape"
else
  fail "C: rc=$RC on the corrected fixture; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

########## D. why this tool exists: full-file `bash -n` is GREEN on B #########
if bash -n "$W/bad.sh" 2>/dev/null; then
  ok "D: \`bash -n\` on the WHOLE broken fixture returns 0 -- the gate every driver already ran cannot see this, and that is the case for the truncation"
else
  fail "D: full-file bash -n already refused the fixture: $(bash -n "$W/bad.sh" 2>&1 | tr "\n" "|")"
fi
# ... and the swallowing is REAL, not just a parse curiosity: the body line must
# NOT run. Without this the arm above only says "it parses".
BODY=$(bash "$W/bad.sh" A B 2>&1)
if ! printf '%s\n' "$BODY" | grep -q 'fixture reached the body'; then
  ok "D2: and running the broken fixture never reaches its body -- the region really did swallow the script below it"
else
  fail "D2: the broken fixture reached its body, so the fixture does not reproduce the defect"
fi

########## E. no end marker -> refuse, loudly #################################
printf '#!/usr/bin/env bash\necho hi\n' > "$W/nomark.sh"
OUT=$(bash "$CHK" "$W/nomark.sh" "$EM" "$NOHOP" 2>&1); RC=$?
if [ "$RC" = 4 ] && printf '%s\n' "$OUT" | grep -q 'has no line'; then
  ok "E: a file with no SUBSTITUTE end marker is REFUSED rc 4 -- it is not silently passed"
else
  fail "E: rc=$RC; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

########## F. it is about quoting, not about apostrophes ######################
mk_fixture "$W/dq.sh" 'D=${2:-a "dangling double quote}'
OUT=$(bash "$CHK" "$W/dq.sh" "$EM" "$NOHOP" 2>&1); RC=$?
if [ "$RC" = 3 ]; then
  ok "F: an unbalanced DOUBLE quote in the region is refused too -- the check is the shell's quoting rules, not a pattern for one character"
else
  fail "F: rc=$RC on an unbalanced double quote; out: $(printf '%s' "$OUT" | tr '\n' '|')"
fi


########## G/H/I. DET-1-6-3b: the FAST_ENV hop INSIDE the stripped region #####
# These three run on REAL files: the shipped template as the tip, and copies of
# it as the derived wrappers. A fixture that merely LOOKS like a wrapper would
# not prove the tip's own hop is what gets reproduced.
if [ -f "$TPL" ]; then
  cp "$TPL" "$W/drv_ok.sh"
  cp "$TPL" "$W/drv_stale.sh"
  # THE EXACT PRE-2026-09-07 HOP, put back where det163's wrapper carried it:
  # inside the SUBSTITUTE region, where the stripped diff cannot reach it.
  sed -i "0,/^FAST_ENV=/s|^FAST_ENV=.*|FAST_ENV=\$(dirname \"\$0\")/../retread_fast_env.sh|" "$W/drv_stale.sh"
  if grep -q 'dirname "\$0")/\.\./retread_fast_env\.sh' "$W/drv_stale.sh" \
     && ! cmp -s "$W/drv_stale.sh" "$TPL"; then
    ok "G0: the stale fixture really carries the dead ../retread_fast_env.sh hop and differs from the tip"
  else
    fail "G0: could not build the stale fixture from $TPL -- G/H measure nothing"
  fi
  # ... and the fixture is INVISIBLE to the gate that already runs: strip the
  # SUBSTITUTE region from both and the diff is empty. This is arm D's role for
  # the hop, and without it G proves only that a check refuses something.
  strip_region () { awk '/^### SUBSTITUTE: BEGIN/{s=1} /^### SUBSTITUTE: END/{s=0; next} s{next} {print}' "$1"; }
  NDIFF=$(diff <(strip_region "$TPL") <(strip_region "$W/drv_stale.sh") | grep -c '^[<>]' || true)
  if [ "$NDIFF" = 0 ]; then
    ok "G1: THE CONTROL -- the stripped diff the drivers run reports 0 diff lines on the stale wrapper, exactly as it did for job 6020526"
  else
    fail "G1: the stripped diff shows $NDIFF line(s) on the stale fixture, so it is not the invisible defect this arm is for"
  fi

  OUT=$(bash "$CHK" "$W/drv_stale.sh" "$EM" "$TPL" 2>&1); RC=$?
  if [ "$RC" = 5 ] \
     && printf '%s\n' "$OUT" | grep -q '^### DERIVED FAST_ENV hop=STALE tip=[0-9a-f]\{32\} derived=[0-9a-f]\{32\}$' \
     && printf '%s\n' "$OUT" | grep -q 'ACTUATOR:'; then
    ok "G: a derived wrapper carrying the dead hop is REFUSED rc 5 with hop=STALE and both md5s"
  else
    fail "G: rc=$RC (want 5); out: $(printf '%s' "$OUT" | tr '\n' '|')"
  fi

  OUT=$(bash "$CHK" "$W/drv_ok.sh" "$EM" "$TPL" 2>&1); RC=$?
  if [ "$RC" = 0 ] && printf '%s\n' "$OUT" | grep -q '^### DERIVED FAST_ENV hop=ok tip=[0-9a-f]\{32\} derived=[0-9a-f]\{32\}$'; then
    ok "H: NON-VACUITY -- the same wrapper with the TIP's hop passes rc 0 and prints hop=ok"
  else
    fail "H: rc=$RC on a wrapper that reproduces the tip verbatim; out: $(printf '%s' "$OUT" | tr '\n' '|')"
  fi

  # I. THE MUTATION. Delete the hop block from a COPY of the check and the stale
  # fixture must sail through. A guard that cannot be made to fail is decorative.
  awk '/^### FAST_ENV-HOP BEGIN/{m=1} !m{print} /^### FAST_ENV-HOP END/{m=0}' "$CHK" > "$W/mutant.sh"
  if grep -q 'FAST_ENV-HOP BEGIN' "$W/mutant.sh" || ! bash -n "$W/mutant.sh" 2>/dev/null; then
    fail "I: could not build the hop-removed mutant of $CHK"
  else
    OUT=$(bash "$W/mutant.sh" "$W/drv_stale.sh" "$EM" "$TPL" 2>&1); RC=$?
    if [ "$RC" = 0 ] && ! printf '%s\n' "$OUT" | grep -q 'FAST_ENV hop='; then
      ok "I: MUTATION -- with the hop check removed the SAME stale fixture passes rc 0, so G is a real refusal and not a constant"
    else
      fail "I: the mutant still refused (rc=$RC), so arm G proves nothing; out: $(printf '%s' "$OUT" | tr '\n' '|')"
    fi
  fi

  # J. a missing tip is FATAL, not a silent pass.
  OUT=$(bash "$CHK" "$W/drv_stale.sh" "$EM" "$W/no-such-template.sh" 2>&1); RC=$?
  if [ "$RC" = 4 ] && printf '%s\n' "$OUT" | grep -q 'hop=FATAL'; then
    ok "J: no tip template -> rc 4 FATAL, never a pass"
  else
    fail "J: rc=$RC with an absent tip; out: $(printf '%s' "$OUT" | tr '\n' '|')"
  fi
else
  fail "G/H/I/J: no template at $TPL -- the hop arms cannot run"
fi
echo "### derived_region_parse_check_guard: $( [ "$FAIL" = 0 ] && echo PASS || echo FAIL )"
exit "$FAIL"
