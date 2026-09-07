#!/usr/bin/env bash
# Guard for CERT_UV_LINK_MODE (C7, 2026-09-03).
#
# Two properties, both of which have already been wrong in this campaign:
#
#   1. ORDERING. `retread_fast_env` exports UV_LINK_MODE=copy and used to be the
#      last word on it. A knob that is exported BEFORE it is silently a no-op --
#      the same shape as the RETREAD_BUILT_OUTPUT_STORE export that turned out to
#      be read by nothing. So: with the knob set to hardlink, the value the env
#      loop actually runs under must be hardlink, even though the (stubbed, but
#      faithfully copy-setting) fast env ran in between.
#   2. REFUSAL. An unrecognised mode must stop the run with a named reason, not
#      hand uv a garbage link mode.
#   3. THE SHIPPED DEFAULT. Case B drives the CERT_UV_LINK_MODE line exactly as
#      it ships (no substitution), so flipping the default without meaning to is
#      a guard failure rather than a silent change of what every cert does.
#
# It drives the REAL phaseN_cert.sh through a derivation, like a phase harness is
# derived, and stops at DRY_RUN=1 -- after the env block, the fast env and the
# link-mode decision, before any install. Nothing is submitted and nothing
# outside the guard's own temp dir is written.
#
# Falsification (checked by hand): move the knob's export ABOVE the
# `. "$FAST_ENV"` line and case A fails with copy; delete the `case` and case C
# exits 0 instead of 2; set the shipped default back to `copy` and case B fails
# while case D still passes.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
FAIL=0
say() { printf '%s\n' "$*"; }

printf '#!/bin/sh\necho guard-stub 0.0.0\n' > "$W/pixi-build-retread"; chmod +x "$W/pixi-build-retread"
mkdir -p "$W/harness/artifacts" "$W/ws" "$W/c"
: > "$W/ws/pixi.toml"; : > "$W/ws/pixi.lock"; : > "$W/lock"
cat > "$W/harness/artifacts/relock_env.sh" <<STAMP
WS=$W/ws
LOCK=$W/lock
EXPECT_LOCK_MD5=$(md5sum < "$W/lock" | awk '{print $1}')
P1_JOB=999998
STAMP
# a probes row for every env the template certifies, so the A5a gate is silent
ENVS=$(sed -n '/^CERT_ENVS=/,/"$/p' "$HERE/phaseN_cert.sh" | tr '\\\n' ' ' | sed 's/CERT_ENVS=//; s/"//g')
for E in $ENVS; do printf '%s\tsys\tprint(1)\n' "$E" >> "$W/probes.tsv"; done
: > "$W/baseline.tsv"
printf '#!/bin/sh\nexit 0\n' > "$W/verdict.sh"; chmod +x "$W/verdict.sh"
# A stub fast env that does the ONE thing the real one does to this question.
# Every retread_* function phaseN_cert.sh calls between here and the point-of-use
# line must be stubbed FAITHFULLY w.r.t. UV_LINK_MODE (i.e. it must not touch it
# unless the real one does). The FIXTURE COMPLETENESS section below is the reader
# that refuses when the cert grows a call this stub never got -- which is exactly
# how this guard went red at a2ca72d (C31-4 added retread_scope_sdist_builds and
# the stub kept defining only retread_fast_env, so every case died at
# "### FATAL retread_scope_sdist_builds refused" with rc=2).
cat > "$W/fastenv.sh" <<'FE'
retread_fast_env () { export UV_LINK_MODE=copy; echo "guard fast env: UV_LINK_MODE=copy"; return 0; }
# C31-4. The real one job-scopes the uv sdist build trees and re-exports the two
# cache dirs; it does NOT touch UV_LINK_MODE (grep it: there is no UV_LINK_MODE
# between `retread_scope_sdist_builds ()` and the end of that function), so a
# stub that is silent about the link mode is the faithful one.
retread_scope_sdist_builds () { echo "guard scope sdist builds: root=$1 (stub, UV_LINK_MODE untouched)"; return 0; }
FE
# The cert reaches four scripts as siblings of $FAST_ENV. Three have an
# absent-branch by design and are deliberately left absent here; the fourth is a
# hard FATAL, so the fixture must supply it.
FIXTURE_OPTIONAL_SIBLINGS="harness_sync.sh harness_commit_resolve.sh harness_drift_check.sh"
printf '#!/bin/sh\nexit 0\n' > "$W/sdist_build_poison_guard.sh"; chmod +x "$W/sdist_build_poison_guard.sh"

say "== FIXTURE COMPLETENESS: the stub covers every dependency the cert grew =="
FIXMISS=""
for FN in $(grep -oE '\bretread_[a-z_]+\b' "$HERE/phaseN_cert.sh" | sort -u); do
  grep -q "^$FN ()" "$W/fastenv.sh" || FIXMISS="$FIXMISS fn:$FN"
done
for SIB in $(grep -oE 'dirname "\$FAST_ENV"\)/[A-Za-z0-9_]+\.sh' "$HERE/phaseN_cert.sh" | sed 's|.*)/||' | sort -u); do
  [ -f "$W/$SIB" ] && continue
  case " $FIXTURE_OPTIONAL_SIBLINGS " in *" $SIB "*) ;; *) FIXMISS="$FIXMISS sib:$SIB";; esac
done
if [ -z "$FIXMISS" ]; then say "  PASS  (stub covers every retread_* call and every required FAST_ENV sibling)"
else say "  FAIL  the cert reaches dependencies this fixture does not supply:$FIXMISS"; FAIL=1; fi

derive() { # dst mode
  sed -e "s|^SNAP=.*|SNAP=$W/pixi-build-retread|" \
      -e "s|^SNAPDIR=.*|SNAPDIR=$W|" \
      -e "s|^D=.*|D=$W/harness|" \
      -e "s|^P1D=.*|P1D=$W/harness|" \
      -e "s|^PROBES_CANON=.*|PROBES_CANON=$W/probes.tsv|" \
      -e "s|^PROBES=.*|PROBES=$W/probes.tsv|" \
      -e "s|^VERDICT=.*|VERDICT=$W/verdict.sh|" \
      -e "s|^CERT_BASELINE=.*|CERT_BASELINE=$W/baseline.tsv|" \
      -e "s|^FAST_ENV=.*|FAST_ENV=$W/fastenv.sh|" \
      -e "s|^FAST_ENV_ALT=.*|FAST_ENV_ALT=\$FAST_ENV|" \
      -e "s|^\[ -f \"\$FAST_ENV\" \].*||" \
      -e "s|^C=/oscar.*|C=$W/c|" \
      ${2:+-e "s|^CERT_UV_LINK_MODE=.*|CERT_UV_LINK_MODE=$2|"} "$HERE/phaseN_cert.sh" > "$1"
  bash -n "$1" || { say "  FAIL  $1 does not parse"; FAIL=1; }
}

run() { DRY_RUN=1 SLURM_JOB_ID=999999 bash "$1" > "$2" 2>&1; echo $?; }

say "== A. hardlink survives the fast env's copy =="
derive "$W/a.sh" hardlink; RC=$(run "$W/a.sh" "$W/a.log")
if [ "$RC" = 0 ] && grep -q '^### link mode AT THE POINT OF USE: UV_LINK_MODE=hardlink ' "$W/a.log"; then say "  PASS  (rc=$RC)"
else say "  FAIL  want rc=0 and the point-of-use line saying hardlink; got rc=$RC"; grep -n 'UV_LINK_MODE\|FATAL' "$W/a.log" | head; FAIL=1; fi

say "== B. the SHIPPED default is hardlink, and survives the fast env =="
# NOTE: no substitution -- this drives the real CERT_UV_LINK_MODE line as it
# ships, so the guard fails if someone edits the default without meaning to.
derive "$W/b.sh" ""; RC=$(run "$W/b.sh" "$W/b.log")
if [ "$RC" = 0 ] && grep -q '^### link mode AT THE POINT OF USE: UV_LINK_MODE=hardlink ' "$W/b.log"; then say "  PASS  (rc=$RC)"
else say "  FAIL  want rc=0 and the point-of-use line saying hardlink; got rc=$RC"; grep -n 'UV_LINK_MODE\|FATAL' "$W/b.log" | head; FAIL=1; fi

say "== D. copy is still reachable, so the escape hatch has a live reader =="
derive "$W/d.sh" copy; RC=$(run "$W/d.sh" "$W/d.log")
if [ "$RC" = 0 ] && grep -q '^### link mode AT THE POINT OF USE: UV_LINK_MODE=copy ' "$W/d.log"; then say "  PASS  (rc=$RC)"
else say "  FAIL  want rc=0 and the point-of-use line saying copy; got rc=$RC"; grep -n 'UV_LINK_MODE\|FATAL' "$W/d.log" | head; FAIL=1; fi

say "== C. an unrecognised mode refuses, naming it =="
derive "$W/c.sh" symlink; RC=$(run "$W/c.sh" "$W/c.log")
if [ "$RC" = 2 ] && grep -q "CERT_UV_LINK_MODE must be copy|hardlink, got 'symlink'" "$W/c.log"; then say "  PASS  (rc=$RC)"
else say "  FAIL  want rc=2 and the named refusal; got rc=$RC"; tail -5 "$W/c.log"; FAIL=1; fi

[ "$FAIL" = 0 ] && { say "CERT_UV_LINK_MODE guard: ALL PASS"; exit 0; }
say "CERT_UV_LINK_MODE guard: FAILED"; exit 1
