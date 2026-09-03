#!/usr/bin/env bash
# Certification verdict. Replaces the launcher's `rg`-based gate, which was
# unsound twice over: `rg` is absent from the launcher's exported PATH (so the
# `if` was command-not-found -> false -> fell through to the SUCCESS branch
# without inspecting anything), and it scored against a 9-row baseline so any
# 26-row run produced NEW_ROW and could never have passed had it run.
#
# This compares a full run against a full run, row for row, on the fields that
# carry meaning: install rc, TierA, TierB, verify, verdict.
# Wall-clock is deliberately excluded -- it varies run to run and is not a
# correctness signal.
#
# 2026-09-01 (operator ruling): ATT (repair-attempt count, field 7 / f[6]) is
# NO LONGER SCORED -- it is informational only. ATT measures repair EFFORT, not
# outcome; the verdict field (f[7]) already downgrades to AMBER-repaired on any
# nonzero ATT, so scoring the raw count double-counts retry noise. Control cert
# 5469527 (unmodified manifest) failed this gate on 2 ATT-only rows, which is
# exactly the false positive the ruling removes. Prior tuple was
# (f[1],f[3],f[4],f[5],f[6],f[7]); f[6] dropped. Backup of the pre-ruling
# script: cert_verdict.sh.bak-preATT-20260901.
#
# Exit codes are typed so a caller cannot mistake a setup failure for a pass:
#   0  every row present and identical
#   1  at least one row differs, or a baseline row is missing from current
#   2  a required file is unreadable (setup failure, NOT a verdict)
set -uo pipefail

CUR=${1:?usage: cert_verdict.sh <current_results.tsv> <baseline_results.tsv>}
BASE=${2:?usage: cert_verdict.sh <current_results.tsv> <baseline_results.tsv>}
for f in "$CUR" "$BASE"; do
  [ -r "$f" ] || { echo "SETUP-FAIL unreadable: $f" >&2; exit 2; }
  [ -s "$f" ] || { echo "SETUP-FAIL empty: $f" >&2; exit 2; }
done

python3 - "$CUR" "$BASE" <<'PY'
import sys
def load(p):
    d={}
    for line in open(p):
        f=line.rstrip("\n").split("\t")
        # env, install_rc, wall, tierA, tierB, verify, attempts, verdict
        # f[2] (wall) and f[6] (attempts) are informational, not scored.
        if len(f)>=8:
            d[f[0]]=(f[1],f[3],f[4],f[5],f[7])
    return d
cur=load(sys.argv[1]); base=load(sys.argv[2])
if not cur or not base:
    print("SETUP-FAIL no parseable rows"); sys.exit(2)
missing=[e for e in base if e not in cur]
extra=[e for e in cur if e not in base]
diff=[e for e in cur if e in base and cur[e]!=base[e]]
for e in sorted(missing): print(f"NOT_RUN\t{e}\tbaseline={base[e]}")
for e in sorted(extra):   print(f"NEW_ROW\t{e}\tcurrent={cur[e]}")
for e in sorted(diff):    print(f"OUTCOME_DIFF\t{e}\tbaseline={base[e]}\tcurrent={cur[e]}")
print(f"### rows current={len(cur)} baseline={len(base)} "
      f"not_run={len(missing)} new={len(extra)} differing={len(diff)}")
# NEW_ROW alone is not a failure: a run may legitimately cover more
# environments than an older baseline. NOT_RUN and OUTCOME_DIFF are.
sys.exit(1 if (missing or diff) else 0)
PY
