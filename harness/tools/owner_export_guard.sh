#!/usr/bin/env bash
# owner_export_guard.sh -- THE READER OF OWNER-EXPORT-1.
#
# WHAT IT GUARDS. Every harness owner submit path builds its `--export` clause
# from ONE producer, `tools/owner_export.sh`, and that clause NEVER carries
# `ALL`. `--export=ALL,...` makes Slurm retrieve the submitting process's
# environment on the target node; a failed retrieval does not refuse the job, it
# HOLDS it, with the reason `user env retrieval failed requeued held` that
# nothing in this harness reads. Four owners sat in exactly that state -- 
# 6013350, 6013351, 6014485 and 5841188 (the last since 2026-09-05) -- while
# REAP-3's own two owners, submitted with an explicit list, came up RUNNING
# (6017145, 6017160 on node2333).
#
# HOW IT GUARDS IT: BY EXECUTION, NOT BY GREP. Arms C and D SHIM `sbatch` on
# PATH and drive the real generated `owner.sbatch` through its own continuation
# path, then read the argv the shim recorded. A guard that greps a file for the
# clause it wants passes a file that says the right thing and does the wrong one
# -- which is the failure mode this campaign has now paid for twice.
#
# ARMS
#   A  STATIC, WHOLE TREE. No `--export=` clause anywhere in harness/ may carry
#      `ALL`, in code OR in a printed/documented usage line -- a printed clause
#      is a submit path, because it is what the next driver copies. Prose that
#      EXPLAINS the ban is allowed; `--export=ALL` inside a clause is not.
#   B  THE PRODUCER, EXECUTED. Every declared name that is set reaches the
#      clause; an unset name is skipped and NAMED; a value with a comma or a
#      space is REFUSED rather than silently truncated.
#   C  MUTATION. `ALL` restored at the head of the producer's list -> arm A and
#      the clause assertion must go RED. A guard that cannot fail is a defect.
#   D  THE CONTINUATION, EXECUTED END TO END. owner_snapshot.sh generates an
#      owner.sbatch; that sbatch is run with a shimmed `sbatch` and a forced
#      short wall so `owner_wall_check` takes its continuation branch; the argv
#      the shim recorded must carry an explicit clause, no `ALL`, and the bumped
#      OWNER_CONT_N.
#   E  THE TASK LAYOUT (MERGE-V-2-3). Arms C and D drive the REPO-shaped copy.
#      Production runs the SYNCED copy, <task>/tools/phase_template beside
#      <task>/tools, whose sibling spelling is `$HERE/../` -- and that spelling
#      was missing, so the production copy refused rc 2 on every run while this
#      guard was green 27/0. Arm E builds the task shape and drives it end to
#      end, and reads the `### OWNER SNAPSHOT export_lib=` row to say WHICH copy
#      of the producer was sourced.
#   F  MUTATION for E: delete the task spellings from a copy and the SAME
#      fixture refuses rc 2 again -- the production defect, reproduced.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)                 # the harness/ tree
PROD=$HERE/owner_export.sh
SNAPT=$ROOT/phase_template/owner_snapshot.sh
W=$(mktemp -d "${TMPDIR:-/tmp}/ownerexport.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
fail=0
say () { echo "### OWNER-EXPORT GUARD $*"; }
ok  () { say "PASS $*"; }
bad () { say "FAIL $*"; fail=1; }

[ -f "$PROD" ]  || { say "FATAL no owner_export.sh next to this guard ($PROD)"; exit 3; }
[ -f "$SNAPT" ] || { say "FATAL no owner_snapshot.sh ($SNAPT)"; exit 3; }

# ---- ARM A: STATIC, WHOLE TREE ----------------------------------------------
# WHAT COUNTS AS A SUBMIT PATH, stated mechanically so the rule can be applied
# rather than argued. Every `--export=` token in harness/ is collected with its
# whole line, and the line decides:
#   * a SHELL COMMENT line (first non-space character `#`) is PROSE. Several
#     files explain, at length, why `ALL` is banned and what it did to the
#     HARNESS_COMMIT pin -- that text must survive, it is the reason.
#   * an INDENTED line in a `.md` file is a USAGE BLOCK -- what an operator
#     copies -- and is a submit path exactly like code.
#   * anything else is CODE.
# CODE and USAGE carrying `ALL` are failures; PROSE is counted and reported.
AV=$W/a.violations; AP=$W/a.prose; : > "$AV"; : > "$AP"; ACOUNT=0
while IFS= read -r hit; do
  f=${hit%%:*}; rest=${hit#*:}; n=${rest%%:*}; line=${rest#*:}
  ACOUNT=$((ACOUNT + 1))
  # A REAL CLAUSE, not a mention. `ALL` followed by an actual `NAME=` is what
  # would be handed to sbatch; `--export=ALL,...` and a bare `--export=ALL` are
  # how this file, cleanup_gated.sh's refusal rows and this guard itself NAME
  # the banned shape in order to forbid it. A rule that cannot tell the ban from
  # the thing banned would force every explanation of the defect out of the
  # tree, and the explanation is the reason.
  printf '%s' "$line" | grep -qE -- '--export=ALL,[A-Za-z_][A-Za-z0-9_]*=' || continue
  trimmed=${line#"${line%%[![:space:]]*}"}
  case $trimmed in
    '#'*) printf '%s:%s: %s\n' "$f" "$n" "$trimmed" >> "$AP"; continue ;;
  esac
  case $f in
    *.md) case $line in
            [[:space:]]*) : ;;                       # indented = usage block = a submit path
            *) printf '%s:%s: %s\n' "$f" "$n" "$trimmed" >> "$AP"; continue ;;
          esac ;;
  esac
  printf '%s:%s: %s\n' "$f" "$n" "$trimmed" >> "$AV"
done < <(grep -rn -- '--export=' "$ROOT" 2>/dev/null)
AVN=$(grep -c . "$AV" || true); APN=$(grep -c . "$AP" || true)
say "A scanned lines carrying --export= : $ACOUNT (code/usage violations=$AVN, prose mentions=$APN)"
[ "$ACOUNT" -ge 1 ] || bad "A found NO --export= line at all -- this arm would be green against nothing"
if [ "$AVN" = 0 ]; then
  ok "A no --export= clause in CODE or in a copyable usage block carries ALL"
else
  bad "A $AVN --export= clause(s) in code or usage still carry ALL:"
  sed 's/^/###     /' "$AV"
fi

# ---- ARM B: THE PRODUCER, EXECUTED ------------------------------------------
B1=$( set +e
  # shellcheck source=/dev/null
  . "$PROD"
  D=/tmp/harnessdir TAG=D99 RJ=123456 DRY_RUN=0 PATH=/usr/bin:/bin HOME=/tmp/home
  export D TAG RJ DRY_RUN PATH HOME
  owner_export_clause 2>"$W/b1.err" )
B1E=$(cat "$W/b1.err")
case $B1 in
  --export=*) ok "B the clause is a single --export token: $B1" ;;
  *) bad "B the producer did not emit an --export token (got '$B1')" ;;
esac
printf '%s' "$B1" | grep -q 'ALL' && bad "B the clause carries ALL" || ok "B the clause carries NO ALL"
for v in D TAG RJ DRY_RUN PATH HOME; do
  printf '%s' "$B1" | grep -q -- "[=,]$v=" && ok "B $v is in the clause" \
    || bad "B $v is NOT in the clause -- the owner would run with half its contract"
done
printf '%s\n' "$B1E" | grep -q '### OWNER SUBMIT export=' \
  && ok "B the row prints: $(printf '%s\n' "$B1E" | grep -m1 '### OWNER SUBMIT export=')" \
  || bad "B no '### OWNER SUBMIT export=' row was printed -- an unread submit contract is the defect"

# an UNSET name is skipped and NAMED, not silently dropped
B2=$( set +e
  # shellcheck source=/dev/null
  . "$PROD"
  unset DRY_RUN
  D=/tmp/harnessdir TAG=D99 RJ=123456 PATH=/usr/bin:/bin HOME=/tmp/home
  export D TAG RJ PATH HOME
  owner_export_clause 2>"$W/b2.err" )
printf '%s' "$B2" | grep -q 'DRY_RUN=' && bad "B2 an UNSET DRY_RUN reached the clause" \
  || ok "B2 an unset name is skipped"
grep -q 'unset, not exported: DRY_RUN' "$W/b2.err" \
  && ok "B2 the skipped name is NAMED on the row" \
  || bad "B2 the skipped name is not named -- 'not exported' and 'exported empty' must be distinguishable"

# a comma in a value REFUSES rather than truncating
B3RC=$( set +e
  # shellcheck source=/dev/null
  . "$PROD"
  D='/tmp/a,b' TAG=D99 RJ=123456 DRY_RUN=0 PATH=/usr/bin:/bin HOME=/tmp/home
  export D TAG RJ DRY_RUN PATH HOME
  owner_export_clause >"$W/b3.out" 2>"$W/b3.err"; echo $? )
[ "$B3RC" = 2 ] && ok "B3 a comma in a value REFUSES (rc=2) instead of truncating the clause" \
  || bad "B3 a comma in a value returned rc=$B3RC, want 2 -- Slurm splits on comma and the rest of the contract would vanish"
grep -q 'OWNER SUBMIT REFUSED' "$W/b3.err" \
  && ok "B3 the refusal names itself" || bad "B3 the refusal is silent"

# ---- ARM C: MUTATION -- restore ALL and both readers must go RED -------------
# The mutation puts the banned word back at the head of the emitted list, and it
# is written as `"ALL,$out"` rather than as a literal clause ON PURPOSE: arm A
# scans THIS FILE too, and a guard that must break its own rule in order to test
# its own rule is a guard nobody can read.
MUT=$W/owner_export_mutant.sh
sed 's|"\$out"$|"ALL,$out"|' "$PROD" > "$MUT"
if cmp -s "$PROD" "$MUT"; then
  bad "C the mutation edited nothing -- arm C is asserting against an unmutated file and cannot fail"
else
  C1=$( set +e
    # shellcheck source=/dev/null
    . "$MUT"
    D=/tmp/harnessdir TAG=D99 RJ=123456 DRY_RUN=0 PATH=/usr/bin:/bin HOME=/tmp/home
    export D TAG RJ DRY_RUN PATH HOME
    owner_export_clause 2>/dev/null )
  if printf '%s' "$C1" | grep -q -- '--export=ALL'; then
    ok "C (mutation) ALL restored -> the clause is '$C1', which arm A's rule rejects and arm B's does too"
  else
    bad "C the mutant with ALL restored did NOT emit it ('$C1') -- arm B proves nothing"
  fi
fi

# ---- ARM D: THE CONTINUATION, EXECUTED --------------------------------------
# A real generated owner.sbatch, run with a shimmed sbatch and a wall too small
# for the census it takes, so owner_wall_check MUST take the continuation branch.
#
# THE FIXTURE IS TASK-SHAPED, and it has to be. DET-1-6-a made owner_snapshot.sh
# REFUSE a source whose provenance it cannot identify -- "Freezing it would put a
# copy of nobody-knows-what in the job root under a row claiming provenance" --
# so a stub in a bare temp directory is neither a git worktree nor a task copy
# and every arm below dies rc 2 for a reason that has nothing to do with the
# export clause. Job 6020490 measured exactly that. `merge-h/` beside a
# `tools/.harness_synced_commit` is the shape cleanup_wall_guard.sh already uses
# for the same tool and the same reason.
JR=$W/jobroot; mkdir -p "$JR"
FAKE=$W/task/merge-h; mkdir -p "$FAKE" "$W/task/tools"
printf '3333333333333333333333333333333333333333\n' > "$W/task/tools/.harness_synced_commit"
# CLEANUP-WALL-3 (2026-09-07): the stub now prints the `### removed <root>` row
# in cleanup.sh's shape. A continuation is EARNED by a pass since CLEANUP-WALL-3
# -- a pass that removed nothing was not cut short, it was done or refused -- so
# a stub that removes nothing gets no continuation and arm D would measure
# nothing. It still unlinks nothing: it prints the row and returns.
cat > "$FAKE/cleanup_gated.sh" <<'EOS'
#!/usr/bin/env bash
echo "### FIXTURE cleanup_gated: roots=$*"
for r in "$@"; do echo "### removed $r rc=0 wall=0s exists_after=YES (FIXTURE: nothing was unlinked)"; done
exit 0
EOS
chmod +x "$FAKE/cleanup_gated.sh"
ROOTD=$W/roots/certD99-123456; mkdir -p "$ROOTD/a/b"; : > "$ROOTD/a/b/f1"; : > "$ROOTD/a/f2"
( cd "$W" && bash "$SNAPT" "$JR" "$FAKE/cleanup_gated.sh" --roots "$ROOTD" ) > "$W/snap.out" 2>&1
OSB=$JR/owner-snapshot/owner.sbatch
if [ ! -f "$OSB" ]; then
  bad "D owner_snapshot.sh produced no owner.sbatch -- see rows:"; sed 's/^/###     /' "$W/snap.out"
else
  grep -q 'owner_export_clause' "$OSB" \
    && ok "D the generated owner.sbatch SHIPS owner_export_clause (declare -f)" \
    || bad "D the generated owner.sbatch does not ship owner_export_clause -- the continuation would build its own clause"
  grep -q "^OWNER_EXPORT_VARS=" "$OSB" \
    && ok "D the generated owner.sbatch carries the declared name set as a literal" \
    || bad "D the generated owner.sbatch does not carry OWNER_EXPORT_VARS"
  SHIMD=$W/shim; mkdir -p "$SHIMD"
  cat > "$SHIMD/sbatch" <<EOS
#!/usr/bin/env bash
printf '%s\n' "\$@" > "$W/sbatch.argv"
echo 999999
exit 0
EOS
  chmod +x "$SHIMD/sbatch"
  # OWNER_WALL_COVERS is written into the sbatch by the generator; force it to 0
  # so the live census (which is >0) exceeds it and the continuation branch runs.
  # CLEANUP-WALL-3: the pressure term goes with it. The continuation now also
  # requires the pass to have consumed OWNER_WALL_PRESSURE_NUM/DEN of a DERIVED
  # wall; 0/5 is always reached, so this arm keeps measuring the export clause
  # and not the clock. (An owner that had to burn 4/5 of a 3600 s wall to reach
  # its own sbatch line would be a guard that measures NFS.)
  sed -e 's/^OWNER_WALL_COVERS=.*/OWNER_WALL_COVERS=0/' \
      -e 's/^OWNER_WALL_PRESSURE_NUM=.*/OWNER_WALL_PRESSURE_NUM=0/' "$OSB" > "$W/owner.forced.sbatch"
  ( export PATH=$SHIMD:$PATH SLURM_JOB_ID=111111 SLURM_JOB_NAME=guard-owner
    export D=/tmp/harnessdir TAG=D99 RJ=123456 DRY_RUN=0
    bash "$W/owner.forced.sbatch" "$ROOTD" ) > "$W/owner.out" 2>&1
  if [ ! -s "$W/sbatch.argv" ]; then
    bad "D the continuation never reached sbatch -- this arm measured nothing. Rows:"
    sed 's/^/###     /' "$W/owner.out" | head -30
  else
    DCL=$(grep -m1 -- '^--export=' "$W/sbatch.argv" || true)
    if [ -z "$DCL" ]; then
      bad "D the continuation submitted with NO --export clause at all"
      sed 's/^/###     argv: /' "$W/sbatch.argv"
    else
      printf '%s' "$DCL" | grep -q 'ALL' \
        && bad "D the continuation clause carries ALL: $DCL" \
        || ok "D the continuation clause carries NO ALL: $DCL"
      printf '%s' "$DCL" | grep -q 'OWNER_CONT_N=1' \
        && ok "D the continuation bumped OWNER_CONT_N to 1 inside the explicit clause" \
        || bad "D OWNER_CONT_N=1 is not in the clause ($DCL) -- the chain counter would never advance and OWNER_CONT_MAX could not stop it"
      for v in D TAG RJ DRY_RUN PATH HOME; do
        printf '%s' "$DCL" | grep -q -- "[=,]$v=" && ok "D the continuation clause names $v" \
          || bad "D the continuation clause omits $v"
      done
    fi
    grep -q '### OWNER SUBMIT export=' "$W/owner.out" \
      && ok "D the owner printed its '### OWNER SUBMIT export=' row" \
      || bad "D the owner submitted a continuation without printing what it exported"
  fi
fi


# ---- ARM E: THE TASK LAYOUT, WHICH IS WHERE PRODUCTION RUNS -----------------
# MERGE-V-2-3. Every arm above drives the REPO-shaped copy: harness/phase_template
# beside harness/tools. Production runs the SYNCED copy, <task>/tools/phase_template
# beside <task>/tools -- a different sibling spelling entirely -- and this guard
# was green 27/0 while that copy refused rc 2 on every run, because it had never
# been asked to run in the shape it ships in. CLAUDE.md law 7 hazard (b): two
# copies of one module is the normal state here, and path order alone decides
# which one a process reads, so a fixture that only builds one shape measures
# one shape.
TASK=$W/task2
mkdir -p "$TASK/tools/phase_template" "$TASK/merge-h" "$TASK/roots"
cp "$SNAPT" "$TASK/tools/phase_template/owner_snapshot.sh"
cp "$PROD"  "$TASK/tools/owner_export.sh"
cp "$HERE/script_refs.sh" "$TASK/tools/script_refs.sh"
printf '3333333333333333333333333333333333333333\n' > "$TASK/tools/.harness_synced_commit"
cp "$FAKE/cleanup_gated.sh" "$TASK/merge-h/cleanup_gated.sh"
TROOT=$TASK/roots/certE99-123456; mkdir -p "$TROOT/a/b"; : > "$TROOT/a/b/f1"; : > "$TROOT/a/f2"
TJR=$W/jobroot_task; mkdir -p "$TJR"
( cd "$TASK" && bash "$TASK/tools/phase_template/owner_snapshot.sh" "$TJR" "$TASK/merge-h/cleanup_gated.sh" --roots "$TROOT" ) > "$W/snapE.out" 2>&1
ERC=$?
if grep -q 'OWNER SNAPSHOT REFUSED: no owner_export.sh' "$W/snapE.out"; then
  bad "E the TASK-shaped copy still refuses to find its sibling owner_export.sh (rc=$ERC) -- MERGE-V-2-3 is not fixed:"
  sed 's/^/###     /' "$W/snapE.out" | head -8
else
  ok "E the TASK-shaped copy (<task>/tools/phase_template beside <task>/tools) resolves owner_export.sh"
fi
ELIB=$(grep -m1 '^### OWNER SNAPSHOT export_lib=' "$W/snapE.out" | sed 's/^### OWNER SNAPSHOT export_lib=//')
if [ "$ELIB" = "$TASK/tools/owner_export.sh" ]; then
  ok "E the run PRINTED which copy it sourced, and it is the task one ($ELIB)"
else
  bad "E export_lib row is '$ELIB', want '$TASK/tools/owner_export.sh' -- a job log cannot say which producer it read"
fi
SLIB=$(grep -m1 '^### OWNER SNAPSHOT script_refs=' "$W/snapE.out" | sed 's/^### OWNER SNAPSHOT script_refs=//')
if [ "$SLIB" = "$TASK/tools/script_refs.sh" ]; then
  ok "E the SECOND sibling (script_refs.sh) resolved through the task spelling too, and said so ($SLIB)"
else
  bad "E script_refs row is '$SLIB', want '$TASK/tools/script_refs.sh' -- the second ladder still names only the repo layout"
fi
if [ -f "$TJR/owner-snapshot/owner.sbatch" ]; then
  ok "E the TASK-shaped copy went on to generate owner.sbatch end to end (rc=$ERC)"
else
  bad "E the TASK-shaped copy produced no owner.sbatch (rc=$ERC):"; sed 's/^/###     /' "$W/snapE.out" | head -12
fi

# ---- ARM F: THE MUTATION, which is the production defect itself -------------
# Delete the two task spellings from a COPY and the SAME fixture must go back to
# refusing rc 2. Without this, arm E says only that some ladder exists.
MUT=$TASK/tools/phase_template/owner_snapshot.mut.sh
# The mutation is the PRE-FIX LADDER, restored inside the one resolver: the repo
# spellings only, exactly what shipped before MERGE-V-2-3.
sed -e 's|^  for c in "\$HERE/\$n" "\$HERE/\.\./\$n" "\$HERE/\.\./tools/\$n" \\$|  for c in "$HERE/$n" "$HERE/../tools/$n"; do|' \
    -e '/^           "\/oscar\/data\/stellex\/glvov\/agrescap\/tasks\/retread-4-11\/tools\/\$n"; do$/d' \
    "$SNAPT" > "$MUT"
MJR=$W/jobroot_mut; mkdir -p "$MJR"
if bash -n "$MUT" 2>/dev/null && grep -q '^  for c in "\$HERE/\$n" "\$HERE/\.\./tools/\$n"; do$' "$MUT"; then
  ( cd "$TASK" && bash "$MUT" "$MJR" "$TASK/merge-h/cleanup_gated.sh" --roots "$TROOT" ) > "$W/snapF.out" 2>&1
  MRC=$?
  if [ "$MRC" = 2 ] && grep -q 'OWNER SNAPSHOT REFUSED: no owner_export.sh' "$W/snapF.out"; then
    ok "F MUTATION -- with the task spellings removed the SAME fixture refuses rc 2 again, so arm E is a real resolution and not a constant"
  else
    bad "F the mutant did NOT refuse (rc=$MRC) -- arm E proves nothing"
  fi
else
  bad "F could not build the spelling-removed mutant of owner_snapshot.sh"
fi
say "rc=$fail"
[ "$fail" = 0 ] && { say "ALL ARMS PASS"; exit 0; }
say "FAILED"; exit 1
