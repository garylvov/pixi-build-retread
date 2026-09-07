#!/usr/bin/env bash
# owner_export.sh -- THE ONE PRODUCER OF A CLEANUP OWNER'S `--export` CLAUSE.
#
# THE DEFECT, MEASURED (REAP-3 finding 3, 2026-09-07). Four cleanup owners sat
# in the queue with the reason `user env retrieval failed requeued held`:
# 6013350, 6013351, 6014485 and 5841188. Every one of them was submitted with a
# clause of the shape
#
#     --export=ALL,D=<harness dir>,TAG=<tag>,RJ=<relock job>
#
# and the `ALL` is the whole cause: it makes Slurm RETRIEVE the submitting
# process's environment on the target node, and when that retrieval fails the
# job is not refused, it is HELD -- silently, with a reason nobody reads, for
# days. 5841188 has been held since 2026-09-05. REAP-3's own two owners were
# submitted with an EXPLICIT list and no `ALL` (`D=`,`TAG=`,`RJ=`,`DRY_RUN=`,
# `PATH=`,`HOME=`) and came up RUNNING on node2333 (6017145, 6017160), which is
# the reader for that choice. HARNESS-CONSOL-8's owner 6016297 passed no
# `--export` at all and also ran.
#
# WHY A WHOLE FILE FOR ONE STRING. Because there are two submit sites for the
# same owner and they must not drift: phaseN_cert.sh submits the owner, and the
# owner's OWN `owner_wall_check` submits its continuation from inside the
# generated `owner.sbatch`. The continuation site cannot source anything from
# the task tree (a frozen owner reads job-local bytes only -- HARNESS-SYNC-5),
# so this function is shipped into the generated sbatch by `declare -f`, exactly
# as `owner_wall_check` is. One text, two callers, no drift.
#
# WHAT GOES IN THE LIST, AND WHERE EACH NAME IS READ. Derived by grep from the
# scripts the owner actually executes, not from habit:
#   D        phase_template/cleanup_gated.sh  -- `[ -z "${D:-}" ]` guard, `A=$D/artifacts`,
#                                               the `find "$D" -maxdepth 2` job-stdout search.
#   TAG      phase_template/cleanup_gated.sh  -- the gate row, the artifact glob `$TAG-$RJ*`.
#   RJ       phase_template/cleanup_gated.sh  -- the relock job id in every ownership test.
#   DRY_RUN  phase_template/cleanup.sh        -- `${DRY_RUN:-}`, the count-only mode.
#   PATH     the shell itself, and cleanup.sh's `command -v checkquota` fallback.
#   HOME     the shell itself; an owner with no HOME writes its bash state to `/`.
# Everything else those two files read is a LITERAL assignment in the file
# (`T=`, `PERSISTENT_CACHE=`, `ISO_CACHE_PREFIX=`, `ALLOWED_PREFIX=`,
# `ALLOWED_CACHE_PREFIX=`, `SETUP_REFUSED_DEPTH=`, `JOB_FATAL_SEAL_DEPTH=`,
# `JOB_STDOUT_FALLBACK_DEPTH=`, `CQ=`) and therefore must NOT travel in the
# environment: a literal that can be overridden from outside is a second
# producer of a constant.
#
# D/TAG/RJ ARE STILL DERIVABLE. p6ad-4's ROOT FIX (d) taught cleanup_gated.sh to
# derive all three from the root basenames when they are unset, and that stays.
# This clause EXPORTS them when the submitter knows them -- which is the case
# REAP-3 hit, where the derivation would have returned nothing because the
# refused job wrote no `<TAG>-<RJ>*` artifact at all.
#
# ITS READER: tools/owner_export_guard.sh, which shims `sbatch`, drives both
# submit sites, and refuses a clause carrying `ALL` or missing a declared name.

# The declared set, in the order it is emitted. PATH and HOME last so a reader
# sees the payload first.
OWNER_EXPORT_VARS=${OWNER_EXPORT_VARS:-'D TAG RJ DRY_RUN PATH HOME'}

# owner_export_clause [EXTRA=value ...] -> echoes `--export=<list>` on stdout.
#
# A name that is UNSET is skipped -- and said so on the row, because a silently
# absent `D` is exactly what the derivation exists to survive and a reader must
# be able to tell "not exported" from "exported empty". A value carrying a comma
# or whitespace CANNOT travel in this clause (Slurm splits on comma) and the
# function REFUSES rather than emitting a clause that would truncate: an owner
# submitted with half its contract is the failure this file exists to end.
owner_export_clause () {
  local out= names= skipped= v n val extra
  for n in $OWNER_EXPORT_VARS; do
    if [ -n "${!n+set}" ]; then
      val=${!n}
      case $val in
        *,*|*[[:space:]]*)
          echo "### OWNER SUBMIT REFUSED: \$$n contains a comma or whitespace and cannot travel in --export ($n='$val')" >&2
          return 2 ;;
      esac
      out="${out:+$out,}$n=$val"; names="${names:+$names }$n"
    else
      skipped="${skipped:+$skipped }$n"
    fi
  done
  for extra in "$@"; do
    case $extra in
      *=*) ;;
      *) echo "### OWNER SUBMIT REFUSED: extra export '$extra' is not NAME=value" >&2; return 2 ;;
    esac
    case $extra in
      *,*|*[[:space:]]*)
        echo "### OWNER SUBMIT REFUSED: extra export '$extra' contains a comma or whitespace" >&2; return 2 ;;
    esac
    out="${out:+$out,}$extra"; names="${names:+$names }${extra%%=*}"
  done
  [ -n "$out" ] || {
    echo "### OWNER SUBMIT REFUSED: the export list is EMPTY -- not even PATH is set, so the owner would run with no shell environment at all" >&2
    return 2; }
  # The row REAP-3 asked for, on stderr so it cannot land inside a command
  # substitution that is capturing the clause.
  echo "### OWNER SUBMIT export=$names${skipped:+ (unset, not exported: $skipped)} -- NO \`ALL\`: \`--export=ALL,...\` is what held 6013350/6013351/6014485/5841188 in \`user env retrieval failed requeued held\` (REAP-3)" >&2
  printf -- '--export=%s\n' "$out"
  return 0
}
