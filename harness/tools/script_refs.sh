#!/usr/bin/env bash
# script_refs.sh -- THE ONE PARSER for "which scripts does this script read".
# HARNESS-SYNC-5.
#
# It was a function inside harness_sync.sh and it now has TWO callers: the sync's
# read-set check, and phase_template/owner_snapshot.sh, which must copy exactly
# the files a job will read into that job's own root. Two copies of a parser is
# how a snapshot ends up missing the one file the job actually sources, so there
# is one, here, sourced by both:
#
#     . "$(dirname "$0")/script_refs.sh"
#
# OUTPUT, one line per reference, TWO tab-separated fields:
#
#     <basename>\t<absolute path>      the reference resolved to a literal path
#     <basename>\t-                    the basename is known, the path is not
#     ?\t-                             the reference could not be resolved at all
#
# THE SECOND FIELD IS THE HARNESS-SYNC-5 ADDITION AND IT IS THE WHOLE POINT.
# The read-set check intersected the install set with the read set BY BASENAME,
# so a job running a JOB-LOCAL SNAPSHOT of cleanup_gated.sh still collided with
# the task copy of that name and pinned the whole harness against any install --
# which is what det1f-cleanup 5999937 and det141-cleanup 6001240 did for hours
# while unlinking millions of entries. A reference that resolves to a literal
# path OUTSIDE the task tree is not a read of the file the sync would rewrite,
# and now it can be seen not to be.
#
# WHAT IS NOT RESOLVED STAYS DANGEROUS. A relative token, or one built from a
# variable, yields `-` and the caller must treat it as possibly-that-file. An
# unresolvable reference yields `?` and the caller must treat the job as reading
# EVERYTHING (law 9: an unreadable job is not a safe job).

refs_of () {                      # $1 = script path
  local f=$1
  # TWO patterns, not one.  `bash` and `source` are verbs anywhere a command may
  # start, INCLUDING inside `$( )` -- that is how 5992569 reaches
  # harness_commit_resolve.sh.  The BARE DOT is not: in `grep -c . "$OB"` the dot
  # is an ARGUMENT, and matching it makes det1_proof2.sh look like it sources
  # four files it never touches, none of them resolvable, which would mark that
  # job undeterminable and turn EVERY sync into a refusal for its whole run.
  # So the dot is a verb only in COMMAND POSITION: line start, or after ; & | (
  # or a backquote.
  { grep -hoE '(^|[[:space:]]|[(`;&|])(bash|source)[[:space:]]+[^[:space:];&|)]+' "$f" 2>/dev/null
    grep -hoE '(^|[;&|(`])[[:space:]]*\.[[:space:]]+[^[:space:];&|)]+'            "$f" 2>/dev/null; } \
  | awk '{print $NF}' | tr -d '\042\047' | while IFS= read -r tok; do
      local b=${tok##*/} v r
      case "$b" in
        *'$'*)
          v=${b#*\$}; v=${v#\{}; v=${v%%[^A-Za-z0-9_]*}
          [ -n "$v" ] || { printf '?\t-\n'; continue; }
          r=$(sed -n "s/^[[:space:]]*$v=.*\/\([A-Za-z0-9_.-]*\.\(sh\|sbatch\|bash\)\).*/\1/p" "$f" | head -1)
          if [ -n "$r" ]; then printf '%s\t-\n' "$r"; else printf '?\t-\n'; fi;;
        *.sh|*.sbatch|*.bash)
          # A LITERAL ABSOLUTE token is the only one whose path is knowable from
          # the text alone. `$T/tools/x.sh` is not: $T is a variable this parser
          # does not evaluate, and guessing it is how a wrong path becomes a
          # missed refusal.
          case "$tok" in
            /*) case "$tok" in *'$'*) printf '%s\t-\n' "$b";; *) printf '%s\t%s\n' "$b" "$tok";; esac;;
            *)  printf '%s\t-\n' "$b";;
          esac;;
        *) ;;                     # `bash -c`, `. /etc/profile`, flags: not a task copy
      esac
    done
}

refs_basenames_of () {            # $1 = script path -> field 1 only, the old contract
  refs_of "$1" | cut -f1
}

refs_of_sibling_resolved () {     # $1 = script path; refs_of + the sibling rule
  # THE SIBLING RULE, and it is what makes an owner snapshot legible to the
  # sync. A script that names a bare basename -- `CLEANUP=$(dirname "$0")/
  # cleanup.sh` then `bash "$CLEANUP"`, which is exactly cleanup_gated.sh -- is
  # calling THE COPY BESIDE ITSELF. So an unresolved reference is resolved to
  # <dir of the referring script>/<basename> WHEN A FILE IS ACTUALLY THERE, and
  # left unresolved when it is not. Both halves matter: with it, the task copy
  # of the gate resolves to the task copy of cleanup.sh (a real read of a synced
  # file -- still a refusal) and the FROZEN gate resolves to the frozen
  # cleanup.sh beside it (not a read of anything the sync installs). Without the
  # `-f` condition it would be a guess, and a wrong path here is a missed
  # refusal, which is the one error class this whole check exists to prevent.
  local f=$1 d=${1%/*} b p
  [ "$d" = "$1" ] && d=.
  refs_of "$f" | while IFS=$'\t' read -r b p; do
    if [ "$p" = "-" ] && [ "$b" != '?' ] && [ -f "$d/$b" ]; then
      printf '%s\t%s\n' "$b" "$d/$b"
    else
      printf '%s\t%s\n' "$b" "$p"
    fi
  done
}
