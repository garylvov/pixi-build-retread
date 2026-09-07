#!/usr/bin/env bash
# script_refs.sh -- THE ONE PARSER for "which scripts does this script read".
# HARNESS-SYNC-5; literal-variable resolution HARNESS-SYNC-7.
#
# It was a function inside harness_sync.sh and it now has TWO callers: the sync's
# read-set check, and phase_template/owner_snapshot.sh, which must copy exactly
# the files a job will read into that job's own root. Two copies of a parser is
# how a snapshot ends up missing the one file the job actually sources, so there
# is one, here, sourced by both:
#
#     . "$(dirname "$0")/script_refs.sh"
#
# OUTPUT, one line per reference, THREE tab-separated fields:
#
#     <basename>\t<absolute path>\tvia=<chain>   resolved THROUGH a variable
#     <basename>\t<absolute path>\t-             resolved from a literal token
#     <basename>\t-\t-                           the basename is known, the path is not
#     ?\t-\t-                                    the reference could not be resolved at all
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
# THE THIRD FIELD IS HARNESS-SYNC-7, AND IT NAMES THE SECOND FIELD'S PROVENANCE.
# `-` means the path was in the text; `via=A->B` means it was computed by
# following literal assignments, and the caller prints that chain so a cleared
# refusal can be audited instead of taken on trust.
#
# WHY THE CHAIN HAD TO BE FOLLOWED.  det162-proof 6015646 ran for hours as a
# RUNNING rc-6 blocker on `moved_row_halves.sh` and `multiarm_preamble.sh`,
# reading NEITHER task copy: its driver says
#
#     WT=/oscar/.../agrescap/worktrees/det162-envseed-proof
#     WTH=$WT/harness
#     MOVED_ROWS=$WTH/tools/moved_row_halves.sh
#     ...
#     source "$WTH/tools/multiarm_preamble.sh"
#
# -- a two- and three-hop chain of LITERAL assignments in the very file being
# parsed, whose value is as knowable from the text as `/abs/path` is.  Refusing
# on it was not caution, it was an unread fact: the parser stopped at the `$`
# and the check then had to assume the worst about a path it could have
# computed exactly.  So a leading `$VAR` is expanded when -- and only when --
# every hop is assigned a LITERAL in the same snapshot, ONE distinct value per
# name, to a depth of at most 3.
#
# WHAT IS STILL NOT RESOLVED STAYS DANGEROUS, and the list is deliberate:
#   * a name assigned from `$( )` or a backquote -- its value is not in the text
#   * a name assigned TWICE to different values -- guessing which one is live is
#     how a wrong path becomes a missed refusal, the one error class this whole
#     check exists to prevent
#   * a name never assigned in this snapshot (it came from the environment)
#   * a chain deeper than 3
#   * a BARE `$(dirname "$0")/x.sh` written directly as the argument of `bash`
#     or `source`: the tokenizer splits on the space inside the substitution and
#     never sees the whole token, so it yields `?` and the job reads EVERYTHING.
#     The FORM THAT PRODUCTION USES -- `CLEANUP=$(dirname "$0")/cleanup.sh` then
#     `bash "$CLEANUP"` -- IS resolved, because there the substitution is on the
#     right-hand side of an assignment where the parser has the whole line.
# All of those yield `-` or `?` and the caller must treat them as before: `-` is
# possibly-that-file, `?` is reads-everything (law 9).

# ---- the literal-assignment side -------------------------------------------
_sr_assign () {                   # $1 = script path, $2 = variable name
  # Echoes the ONE literal value assigned to $2 in $1, or returns 1.
  local f=$1 v=$2 d vals out n
  d=${f%/*}; [ "$d" = "$f" ] && d=.
  vals=$(sed -n "s/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}$v=\(.*\)\$/\2/p" "$f" 2>/dev/null)
  [ -n "$vals" ] || return 1
  # ONE DISTINCT VALUE OR NOTHING.  Two different assignments of the same name
  # in one file is an ambiguity this parser cannot settle -- which one is live
  # depends on control flow it does not evaluate -- and a wrong path here is a
  # MISSED REFUSAL, so ambiguity is refusal.
  out=$(printf '%s\n' "$vals" | sed -e 's/[[:space:]]*#.*$//' -e 's/^"//' -e "s/^'//" -e 's/"$//' -e "s/'\$//" | sort -u)
  n=$(printf '%s\n' "$out" | wc -l)
  [ "$n" -eq 1 ] || return 1
  [ -n "$out" ] || return 1
  # `$(dirname "$0")` is THE ONE command substitution whose value is knowable
  # from the text: $0 is the script being parsed, so its dirname is $d.  This is
  # the shape phase_template/cleanup_gated.sh actually uses.  EVERY OTHER `$( )`
  # or backquote is a value this parser cannot compute and must not guess.
  out=$(printf '%s' "$out" | sed -e 's|\$(dirname[[:space:]][^)]*"\{0,1\}\$0"\{0,1\}[^)]*)|'"$d"'|g' -e 's|\${0%/\*}|'"$d"'|g')
  case "$out" in *'$('*|*'`'*) return 1;; esac
  printf '%s' "$out"
}

_sr_expand () {                   # $1 = script path, $2 = token
  # Echoes "<absolute path>\tvia=<A->B->C>" and returns 0, or returns 1.
  # Only a LEADING `$VAR` / `${VAR}` is expanded, and only into a LITERAL.
  local f=$1 t=$2 depth=0 v rest val chain=
  while : ; do
    case "$t" in '$'*) ;; *) break;; esac
    case "$t" in
      '${'*) v=${t#\$\{}; rest=${v#*\}}; v=${v%%\}*};;
      *)     v=${t#\$};   rest=${v#"${v%%[^A-Za-z0-9_]*}"}; v=${v%%[^A-Za-z0-9_]*};;
    esac
    [ -n "$v" ] || return 1
    depth=$((depth+1)); [ "$depth" -le 3 ] || return 1
    val=$(_sr_assign "$f" "$v") || return 1
    chain=${chain:+$chain->}$v
    t=$val$rest
  done
  case "$t" in
    /*) case "$t" in *'$'*|*'`'*) return 1;; esac
        printf '%s\tvia=%s\n' "$t" "${chain:--}"; return 0;;
  esac
  return 1
}

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
      local b=${tok##*/} v r ex ep ev
      # HARNESS-SYNC-7: a token built from variables is expanded FIRST, and only
      # if every hop is a literal in this same file.  A token with no `$` in it
      # is left to the paths below EXACTLY as before -- this branch may only add
      # resolutions, never change one.
      case "$tok" in
        *'$'*)
          if ex=$(_sr_expand "$f" "$tok"); then   # HARNESS-SYNC-7 CHAIN BRANCH
            ep=${ex%%$'\t'*}; ev=${ex#*$'\t'}
            case "${ep##*/}" in
              *.sh|*.sbatch|*.bash) printf '%s\t%s\t%s\n' "${ep##*/}" "$ep" "$ev"; continue;;
            esac
          fi;;
      esac
      case "$b" in
        *'$'*)
          v=${b#*\$}; v=${v#\{}; v=${v%%[^A-Za-z0-9_]*}
          [ -n "$v" ] || { printf '?\t-\t-\n'; continue; }
          r=$(sed -n "s/^[[:space:]]*$v=.*\/\([A-Za-z0-9_.-]*\.\(sh\|sbatch\|bash\)\).*/\1/p" "$f" | head -1)
          if [ -n "$r" ]; then printf '%s\t-\t-\n' "$r"; else printf '?\t-\t-\n'; fi;;
        *.sh|*.sbatch|*.bash)
          # A LITERAL ABSOLUTE token is the only one whose path is knowable from
          # the text alone.  `$T/tools/x.sh` is knowable only when $T is a
          # literal in this file -- the branch above -- and when it is not, it
          # is not guessed.
          case "$tok" in
            /*) case "$tok" in *'$'*) printf '%s\t-\t-\n' "$b";; *) printf '%s\t%s\t-\n' "$b" "$tok";; esac;;
            *)  printf '%s\t-\t-\n' "$b";;
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
  local f=$1 d=${1%/*} b p v
  [ "$d" = "$1" ] && d=.
  refs_of "$f" | while IFS=$'\t' read -r b p v; do
    if [ "$p" = "-" ] && [ "$b" != '?' ] && [ -f "$d/$b" ]; then
      printf '%s\t%s\tvia=sibling\n' "$b" "$d/$b"
    else
      printf '%s\t%s\t%s\n' "$b" "$p" "${v:--}"
    fi
  done
}
