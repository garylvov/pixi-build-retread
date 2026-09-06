#!/usr/bin/env bash
# driver_exit_scan.sh -- HARNESS-EXIT-3 / HARNESS-SEAM-1. The static half of the
# payload seam.
#
# The runtime seam in driver_exit_payload_shim.sh covers every command invoked
# BY NAME: a function shadows it, PATH cannot reach past the shim, and a name we
# did not stub lands in command_not_found_handle and refuses. The one thing PATH
# and functions cannot reach is a command invoked BY PATH -- `"$BIN" store-reap`
# or a literal `/oscar/.../some-binary`. This scanner finds exactly those, and
# decides, for each one, whether THE SEAM CAN ACTUALLY STUB IT.
#
# HARNESS-SEAM-1: WHY THE RULE CHANGED, AND WHAT IT IS NOW.
# The first version of this file answered that question from a HAND-MAINTAINED
# LIST OF VARIABLE NAMES (`DEG_SCAN_COVERED_VARS = BIN PIXI RETREAD RETREAD_BIN
# PIXI_BIN BINARY EXE CARGO`): a wrapper whose payload variable was called
# anything else was REFUSED, and the fix a lane reached for was to append its
# own name to the list -- or, as STORE-REAP-3 actually did, to RENAME ITS
# VARIABLE from `TIP` to `RETREAD_BIN` so the list would accept it. That is a
# workaround holding a defect shut, and it is the same shape as the wrapper list
# HARNESS-EXIT-3 abolished one layer up: a register of names standing in for a
# capability.
#
# THE RULE NOW, and it names no variable at all:
#
#   a command-position token `"$VAR" args...` is COVERED iff the value of VAR,
#   RESOLVED STATICALLY from the wrapper's own text (its assignments, its
#   `export`s, the pin/config files it sources), has a BASENAME the seam can
#   actually place a stub for -- which is asked of the seam by looking for that
#   stub, not by consulting a list.
#
#   * resolved, basename stubbable  -> COVERED. The caller binds the seam's stub
#     over the resolved path, so the invocation is intercepted rather than run.
#   * resolved, basename not stubbable (a directory, a lane's own tool) ->
#     UNCOVERED. Refused, and the row names the path and the basename.
#   * NOT resolvable statically (a command substitution, a positional, a
#     variable nothing in the wrapper ever assigns) -> UNRESOLVED. Refused, and
#     the row NAMES THE VARIABLE AND THE LINE. It is never executed and never
#     silently passed: an unresolvable payload is exactly the case where we
#     cannot know what would run.
#
# It is deliberately CONSERVATIVE in the safe direction: it over-reports command
# position (an awk program body, a `case` subject) and every over-report costs
# one baselined UNCOVERED row, never an execution.
#
# It also reports the shapes the discovery table is built from.
#
# usage: driver_exit_scan.sh paths    <file>   # path-position tokens, one per line
#        driver_exit_scan.sh verdicts <file>   # COVERED/UNCOVERED/UNRESOLVED rows
#        driver_exit_scan.sh assigns  <file>   # the static assignment table
#        driver_exit_scan.sh payload  <file>   # payload invocation shape
#        driver_exit_scan.sh last     <file>   # last effective command word
#
# The seam's stub table is the AUTHORITY on what can be stubbed, so this file
# reads it from driver_exit_payload_shim.sh rather than keeping a second copy.
if [ -z "${DEG_PAYLOAD_NAMES:-}" ]; then
  DEG_SCAN_SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  # shellcheck disable=SC1091
  . "$DEG_SCAN_SELF_DIR/driver_exit_payload_shim.sh"
fi

deg_scan_awk='
BEGIN { inhd=0 }
{
  line=$0
  if (inhd) { t=line; sub(/^[ \t]*/,"",t); sub(/[ \t]*$/,"",t); if (t==hdtag) inhd=0; next }
  hdnext=0; hdtagnext=""
  if (match(line, /<<-?[ \t]*['"'"'"]?[A-Za-z_][A-Za-z0-9_]*['"'"'"]?/)) {
    tag=substr(line, RSTART, RLENGTH); gsub(/^<<-?[ \t]*/,"",tag); gsub(/['"'"'"]/,"",tag)
    hdnext=1; hdtagnext=tag
  }
  s=line; sub(/^[ \t]*/,"",s)
  if (s ~ /^#/ || s=="") { if(hdnext){inhd=1;hdtag=hdtagnext}; next }
  changed=1
  while (changed) {
    changed=0
    if (s ~ /^(if|then|else|elif|fi|while|until|do|done|case|esac|for|select|function|\{|\}|!|\(|&&|\|\||;)[ \t]+/) { sub(/^[^ \t]+[ \t]+/,"",s); changed=1 }
    else if (s ~ /^(local|export|declare|readonly|typeset|unset|alias)[ \t]/) { s=""; break }
    else if (s ~ /^[A-Za-z_][A-Za-z0-9_]*=[^ \t]*[ \t]+/ && s !~ /^[A-Za-z_][A-Za-z0-9_]*=[^ \t]*[$`]\(/) { first=s; sub(/[ \t].*$/,"",first); nq=gsub(/"/,"\"",first); ns=gsub(/'"'"'/,"'"'"'",first); if (nq%2==0 && ns%2==0) { sub(/^[^ \t]+[ \t]+/,"",s); changed=1 } else { s=""; break } }
  }
  if (s=="") { if(hdnext){inhd=1;hdtag=hdtagnext}; next }
  tok=s; sub(/[ \t].*$/,"",tok)
  if (tok ~ /^[A-Za-z_][A-Za-z0-9_]*=/) tok=""
  if (tok!="") print tok
  if (hdnext) { inhd=1; hdtag=hdtagnext }
}'

# Every static assignment in a file, as VAR<TAB>RAW-VALUE, in file order. The
# LAST row for a name wins, which is what a straight-line wrapper does. A value
# containing a command substitution is emitted verbatim and fails to resolve
# later -- that is the point: we cannot know what `$(...)` would produce.
deg_assign_awk='
BEGIN { inhd=0 }
{
  line=$0
  if (inhd) { t=line; sub(/^[ \t]*/,"",t); sub(/[ \t]*$/,"",t); if (t==hdtag) inhd=0; next }
  hdnext=0; hdtagnext=""
  if (match(line, /<<-?[ \t]*['"'"'"]?[A-Za-z_][A-Za-z0-9_]*['"'"'"]?/)) {
    tag=substr(line, RSTART, RLENGTH); gsub(/^<<-?[ \t]*/,"",tag); gsub(/['"'"'"]/,"",tag)
    hdnext=1; hdtagnext=tag
  }
  s=line; sub(/^[ \t]*/,"",s)
  if (s ~ /^#/ || s=="") { if(hdnext){inhd=1;hdtag=hdtagnext}; next }
  while (s ~ /^(export|local|readonly|declare|typeset)[ \t]+/) { sub(/^[^ \t]+[ \t]+/,"",s) }
  while (s ~ /^-[a-zA-Z]+[ \t]+/) { sub(/^[^ \t]+[ \t]+/,"",s) }
  while (match(s, /^[A-Za-z_][A-Za-z0-9_]*=/)) {
    nm=substr(s, 1, RLENGTH-1); s=substr(s, RLENGTH+1); val=""
    if (substr(s,1,1)=="\"") {
      s=substr(s,2); i=index(s,"\""); if (i==0) { val=s; s="" } else { val=substr(s,1,i-1); s=substr(s,i+1) }
    } else if (substr(s,1,1)=="'"'"'") {
      s=substr(s,2); i=index(s,"'"'"'"); if (i==0) { val=s; s="" } else { val=substr(s,1,i-1); s=substr(s,i+1) }
    } else if (substr(s,1,2)=="${") {
      # A BRACED EXPANSION MAY CONTAIN SPACES, and stopping at the first one is
      # how `J=${SLURM_JOB_ID:?missing Slurm job id}` became the value
      # "${SLURM_JOB_ID:?missing" and every wrapper downstream of J came back
      # UNRESOLVED naming a word out of an error message. Consume to the
      # MATCHING brace, then any trailing unquoted run.
      depth=0; j=0
      for (k=1; k<=length(s); k++) {
        c=substr(s,k,1)
        if (c=="{") depth++
        else if (c=="}") { depth--; if (depth==0) { j=k; break } }
      }
      if (j>0) {
        val=substr(s,1,j); s=substr(s,j+1)
        if (match(s, /^[^ \t;|&)]+/)) { val=val substr(s,1,RLENGTH); s=substr(s,RLENGTH+1) }
      } else { val=s; s="" }
    } else {
      if (match(s, /[ \t;|&)]/)) { val=substr(s,1,RSTART-1); s=substr(s,RSTART) } else { val=s; s="" }
    }
    printf "%s\t%s\n", nm, val
    sub(/^[ \t]+/,"",s)
  }
  if (hdnext) { inhd=1; hdtag=hdtagnext }
}'

deg_join_continuations () { sed -e :a -e '/\\$/{N;s/\\\n[[:space:]]*/ /;ba}' "$1"; }

# every command-position token that is a PATH rather than a name
deg_scan_paths () {   # $1 = file
  deg_join_continuations "$1" | awk "$deg_scan_awk" \
    | grep -E '^["'"'"']?(/|\$\{?[A-Za-z_])' \
    | sort -u
}

# ---------------------------------------------------------------------------
# STATIC RESOLUTION.  Sources, in the order a running shell would apply them:
#   1. the wrapper's own assignments (and those of the pin/config files it
#      sources), last one winning;
#   2. for a self-referential `VAR=${VAR:-DEFAULT}`, the environment the SEAM
#      itself exports -- which is read from the seam, not typed here -- and
#      then DEFAULT;
#   3. nothing else. A name with no assignment and no seam value is UNRESOLVED.
# ---------------------------------------------------------------------------

# A resolver that fails says WHY on its own stdout, prefixed with this marker,
# because every call of it is a command substitution and a command substitution
# is a SUBSHELL: a global set inside one is gone by the time the caller reads
# it. The first version of this file set `DEG_UNRESOLVED_VAR` and the refusal
# rows all named the token instead of the variable that could not be resolved.
DEG_UNRESOLVED_MARK='__DEG_UNRESOLVED__'

# the assignment table for a file, including one level of `source`d files
deg_assign_table () {   # $1 = file ; writes VAR<TAB>VALUE rows to stdout
  local f=$1 line inc tbl
  tbl=$(deg_join_continuations "$f" | awk "$deg_assign_awk")
  # a sourced pin/config file's assignments come FIRST, as the shell would
  # apply them, and only when its path resolves from what we already have.
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    inc=$(DEG_ASSIGN_TBL_CACHE=$tbl deg_resolve_from_table "$tbl" "$line") || continue
    [ -f "$inc" ] || continue
    deg_join_continuations "$inc" | awk "$deg_assign_awk"
  done < <(deg_join_continuations "$f" \
             | sed -n -E 's/^[[:space:]]*(\.|source)[[:space:]]+"?([^"[:space:];]+)"?.*$/\2/p')
  printf '%s\n' "$tbl"
}

# the seam's own exported environment, as VAR<TAB>VALUE. It is asked of the
# seam (deg_static_env_pairs, defined in driver_exit_payload_shim.sh) rather
# than restated here, so a variable the seam stops exporting stops resolving.
deg_seam_env_table () {
  if command -v deg_static_env_pairs >/dev/null 2>&1; then
    deg_static_env_pairs | sed -e 's/=/\t/'
  fi
}

deg_table_lookup () {   # $1 = table text, $2 = name ; rc 1 when absent
  printf '%s\n' "$1" | awk -F'\t' -v n="$2" '$1==n{r=$2;f=1} END{ if(f){print r} else {exit 1} }'
}

# deg_resolve_from_table <table> <string> [<self-var>]
# echoes the fully resolved string; on rc 1 it echoes
# "<DEG_UNRESOLVED_MARK><what could not be resolved>" instead, so the caller --
# which is always a command substitution -- can name it in its refusal row.
deg_resolve_from_table () {
  local tbl=$1 s=$2 self=${3:-} i=0 pre var mod post hasdef def raw val
  while [ $i -lt 24 ]; do
    case $s in *'$'*) ;; *) printf '%s' "$s"; return 0;; esac
    i=$((i + 1))
    hasdef=0; def=""
    if [[ $s =~ ^([^\$]*)\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}(.*)$ ]]; then
      pre=${BASH_REMATCH[1]}; var=${BASH_REMATCH[2]}; mod=${BASH_REMATCH[3]}; post=${BASH_REMATCH[4]}
      case $mod in
        "")            ;;                                   # ${VAR}
        :-*|-*|:=*|=*) hasdef=1; def=${mod#:}; def=${def#[-=]} ;;  # ${VAR:-D} ${VAR-D} ${VAR:=D}
        :\?*|\?*)      ;;                                    # ${VAR:?msg}: no default, just loud
        *)             printf '%s%s' "$DEG_UNRESOLVED_MARK" "{$var$mod} (a parameter expansion this scanner does not evaluate)"; return 1 ;;
      esac
    elif [[ $s =~ ^([^\$]*)\$([A-Za-z_][A-Za-z0-9_]*)(.*)$ ]]; then
      pre=${BASH_REMATCH[1]}; var=${BASH_REMATCH[2]}; post=${BASH_REMATCH[3]}
    else
      # `$(`, `` ` ``, `$1`, `$@`, `$?` -- nothing static about any of them.
      printf '%s%s' "$DEG_UNRESOLVED_MARK" "a command substitution or a positional in '$s'"
      return 1
    fi
    val=""
    if [ "$var" != "$self" ] && raw=$(deg_table_lookup "$tbl" "$var"); then
      case $raw in
        *"\${$var"*|*"\$$var"*)
          val=$(deg_resolve_from_table "$tbl" "$raw" "$var") || { printf '%s' "$val"; return 1; } ;;
        *)
          val=$(deg_resolve_from_table "$tbl" "$raw" "$self") || { printf '%s' "$val"; return 1; } ;;
      esac
    elif raw=$(deg_table_lookup "$(deg_seam_env_table)" "$var"); then
      val=$raw
    elif [ "$hasdef" = 1 ]; then
      val=$(deg_resolve_from_table "$tbl" "$def" "$self") || { printf '%s' "$val"; return 1; }
    else
      printf '%s%s' "$DEG_UNRESOLVED_MARK" "$var (nothing in the wrapper or the seam assigns it)"
      return 1
    fi
    s="$pre$val$post"
  done
  printf '%s%s' "$DEG_UNRESOLVED_MARK" "$var (expansion did not settle in 24 rounds)"
  return 1
}

# CAN THE SEAM STUB THIS NAME?  Asked of the seam's shim directory when one has
# been built -- so `DEG_DROP_STUBS=cargo` removes by-path coverage for `cargo`
# exactly as it removes by-name coverage -- and of the seam's own name lists
# when the scanner is run standalone from the command line.
deg_seam_can_stub () {   # $1 = basename
  local n
  if [ -n "${DEG_SHIM:-}" ] && [ -d "${DEG_SHIM:-}" ]; then
    [ -x "$DEG_SHIM/$1" ]
    return $?
  fi
  for n in ${DEG_PAYLOAD_NAMES:-} ${DEG_NEUTRAL_NAMES:-} tee; do
    [ "$n" = "$1" ] && return 0
  done
  return 1
}

# deg_path_token_verdict <file> <token> -- one TAB-separated row:
#   COVERED    <basename> <resolved path>
#   UNCOVERED  <basename> <resolved path>
#   UNRESOLVED <variable>  <line number> <line text>
deg_path_token_verdict () {
  local f=$1 t=$2 tbl resolved base ln lntext
  t=${t#\"}; t=${t%\"}; t=${t#\'}; t=${t%\'}
  tbl=$(deg_assign_table "$f")
  if resolved=$(deg_resolve_from_table "$tbl" "$t"); then
    base=${resolved##*/}
    if deg_seam_can_stub "$base"; then
      printf 'COVERED\t%s\t%s\n' "$base" "$resolved"
    else
      printf 'UNCOVERED\t%s\t%s\n' "$base" "$resolved"
    fi
    return 0
  fi
  ln=$(grep -n -F -m1 -- "$2" "$f" | cut -d: -f1)
  lntext=$(sed -n "${ln:-1}p" "$f" | sed -e 's/^[[:space:]]*//' -e 's/\t/ /g')
  printf 'UNRESOLVED\t%s\t%s\t%s\n' "${resolved#"$DEG_UNRESOLVED_MARK"}" "${ln:-?}" "$lntext"
}

deg_scan_path_verdicts () {   # $1 = file
  local tokline
  while IFS= read -r tokline; do
    [ -n "$tokline" ] || continue
    deg_path_token_verdict "$1" "$tokline"
  done < <(deg_scan_paths "$1")
}

# the invocation shapes present, for the discovery table
deg_scan_payload_shape () {   # $1 = file
  local f=$1 out=""
  grep -Eq '(^|[^a-zA-Z0-9_/-])bash[[:space:]]+[^-]' "$f" && out="$out bash-driver"
  grep -Eq '(^|[^a-zA-Z0-9_/-])cargo[[:space:]]' "$f" && out="$out cargo"
  grep -Eq '(^|[^a-zA-Z0-9_/-])(srun|sbatch)[[:space:]]' "$f" && out="$out slurm"
  grep -Eq '(^|[^a-zA-Z0-9_/-])pixi[[:space:]]' "$f" && out="$out pixi"
  grep -Eq '(^|[^a-zA-Z0-9_/-])python3?[[:space:]]' "$f" && out="$out python"
  deg_scan_paths "$f" | grep -q . && out="$out path-binary"
  [ -n "$out" ] || out=" other"
  printf '%s\n' "${out# }"
}

# the last effective command word, and whether it re-raises
deg_scan_last () {   # $1 = file
  sed -e 's/#.*$//' "$1" | grep -v '^[[:space:]]*$' | tail -1 | sed 's/^[[:space:]]*//'
}
deg_scan_reraises () {   # $1 = file ; 0 = statically re-raises
  local last; last=$(deg_scan_last "$1")
  case "$last" in
    'exit 0'|exit) return 1 ;;
    exit\ *)  return 0 ;;
    exec\ *)  return 0 ;;
    *)        return 1 ;;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    paths)    deg_scan_paths "$2" ;;
    verdicts) deg_scan_path_verdicts "$2" ;;
    assigns)  deg_assign_table "$2" ;;
    payload)  deg_scan_payload_shape "$2" ;;
    last)     deg_scan_last "$2" ;;
    reraise)  deg_scan_reraises "$2" && echo reraise || echo swallow-static ;;
    *) echo "usage: $0 paths|verdicts|assigns|payload|last|reraise <file>" >&2; exit 2 ;;
  esac
fi
