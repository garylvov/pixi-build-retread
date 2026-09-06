#!/usr/bin/env bash
# driver_exit_scan.sh -- HARNESS-EXIT-3. The static half of the payload seam.
#
# The runtime seam in driver_exit_payload_shim.sh covers every command invoked
# BY NAME: a function shadows it, PATH cannot reach past the shim, and a name we
# did not stub lands in command_not_found_handle and refuses. The one thing PATH
# and functions cannot reach is a command invoked BY PATH -- `"$BIN" store-reap`
# or a literal `/oscar/.../some-binary`. This scanner finds exactly those, so a
# wrapper carrying one is REFUSED before it is ever started rather than being
# run and hoping.
#
# It is deliberately CONSERVATIVE in the safe direction: it over-reports (an
# awk program body, a continuation line) and every over-report costs one
# baselined UNCOVERED row, never an execution.
#
# It also reports the shapes the discovery table is built from.
#
# usage: driver_exit_scan.sh paths   <file>   # path-position tokens, one per line
#        driver_exit_scan.sh payload <file>   # payload invocation shape
#        driver_exit_scan.sh last    <file>   # last effective command word
#
# Variables a wrapper may legitimately use to name a payload binary; these are
# pointed at the payload stub by the shim, so they are COVERED, not uncovered.
DEG_SCAN_COVERED_VARS=${DEG_SCAN_COVERED_VARS:-"BIN PIXI RETREAD RETREAD_BIN PIXI_BIN BINARY EXE CARGO"}

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

deg_join_continuations () { sed -e :a -e '/\\$/{N;s/\\\n[[:space:]]*/ /;ba}' "$1"; }

# every command-position token that is a PATH rather than a name
deg_scan_paths () {   # $1 = file
  deg_join_continuations "$1" | awk "$deg_scan_awk" \
    | grep -E '^["'"'"']?(/|\$\{?[A-Za-z_])' \
    | sort -u
}

# a path token is COVERED when it is exactly a covered variable reference
deg_path_token_covered () {   # $1 = token
  local t=$1 v
  t=${t#\"}; t=${t%\"}
  for v in $DEG_SCAN_COVERED_VARS; do
    [ "$t" = "\$$v" ] && return 0
    [ "$t" = "\${$v}" ] && return 0
  done
  return 1
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
    paths)   deg_scan_paths "$2" ;;
    payload) deg_scan_payload_shape "$2" ;;
    last)    deg_scan_last "$2" ;;
    reraise) deg_scan_reraises "$2" && echo reraise || echo swallow-static ;;
    *) echo "usage: $0 paths|payload|last|reraise <file>" >&2; exit 2 ;;
  esac
fi
