#!/usr/bin/env bash
# wedge_triage.sh <pid> [<jobid>]  --  READ-ONLY liveness triage. Never signals anything.
#
# Why this exists (p6l-1, 2026-09-03). On node2347 a 44m53s site NFS lock-manager
# outage (`lockd: server hpcnfs.ccv.brown.edu not responding` 12:54:22 -> `OK`
# 13:39:15) froze a cert install log for 45 minutes, parked one thread in
# `rpc_wait_bit_killable`, and left the pixi->backend RPC stdin pipe in
# `pipe_read`. Every symptom read as a wedge; the job resumed by itself and
# finished. A SIGTERM there would have destroyed a live 60-minute install five
# minutes before it landed. On 2026-09-02 the SAME two symptoms (D-state in
# rpc_wait) were a real wedge and a kill WAS right. The states are ambiguous, so
# the tie-breakers must be consulted before the word WEDGED is allowed out.
#
# Verdicts, in order; the script STOPS at the first that applies:
#
#   (a) NFS-LOCK-OUTAGE (since <time>)   the node's lock service is down; WAIT.
#   (b) IDLE-ON-RPC-CHANNEL (...)        a pipe_read whose write end is held by a
#                                        live ancestor/child -- an idle RPC
#                                        channel by design, not an orphan.
#   (d) WEDGED (flat <N>s; wchan=<top>)  counters AND output flat across two
#                                        samples with neither (a) nor (b) firing.
#   (else) ADVANCING
#
# (c) is printed between (b) and (d) and never decides: /proc/locks rows for the
# pid family with READ/WRITE, because a SHARED (READ) clone lease is retread's
# documented design (`source_build.rs`, `process_clone_locks` /
# `PendingOsLock::downgrade_and_commit`) and is not contention.
#
# Exit codes: 0 ADVANCING, 10 NFS-LOCK-OUTAGE, 11 IDLE-ON-RPC-CHANNEL, 12 WEDGED,
#             2 usage / pid gone.
#
# Fixture knobs (used by wedge_triage_guard.sh; leave unset in production):
#   WT_DMESG_CMD    command that emits `dmesg -T` output      (default: dmesg -T)
#   WT_SAMPLE_SECS  seconds between the two samples           (default: 120)
#   WT_ARTIFACT_DIR artifact dir, instead of scontrol WorkDir (default: unset)
set -uo pipefail

PID=${1:-}
JOBID=${2:-}
DMESG_CMD=${WT_DMESG_CMD:-dmesg -T}
SAMPLE_SECS=${WT_SAMPLE_SECS:-120}
ART=${WT_ARTIFACT_DIR:-}

say() { printf '%s\n' "$*"; }
hdr() { printf '\n== %s\n' "$*"; }

if [ -z "$PID" ]; then
  say "usage: wedge_triage.sh <pid> [<jobid>]"; exit 2
fi
if [ ! -d "/proc/$PID" ]; then
  say "pid $PID is not alive (no /proc/$PID) -- nothing to triage."; exit 2
fi

comm_of()  { cat "/proc/$1/comm" 2>/dev/null || echo '?'; }
# /proc/<pid>/stat: comm can contain spaces and parens, so cut at the LAST ')'.
# After the cut, field 1 = state, 2 = ppid, 12 = utime, 13 = stime.
statf()    { local s f; s=$(cat "/proc/$1/stat" 2>/dev/null) || return 1; f=${s##*') '}; printf '%s\n' "$f"; }
state_of() { local f; f=$(statf "$1") || { echo '?'; return; }; set -- $f; echo "${1:-?}"; }
ppid_of()  { local f; f=$(statf "$1") || { echo 0; return; }; set -- $f; echo "${2:-0}"; }
cpu_of()   { local f; f=$(statf "$1") || { echo 0; return; }; set -- $f; echo $(( ${12:-0} + ${13:-0} )); }
io_of()    { awk '/^rchar:|^wchar:/ { s += $2 } END { print s+0 }' "/proc/$1/io" 2>/dev/null || echo 0; }
alive()    { [ -d "/proc/$1" ] && [ "$(state_of "$1")" != "Z" ]; }

# The pid family: the pid plus every descendant. ONE pass over /proc builds the
# ppid map -- a per-level rescan of /proc costs minutes on a busy login node and
# a triage tool that takes minutes to answer will not be run when it matters.
family() {
  local -A PP=() KIDMAP=()
  local p f
  for p in /proc/[0-9]*; do
    p=${p#/proc/}
    f=$(statf "$p") || continue
    set -- $f
    PP[$p]=${2:-0}
    KIDMAP[${2:-0}]="${KIDMAP[${2:-0}]:-} $p"
  done
  local out="$PID" frontier="$PID" next k
  while [ -n "${frontier// /}" ]; do
    next=""
    for p in $frontier; do
      for k in ${KIDMAP[$p]:-}; do
        case " $out " in *" $k "*) ;; *) out="$out $k"; next="$next $k";; esac
      done
    done
    frontier=$next
  done
  printf '%s\n' "$out"
}

if [ -z "$ART" ] && [ -n "$JOBID" ]; then
  ART=$(scontrol show job "$JOBID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^WorkDir=//p' | head -1)
fi
newest_file() {
  [ -n "$ART" ] && [ -d "$ART" ] || return 1
  local n; n=$(ls -t "$ART" 2>/dev/null | head -1); [ -n "$n" ] || return 1
  printf '%s/%s' "$ART" "$n"
}

say "wedge_triage.sh  pid=$PID${JOBID:+  jobid=$JOBID}  host=$(hostname -s)  $(date '+%F %T %Z')"
say "pid $PID: $(comm_of "$PID")  state=$(state_of "$PID")  ppid=$(ppid_of "$PID")"
say "READ-ONLY: this script signals nothing."

# ---------------------------------------------------------------- (a) lockd
hdr "(a) lock service -- dmesg -T | grep -i lockd | tail -3"
DM=$($DMESG_CMD 2>/dev/null)
DMRC=$?
LOCKD=$(printf '%s\n' "$DM" | grep -i lockd)
if [ $DMRC -ne 0 ] && [ -z "$DM" ]; then
  say "dmesg unreadable here (rc=$DMRC). A lock outage CANNOT be ruled out from this host --"
  say "re-run on the job's node before believing any WEDGED verdict below."
elif [ -z "$LOCKD" ]; then
  say "(no lockd lines in the ring buffer)"
else
  printf '%s\n' "$LOCKD" | tail -3
  LAST=$(printf '%s\n' "$LOCKD" | tail -1)
  if printf '%s' "$LAST" | grep -qi 'not responding'; then
    WHEN=$(printf '%s' "$LAST" | sed -n 's/^\[\([^]]*\)\].*/\1/p')
    [ -z "$WHEN" ] && WHEN="unknown time"
    hdr "(a) nfs/rpc tail -- dmesg | grep -iE 'nfs|rpc' | tail -3"
    printf '%s\n' "$DM" | grep -iE 'nfs|rpc' | tail -3
    hdr "VERDICT"
    say "NFS-LOCK-OUTAGE (since $WHEN)"
    say "The node's NLM lock daemon is not answering and has not come back."
    say "flock/fcntl on /oscar blocks; threads park in rpc_wait_bit_killable and logs freeze."
    say "ACTION: WAIT. The mount is hard; the process resumes when lockd returns."
    say "Do NOT kill pid $PID for this."
    exit 10
  fi
  say "last lockd line is not an open outage -- lock service is up."
fi
hdr "(a) nfs/rpc tail -- dmesg | grep -iE 'nfs|rpc' | tail -3"
if [ -n "$DM" ]; then printf '%s\n' "$DM" | grep -iE 'nfs|rpc' | tail -3; else say "(dmesg unreadable)"; fi

# (a2) uv's per-sdist flock on the SHARED persistent uv cache lives on the same
# NFS, so an NLM outage turns an ordinary serialization wait into a 1 h timeout
# (hlgd-proof 5688009 died exactly this way on 2026-09-03: `Timeout (3600s) when
# waiting for lock on .../uv-cache/sdists-v9/path/...` while building
# `protomotions` from a path source). lockd is SERVER-side: when hpcnfs stops
# answering it stalls locks on EVERY node, so the outage that explains a uv lock
# timeout may be visible from a different host than the job's.
NF=$(newest_file) && [ -f "$NF" ] && if tail -40 "$NF" 2>/dev/null | grep -qiE 'waiting for lock|Timeout \([0-9]+s\) when waiting'; then
  hdr "(a2) the log's last rows are a uv lock wait -- lockd view (server-side; any node sees it)"
  tail -3 "$NF" | sed 's/^/    /'
  if [ -n "$LOCKD" ]; then printf '%s\n' "$LOCKD" | tail -5 | sed 's/^/    /'; else say "    (no lockd lines on THIS host -- check the job's own node too)"; fi
  say "    uv takes a per-sdist flock on the shared cache: concurrent jobs building the SAME"
  say "    path source (protomotions, pace-sim2real) serialize by design. Under an NLM outage"
  say "    that ordinary wait becomes a 3600 s timeout and the job dies. Not a retread defect."
fi

# ------------------------------------------------------- (b) pipe fd holders
hdr "(b) pipe fds of pid $PID and its threads, resolved to the holder of the other end"

PIPE_THREADS=""
for t in "/proc/$PID/task/"*; do
  [ -d "$t" ] || continue
  w=$(cat "$t/wchan" 2>/dev/null)
  [ "$w" = "pipe_read" ] && PIPE_THREADS="$PIPE_THREADS ${t##*/}"
done
if [ -n "$PIPE_THREADS" ]; then
  say "threads in pipe_read:$PIPE_THREADS"
else
  say "no thread of pid $PID is in pipe_read."
fi

# our fds -> pipe inodes
MY_PIPES=""
for f in "/proc/$PID/fd/"*; do
  [ -e "$f" ] || continue
  l=$(readlink "$f" 2>/dev/null) || continue
  case "$l" in pipe:\[*\])
    ino=${l#pipe:[}; ino=${ino%]}
    say "  fd ${f##*/} -> $l"
    MY_PIPES="$MY_PIPES $ino"
  ;; esac
done
[ -z "$MY_PIPES" ] && say "  (pid $PID holds no pipe fds)"

# ancestors and children of PID
ANC=""; p=$(ppid_of "$PID")
while [ "$p" != "0" ] && [ "$p" != "1" ] && [ -n "$p" ]; do ANC="$ANC $p"; p=$(ppid_of "$p"); done
KIDS=""
for q in /proc/[0-9]*; do
  q=${q#/proc/}; kf=$(statf "$q") || continue; set -- $kf
  [ "${2:-0}" = "$PID" ] && KIDS="$KIDS $q"
done
say "ancestors:${ANC:- none}   children:${KIDS:- none}"

# ONE pass over every readable /proc/<pid>/fd, matching against the set of
# inodes we care about. Rescanning /proc once per pipe fd is minutes on a busy
# node; that cost is why this check would otherwise get skipped.
RPC_HOLDER=""; RPC_INO=""
if [ -n "${MY_PIPES// /}" ]; then
  for q in /proc/[0-9]*; do
    q=${q#/proc/}
    [ "$q" = "$PID" ] && continue
    [ -r "/proc/$q/fd" ] || continue
    for f in "/proc/$q/fd/"*; do
      [ -e "$f" ] || continue
      l=$(readlink "$f" 2>/dev/null) || continue
      case "$l" in pipe:\[*\]) ;; *) continue;; esac
      ino=${l#pipe:[}; ino=${ino%]}
      case " $MY_PIPES " in *" $ino "*) ;; *) continue;; esac
      rel="other"
      case " $ANC " in *" $q "*) rel="LIVE ANCESTOR";; esac
      case " $KIDS " in *" $q "*) rel="LIVE CHILD";; esac
      say "  pipe:[$ino] also held by pid $q ($(comm_of "$q")) state=$(state_of "$q") fd=${f##*/}  [$rel]"
      if [ -n "$PIPE_THREADS" ] && [ "$rel" != "other" ] && alive "$q" && [ -z "$RPC_HOLDER" ]; then
        RPC_HOLDER="$q"; RPC_INO="$ino"
      fi
    done
  done
fi

if [ -n "$RPC_HOLDER" ]; then
  hdr "VERDICT"
  say "IDLE-ON-RPC-CHANNEL (write end held by $RPC_HOLDER $(comm_of "$RPC_HOLDER"))"
  say "pid $PID is blocked in read() on pipe:[$RPC_INO], whose other end is held by a live"
  say "relative -- the pixi->backend build-RPC channel with no request in flight. A backend"
  say "with nothing to do is SUPPOSED to sit here. Not orphaned, not leaked, not wedged."
  say "ACTION: look at the PARENT's progress, not this pid. Do NOT kill pid $PID for this."
  exit 11
fi

# ------------------------------------------------------------- (c) /proc/locks
hdr "(c) /proc/locks rows for the pid family (READ = shared lease, not contention)"
FAM=$(family)
say "family: $FAM"
FOUND=0
while read -r line; do
  [ -n "$line" ] || continue
  lp=$(printf '%s' "$line" | awk '{ for (i=1;i<=NF;i++) if ($i ~ /^(READ|WRITE)$/) { print $(i+1); exit } }')
  case " $FAM " in *" $lp "*)
    kind=$(printf '%s' "$line" | grep -o '\<\(READ\|WRITE\)\>')
    blk=$(printf '%s' "$line" | grep -q '^[0-9]*: *->' && echo "BLOCKED (waiting)" || echo "held")
    say "  [$kind, $blk] pid $lp ($(comm_of "$lp")): $line"
    FOUND=1
  ;; esac
done < <(cat /proc/locks 2>/dev/null)
[ $FOUND -eq 0 ] && say "  (no /proc/locks rows for this family)"
say "  READ rows are SHARED clone leases held for process lifetime by design and block nobody."

# ---------------------------------------------------- (d) two samples, N secs
hdr "(d) two samples ${SAMPLE_SECS}s apart -- cpu, io, wchan histogram, artifact growth"

newest_size() {
  [ -n "$ART" ] && [ -d "$ART" ] || { echo "-"; return; }
  local n; n=$(ls -t "$ART" 2>/dev/null | head -1)
  [ -n "$n" ] || { echo "-"; return; }
  printf '%s:%s' "$n" "$(stat -c %s "$ART/$n" 2>/dev/null || echo -)"
}
sample() {
  local cpu=0 io=0 p
  for p in $(family); do cpu=$(( cpu + $(cpu_of "$p") )); io=$(( io + $(io_of "$p") )); done
  printf '%s %s %s\n' "$cpu" "$io" "$(newest_size)"
}
wchan_hist() {
  local p t out=""
  for p in $(family); do
    for t in "/proc/$p/task/"*; do [ -d "$t" ] || continue; out="$out$(cat "$t/wchan" 2>/dev/null)\n"; done
  done
  printf "$out" | sed '/^$/d' | sort | uniq -c | sort -rn
}
say "artifact dir: ${ART:-<none resolved>}"
S1=$(sample); H1=$(wchan_hist)
say "sample 1: cpu_ticks+io_bytes+newest = $S1"
printf '%s\n' "$H1" | head -8 | sed 's/^/    /'
sleep "$SAMPLE_SECS"
if [ ! -d "/proc/$PID" ]; then
  hdr "VERDICT"; say "ADVANCING (pid $PID exited during the ${SAMPLE_SECS}s window -- it was working)"; exit 0
fi
S2=$(sample); H2=$(wchan_hist)
say "sample 2: cpu_ticks+io_bytes+newest = $S2"
printf '%s\n' "$H2" | head -8 | sed 's/^/    /'

TOP=$(printf '%s\n' "$H2" | head -1 | awk '{print $2}')
hdr "VERDICT"
if [ "$S1" = "$S2" ]; then
  say "WEDGED (flat ${SAMPLE_SECS}s; wchan=${TOP:-unknown})"
  say "Neither (a) nor (b) explains it: the lock service is up and no pipe_read is an idle"
  say "RPC channel. cpu, io and the newest artifact file are all unchanged across ${SAMPLE_SECS}s."
  say "If the operator/agent decides to act, the line is exactly:"
  say "    kill -TERM $PID"
  say "(this script did NOT run it, and never will)"
  exit 12
fi
say "ADVANCING"
say "sample 1: $S1"
say "sample 2: $S2"
say "Something moved. By law 13 output growth decides -- do not kill."
exit 0
