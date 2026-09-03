#!/bin/bash
# poll.sh <iterations>  -- foreground poll; one sample every 300s.
HERE=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup
JOBS="5624707,5627563,5631958,5631959,5631960,5631961,5631962,5631963,5631964,5631965,5631966,5631967,5631968,5631969,5631970,5631971,5631972,5631973,5631996,5631997,5631998,5637105"
N="${1:-2}"
for i in $(seq 1 "$N"); do
  echo "===== $(date '+%F %H:%M:%S %Z')"
  sacct -j "$JOBS" -X -n -P --format=JobID,JobName,State,Elapsed | awk -F'|' '{printf "%s %s %s %s\n",$1,$2,$3,$4}' | sort -k3
  echo "-- quota: $(/oscar/runtime/bin/checkquota 2>/dev/null | awk '/data\+stellex/{print $6, $9, $10}')"
  echo "-- retread cert*/ws.* roots left: $(ls -1d /oscar/data/stellex/glvov/retread/cert* /oscar/data/stellex/glvov/retread/ws.* 2>/dev/null | wc -l)"
  echo "$(date +%s)	$(/oscar/runtime/bin/checkquota 2>/dev/null | awk '/data\+stellex/{print $6}')	$(ls -1d /oscar/data/stellex/glvov/retread/cert* /oscar/data/stellex/glvov/retread/ws.* 2>/dev/null | wc -l)" >> "$HERE/logs/poll-20260902.tsv"
  nonterm=$(sacct -j "$JOBS" -X -n -P --format=State | grep -cE 'RUNNING|PENDING|SUSPENDED|REQUEUED|CONFIGURING')
  echo "-- non-terminal jobs: $nonterm"
  [ "$nonterm" = "0" ] && { echo "ALL TERMINAL"; break; }
  [ "$i" -lt "$N" ] && sleep 280
done
exit 0
