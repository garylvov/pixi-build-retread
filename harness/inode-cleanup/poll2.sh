#!/bin/bash
# one poll row: time, data+stellex inodes, live p6 job count, retread root count
L=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup/logs/poll2-20260902.tsv
inodes=$(/oscar/runtime/bin/checkquota 2>/dev/null | awk '/data\+stellex/{print $6}')
live=$(squeue -u glvov -h -o "%j" | grep -c '^p6-')
roots=$(ls -1 /oscar/data/stellex/glvov/retread 2>/dev/null | wc -l)
printf "%s\t%s\t%s\t%s\n" "$(date +%H:%M:%S)" "$inodes" "$live" "$roots" | tee -a "$L"
