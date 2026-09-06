#!/bin/bash
#SBATCH --job-name=LANE-STEP
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=node2352,node2348,node2424
#SBATCH --output=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/LANE-work/logs/slurm-%j.out
#
# lane_wrapper.sbatch -- THE VERSIONED SHAPE EVERY LANE SBATCH DERIVES FROM.
# STORE-REAP-2, and it exists for the same reason mutation_arms.sh does: the
# thing every lane hand-copies must live in one versioned file, or every lane
# re-inherits the last lane's bug.
#
# ── THE DEFECT IT EXISTS TO MAKE UNREPEATABLE ────────────────────────────────
#
# HARNESS-EXIT-1 found four jobs in ONE DAY that printed a refusal and reported
# `COMPLETED 0:0` to sacct. HARNESS-EXIT-2 fixed thirteen wrappers. And then
# STORE-REAP-2 wrote a FOURTEENTH by hand and reproduced it exactly: `sr2-mut`
# 5966771 crashed on an unbound variable with ZERO mutation arms run, printed
# `### MUT_EXIT=1`, and exited 0. The family idiom
#
#     bash "$D/<payload>.sh"
#     echo "### X_EXIT=$?"          # <-- captures the rc into a STRING
#
# ends the script on an `echo`, whose rc is 0. `sacct` state is not a success
# criterion for a script that ends on an echo, and the next lane to write one
# by hand will do it again -- so the shape is HERE now, and
# `tools/driver_exit_guard.sh` FAMILY B runs this file with its payload stubbed
# failing and requires a non-zero exit.
#
# ── THE FOUR RULES, and each is a line below ─────────────────────────────────
#   1. CAPTURE the rc into a VARIABLE on the line after the payload.
#   2. PRINT it, so a reader greps one row rather than reading sacct.
#   3. `exit "$rc"` -- the last statement, so Slurm gets it.
#   4. Refusals BEFORE the payload (harness pin, drift) exit non-zero too; a
#      wrapper that cannot establish its pin has not run its payload and must
#      not look like one that did.
#
#   usage: copy to <lane>-work/<step>.sbatch, replace LANE / LANE-STEP / the
#          PAYLOAD line, and state the run's PREDICTION in the echo above it.
#          Change nothing else about the epilogue.
set -u
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
D=$T/LANE-work

# --- rule 4: the pin, and its refusal is non-zero ----------------------------
HC=$(bash "$T/tools/harness_commit_resolve.sh" "$D") || { echo "### STOP: harness commit pin REFUSED"; exit 6; }
[ -n "$HC" ] || { echo "### STOP: no harness pin resolved"; exit 6; }
echo "### HARNESS_COMMIT=$HC"
bash "$T/tools/harness_drift_check.sh" "$HC" 2>&1 | tail -3
drc=${PIPESTATUS[0]}
echo "### DRIFT_RC=$drc"
[ "$drc" -eq 0 ] || { echo "### STOP: the task-dir harness is not $HC"; exit 6; }

# --- state the prediction BEFORE the run, in the file that runs it -----------
echo "### PREDICTED: <state the outcome here before submitting; a run with no"
echo "### prediction in its own submit line cannot be checked afterwards>"

# --- rules 1-3: the payload, and its rc reaches Slurm ------------------------
bash "$D/PAYLOAD.sh"
rc=$?
echo "### LANE_EXIT=$rc"
exit "$rc"
