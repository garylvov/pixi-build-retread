#!/usr/bin/env bash
# scripts/resolvo-ab.sh — resolvo-vs-BFS A/B oracle driver.
#
# Rebuilds the local retread backend via rebuild-local.sh (deploys hook-bearing
# backend to local-channel), nukes per-pack caches, runs each target pack with
# RETREAD_RESOLVO_DIFF set, concatenates per-pid JSONL shards into a rollup
# file, then prints a Python-generated summary table.
#
# USAGE
#   bash scripts/resolvo-ab.sh [--report-dir <dir>] [--packs <pack1,pack2,...>]
#
# TARGETS (default: isaac6, isaac51, genesis)
#   isaac6   — isaac-pack gpu variant       (expects GREEN on PyPI-resolved deps)
#   isaac51  — gigastrap / isaaclab-gpu     (expects GREEN on shared deps)
#   genesis  — genesis default env          (fires hook for pyopengl-accelerate etc.)
#
# VERDICT PRECEDENCE (rollup per entry)
#   RED > UNSOLVABLE-EXCLUDED > VERSION-DIFF > GREEN > SKIPPED
#   all-skips => SKIPPED (never GREEN) per the "genesis documents scope boundary" rule.
#
# EXIT CODE
#   0  — all entries GREEN or SKIPPED (or UNSOLVABLE-EXCLUDED)
#   1  — any RED or VERSION-DIFF

set -euo pipefail

# ── Toolchain ─────────────────────────────────────────────────────────────────
export PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH"
export RETREAD_TOOLS_PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin"
PIXI="/home/garylvov/.pixi/bin/pixi"

# ── Defaults ──────────────────────────────────────────────────────────────────
REPORT_DIR="${RETREAD_AB_REPORT_DIR:-/tmp/retread-ab-$$}"
PACKS="${RETREAD_AB_PACKS:-isaac6,isaac51,genesis}"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --packs)      PACKS="$2";      shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$REPORT_DIR"
ROLLUP="$REPORT_DIR/rollup.jsonl"
SUMMARY="$REPORT_DIR/summary.txt"

warn() { echo "[resolvo-ab] WARN: $*" >&2; }

echo "[resolvo-ab] Report dir: $REPORT_DIR"
echo "[resolvo-ab] Packs: $PACKS"

# ── Step 1: rebuild local backend ─────────────────────────────────────────────
echo "[resolvo-ab] Building retread via rebuild-local.sh ..."
bash scripts/rebuild-local.sh

# ── Step 2: pack -> (consumer dir, env) mapping ───────────────────────────────
declare -A CONSUMER_DIR
declare -A CONSUMER_ENV
CONSUMER_DIR[isaac6]="examples/isaac6"
CONSUMER_ENV[isaac6]="gpu"
CONSUMER_DIR[isaac51]="examples/gigastrap"
CONSUMER_ENV[isaac51]="isaaclab-gpu"
CONSUMER_DIR[genesis]="examples/genesis"
CONSUMER_ENV[genesis]="default"

# ── Step 3: run each target pack ─────────────────────────────────────────────
IFS=',' read -ra PACK_LIST <<< "$PACKS"

: > "$ROLLUP"  # create/truncate rollup once before the loop

for pack in "${PACK_LIST[@]}"; do
    cdir="${CONSUMER_DIR[$pack]:-}"
    cenv="${CONSUMER_ENV[$pack]:-}"
    if [[ -z "$cdir" || -z "$cenv" ]]; then
        echo "[resolvo-ab] ERROR: unknown pack '$pack' (not in CONSUMER_DIR/CONSUMER_ENV map)" >&2
        exit 1
    fi

    echo "[resolvo-ab] Running pack: $pack (dir=$cdir env=$cenv) ..."

    # Nuke retread caches (probe, repodata, build) and pixi consumer state.
    rm -rf \
        "$HOME/.cache/retread" \
        "$HOME/.cache/rattler/cache/retread-probes" \
        "$HOME/.cache/rattler/cache/retread-repodata" \
        "$HOME/.cache/rattler/cache/bld" \
        "$HOME/.cache/rattler/cache/backends-v0"/pixi-build-retread-* \
        "${cdir}/.pixi/envs" \
        "${cdir}/.pixi/bld" \
        "${cdir}/.pixi/meta-v0" \
        "${cdir}/.pixi/artifacts-v0" \
        2>/dev/null || true

    # Run the install with the oracle env var active.
    ( cd "$cdir" && \
        RETREAD_RESOLVO_DIFF="$REPORT_DIR/${pack}" \
        RETREAD_NO_REPLAY=1 \
        OMNI_KIT_ACCEPT_EULA=YES \
        "$PIXI" install -e "$cenv" \
    ) > "$REPORT_DIR/${pack}.log" 2>&1 \
    || warn "pack $pack exited non-zero (see ${pack}.log)"

    # Concatenate per-pid shards for this pack into the rollup.
    shopt -s nullglob
    shards=("$REPORT_DIR"/${pack}.*.jsonl)
    shopt -u nullglob
    if [ ${#shards[@]} -eq 0 ]; then
        echo "FATAL: pack $pack produced 0 shards — hook never fired; check backend deploy" >&2
        exit 1
    fi
    cat "${shards[@]}" >> "$ROLLUP"
    n=$(cat "${shards[@]}" | grep -c .)
    echo "[resolvo-ab] pack $pack: ${#shards[@]} shards, $n records"
done

# ── Step 4: Python rollup + summary ──────────────────────────────────────────
python3 - "$ROLLUP" "$SUMMARY" <<'EOF'
import sys, json, collections

rollup_path, summary_path = sys.argv[1], sys.argv[2]

# Verdict precedence (higher index = worse)
PRECEDENCE = ["SKIPPED", "GREEN", "VERSION-DIFF", "UNSOLVABLE-EXCLUDED", "RED"]


def precedence(v):
    try:
        return PRECEDENCE.index(v)
    except ValueError:
        return len(PRECEDENCE)  # unknown -> worst


# Read all records; key = (entry_name, target)
records = collections.defaultdict(list)
try:
    with open(rollup_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec.get("entry_name", "?"), rec.get("target", "?"))
            records[key].append(rec)
except FileNotFoundError:
    print(f"[resolvo-ab] No rollup file found at {rollup_path}", file=sys.stderr)
    sys.exit(1)

# Per-entry rollup: pick highest-precedence verdict.
# "all-skips => SKIPPED never GREEN" rule.
rolled = {}
for key, recs in sorted(records.items()):
    verdicts = [r.get("verdict", "RED") for r in recs]
    all_skipped = all(v == "SKIPPED" for v in verdicts)
    if all_skipped:
        rolled[key] = "SKIPPED"
    else:
        # Filter out SKIPPED when mixing with non-SKIPPED.
        active = [v for v in verdicts if v != "SKIPPED"]
        rolled[key] = max(active, key=precedence)

# Print summary table.
lines = []
lines.append(f"{'ENTRY':<40} {'TARGET':<20} {'VERDICT'}")
lines.append("-" * 75)
exit_code = 0
for (entry, target), verdict in sorted(rolled.items()):
    lines.append(f"{entry:<40} {target:<20} {verdict}")
    if verdict in ("RED", "VERSION-DIFF"):
        exit_code = 1

lines.append("")
lines.append(f"Total entries: {len(rolled)}")
green_count = sum(1 for v in rolled.values() if v == "GREEN")
lines.append(
    f"GREEN: {green_count}  SKIPPED: {sum(1 for v in rolled.values() if v == 'SKIPPED')}"
)
lines.append(
    f"RED: {sum(1 for v in rolled.values() if v == 'RED')}  VERSION-DIFF: {sum(1 for v in rolled.values() if v == 'VERSION-DIFF')}"
)

output = "\n".join(lines)
print(output)
with open(summary_path, "w") as f:
    f.write(output + "\n")

sys.exit(exit_code)
EOF

echo "[resolvo-ab] Summary written to $SUMMARY"
