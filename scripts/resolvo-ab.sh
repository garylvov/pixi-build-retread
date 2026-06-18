#!/usr/bin/env bash
# scripts/resolvo-ab.sh — resolvo-vs-BFS A/B oracle driver.
#
# Rebuilds the local retread backend, nukes per-pack caches, runs each target
# pack with RETREAD_RESOLVO_DIFF set, concatenates per-pid JSONL shards into a
# rollup file, then prints a Python-generated summary table.
#
# USAGE
#   bash scripts/resolvo-ab.sh [--report-dir <dir>] [--packs <pack1,pack2,...>]
#
# TARGETS (default: isaac6, isaac51, genesis)
#   isaac6   — isaac-pack gpu variant       (expects GREEN on PyPI-resolved deps)
#   isaac51  — gigastrap / isaaclab-gpu     (expects GREEN on shared deps)
#   genesis  — genesis default env          (EXPECTED all-SKIPPED: source-form)
#
# VERDICT PRECEDENCE (rollup per entry)
#   RED > UNSOLVABLE-CONFLICT > UNSOLVABLE-EXCLUDED > VERSION-DIFF > GREEN > SKIPPED
#   all-skips => SKIPPED (never GREEN) per the "genesis documents scope boundary" rule.
#
# EXIT CODE
#   0  — all entries GREEN or SKIPPED (or UNSOLVABLE-EXCLUDED)
#   1  — any RED or VERSION-DIFF

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
REPORT_DIR="${RETREAD_AB_REPORT_DIR:-/tmp/retread-ab-$$}"
PACKS="${RETREAD_AB_PACKS:-isaac6,isaac51,genesis}"
RETREAD_BIN="${RETREAD_BIN:-./target/release/pixi-build-retread}"

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

echo "[resolvo-ab] Report dir: $REPORT_DIR"
echo "[resolvo-ab] Packs: $PACKS"

# ── Step 1: rebuild local backend ─────────────────────────────────────────────
echo "[resolvo-ab] Building retread (release)..."
PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" \
    cargo build --release 2>&1

# ── Step 2: nuke per-pack caches ─────────────────────────────────────────────
CACHE_BASE="${RETREAD_CACHE_DIR:-$HOME/.cache/pixi-build-retread}"
echo "[resolvo-ab] Clearing caches under $CACHE_BASE ..."
rm -rf "$CACHE_BASE" 2>/dev/null || true

# ── Step 3: run each target pack ─────────────────────────────────────────────
IFS=',' read -ra PACK_LIST <<< "$PACKS"

for pack in "${PACK_LIST[@]}"; do
    echo "[resolvo-ab] Running pack: $pack ..."
    SHARD_BASE="$REPORT_DIR/${pack}"

    # Set RETREAD_RESOLVO_DIFF to the shard base path; the backend appends .<pid>.jsonl.
    RETREAD_RESOLVO_DIFF="$SHARD_BASE" \
    RETREAD_NO_REPLAY=1 \
        pixi run --manifest-path "examples/${pack}/pixi.toml" \
        retread-build 2>&1 | tee "$REPORT_DIR/${pack}.log" || {
        echo "[resolvo-ab] WARN: pack ${pack} exited non-zero (may be expected for genesis)" >&2
    }

    # Concatenate all per-pid shards for this pack into the rollup.
    for shard in "$REPORT_DIR"/${pack}.*.jsonl; do
        [[ -f "$shard" ]] && cat "$shard" >> "$ROLLUP" || true
    done
done

# ── Step 4: Python rollup + summary ──────────────────────────────────────────
python3 - "$ROLLUP" "$SUMMARY" <<'EOF'
import sys, json, collections

rollup_path, summary_path = sys.argv[1], sys.argv[2]

# Verdict precedence (higher index = worse)
PRECEDENCE = ["SKIPPED", "GREEN", "VERSION-DIFF", "UNSOLVABLE-EXCLUDED",
              "UNSOLVABLE-CONFLICT", "RED"]

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
lines.append(f"GREEN: {green_count}  SKIPPED: {sum(1 for v in rolled.values() if v == 'SKIPPED')}")
lines.append(f"RED: {sum(1 for v in rolled.values() if v == 'RED')}  VERSION-DIFF: {sum(1 for v in rolled.values() if v == 'VERSION-DIFF')}")

output = "\n".join(lines)
print(output)
with open(summary_path, "w") as f:
    f.write(output + "\n")

sys.exit(exit_code)
EOF

echo "[resolvo-ab] Summary written to $SUMMARY"
