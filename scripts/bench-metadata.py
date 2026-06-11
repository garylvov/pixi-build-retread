#!/usr/bin/env python3
"""Benchmark retread's conda/outputs metadata phase.

Drives the release binary over JSON-RPC against a source pack (default:
examples/isaac6/isaac-pack, warm wheel cache assumed), captures stderr,
and summarizes the `bench:` timing lines. Each invocation is a fresh
backend process, so in-process caches start cold -- exactly what a pixi
run pays after a backend upgrade.

Usage: python3 scripts/bench-metadata.py [source_dir] [label]
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "target/release/pixi-build-retread"
SRC = (Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "examples/isaac6/isaac-pack").resolve()
LABEL = sys.argv[2] if len(sys.argv) > 2 else "bench"

requests = [
    {"jsonrpc": "2.0", "id": 1, "method": "negotiateCapabilities",
     "params": {"capabilities": {}}},
    {"jsonrpc": "2.0", "id": 2, "method": "initialize",
     "params": {"manifestPath": str(SRC / "pixi.toml"),
                "sourceDirectory": str(SRC),
                "configuration": json.loads(subprocess.run(
                    ["python3", "-c",
                     "import tomllib,json,sys;"
                     f"d=tomllib.load(open('{SRC}/pixi.toml','rb'));"
                     "print(json.dumps(d['package']['build']['config']))"],
                    capture_output=True, text=True, check=True).stdout)}},
    {"jsonrpc": "2.0", "id": 3, "method": "conda/outputs",
     "params": {"hostPlatform": "linux-64", "buildPlatform": "linux-64",
                "channels": ["https://prefix.dev/conda-forge"],
                "workDirectory": "/tmp/retread-bench-work"}},
]

t0 = time.monotonic()
proc = subprocess.run(
    [str(BIN)],
    input="".join(json.dumps(r) + "\n" for r in requests),
    capture_output=True, text=True,
    env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
         "PIXI_BUILD_RETREAD_LOG": "info"},
)
wall = time.monotonic() - t0

for line in proc.stdout.splitlines():
    r = json.loads(line)
    if r.get("id") == 3 and "error" in r:
        print("conda/outputs ERROR:", r["error"]["message"][:300])
        sys.exit(1)

print(f"== {LABEL}: wall {wall:.1f}s ==")
bench_re = re.compile(r"bench: (.+?)(?:\x1b\[\d*m)?\s*$")
field_re = re.compile(r"(\w+)=((?:\x1b\[\d*m)?[\w./:-]+)")
totals = {}
for line in proc.stderr.splitlines():
    if "bench:" not in line:
        continue
    clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
    msg = clean.split("bench:")[1].strip()
    fields = dict(re.findall(r"(\w+)=([\w./:-]+)", clean))
    ms = fields.get("elapsed_ms") or fields.get("parse_ms")
    print(f"  {msg}")
    key = msg.split(" ")[0] + ("/" + fields.get("subdir", "") if "subdir" in fields else "")
    if ms:
        totals.setdefault(msg.split("(")[0].strip(), []).append(int(ms))
print("-- rollup (ms) --")
for k, v in totals.items():
    print(f"  {k}: total={sum(v)} n={len(v)}")
