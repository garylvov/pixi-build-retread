# Isaac Sim 6.0 example

Minimal workspace that repacks `isaacsim[all,extscache]==6.0.0.1` as a
single conda package via pixi-build-retread. Counterpart to
`examples/gigastrap/` (Isaac Sim 5.1), kept as the regression workspace
for the Isaac Sim 6 issues found 2026-06-10.

## Run it

```bash
# from the repo root: build the local backend artifact first
bash scripts/rebuild-local.sh

cd examples/isaac6
pixi lock
```

The first run downloads the Isaac Sim 6 wheel set into
`isaac-pack/wheels/` (~8 GB). Routing diagnostics land in
`isaac-pack/retread-probe-trace-isaac-pack-6.json`; a
`RETREAD-SOLVE-FAILED-isaac-pack-6.md` appears only when an env is
genuinely unsolvable.

## What this example pinned down

1. **Isaac Sim 6.0 wheels are cp312-only** (5.1 was cp311). A workspace
   pinned to `python ==3.11` fails at wheel resolution with
   "no wheel for isaacsim ==6.0.0.1 ... matches target python=3.11"
   before any dependency routing happens. This workspace pins 3.12 and
   sets `build-variants = { python = ["3.12"] }`.
2. **The name-skew drop leak (fixed v1.4.x)**: isaacsim 6 requires
   `tinyobjloader`, whose conda emission name_maps to
   `tinyobjloader-python` — a package with zero conda-forge candidates.
   The cascade correctly bundled the wheel from PyPI and recorded the
   drop under the PyPI name, but the emission filters only matched the
   conda name, so the doomed conda run-dep shipped anyway and the solve
   died with "No candidates were found for tinyobjloader-python". The
   drop/vendored/conda-deps filters now match on either namespace.
3. This pack uses `retread-bundle = "isaac-pack-6"` (v1.4.0) instead of
   a per-entry `bundle` field.
