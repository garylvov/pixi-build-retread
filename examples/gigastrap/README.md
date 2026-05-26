# gigastrap example workspace

Verbatim snapshot of gigastrap's workspace pixi.toml + isaac-pack
source package, copied here so retread changes can be iterated against
a real-world workload without destabilizing the upstream gigastrap
workspace.

## What's here

- `pixi.toml` — full workspace manifest, copied verbatim from
  `gigastrap/pixi.toml`. References many files that **don't exist in
  this repo**: post-install activation scripts, third_party
  submodules, editable path-deps. Solves will fail on missing paths
  until those are stubbed or commented out. Kept as-is on purpose so
  the cascade sees the same constraints the upstream workspace has.
- `isaac-pack/pixi.toml` — retread source package, verbatim.
- `patches/conda_pypi_map.json` — pywin32 + torch name mapping.

## The per-env numpy conflict (motivating the transitive constraint feature)

| env       | numpy constraint              | source                              |
|-----------|-------------------------------|-------------------------------------|
| `gsi`     | `==1.26.4`                    | `feature.gigastrap_sim_physx`       |
| `gsi-ros2`| `<2` (transitive)             | `ros-humble-joint-state-publisher`  |
| `gsn`     | `>2` (newton needs this)      | newton runtime requirements         |
| `gsf`     | follows torch 2.9.1 / JAX 0.7 | standalone, no shared features      |

gsi and gsi-ros2 agree (1.26 satisfies <2). gsn directly conflicts
with gsi/gsi-ros2 — there's no single numpy version that solves all
three envs at once. This is the case Option A (per-env wheel outputs)
exists to address; Option C (read transitive constraints from the
workspace's already-pinned conda deps) closes the gsi-ros2 vs
retread-emission gap but does not resolve the gsn vs gsi numpy split.

## Environments worth solving (retread-relevant)

- `pixi s -e gsi` — default isaaclab + gpu, no ros2; numpy 1.26.4
- `pixi s -e gsi-ros2` — adds ros2; numpy must stay <2 to keep
  robostack-humble's joint-state-publisher py3.11 (np126) builds
- `pixi s -e gsn` — Newton physics; numpy >2

## Sync back

When iterating retread alongside a corresponding gigastrap change,
edits propagate manually in both directions. There's no automation —
the copy is intentional so retread experiments can't destabilize
gigastrap's lock file.
