# `manifests/a3b/` — pack-manifest cessions used by the A3b family of arms

These are `patch(1)` diffs against `pypi-packs/<pack>/pixi.toml` in
`imprint-data`. They are versioned here because the task tree
(`agrescap/tasks/retread-4-11/a3b-work/`) is not a git repository and anything
that lives only there is one `rm` from gone (CLAUDE.md law 7). Boarded as
C29-5 in `LANE-C-WARM-LOG.md` §29.

| file | md5 | what it is | produced by |
| --- | --- | --- | --- |
| `isaaclab-2.3x-pack.pixi.toml.a3b2only.diff` | `5c116bc032f62e38e1a822b44284ef06` | The a3b2 cession REBASED onto the CANONICAL `isaaclab-2.3x-pack/pixi.toml` — the standalone form, applying with no a3b prerequisite. Its one functional line adds `typing-extensions` to `retread-drop-deps`; everything else it adds is comment. `patch -p0` on canonical md5 `74d3aaa6374154be07b42d37fe0b3f32` yields `25e00bb4f886c7de4e37bfd24d21df53` (173 lines). | `LANE-C-WARM-LOG.md` §29.2 (C29/B5, job 5831726); apply matrix re-measured and recorded in `ACCEPTANCE-PACKET.md` §12.16 |

The unrebased sibling, `a3b-work/isaaclab-2.3x-pack.pixi.toml.a3b2.diff`
(md5 `11ab83a65d10e7250db497836a621126`), is deliberately NOT here: it was cut
against the a3b-PATCHED pack and fails `Hunk #1 FAILED at 60` on the canonical
one. See ACCEPTANCE-PACKET §12.16 for the full apply matrix and C29-4 for its
retirement.
