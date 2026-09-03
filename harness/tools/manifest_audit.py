#!/usr/bin/env python3
"""Classify every manifest dependency line by REMOVABILITY EVIDENCE.

Read-only. Emits candidates; deletes nothing. A pin is only defensible if
something testifies for it — an INTENT comment, a pack audit edge, or a
conda-provider fact. Everything else is a candidate for delete-first.

Classes:
  INTENT-PINNED   an adjacent `# INTENT:` comment explains the pin -> KEEP
  PACK-EDGE       a pack audit/lock names this distribution -> pin may be
                  redundant with the pack's own metadata; CANDIDATE
  IMPORT-BACKED   some source tree actually imports it (needs import_audit)
  ORPHAN          no INTENT, no pack edge, no import -> STRONGEST CANDIDATE
"""
import json, re, sys, glob
from pathlib import Path
from collections import defaultdict

MAN = Path(sys.argv[1]); PACKS = Path(sys.argv[2])

# --- collect every name any pack audit/lock knows about -------------------
pack_names, pack_of = set(), defaultdict(set)
drop_dep_names = set()
for t in PACKS.glob("*/pixi.toml"):
    try: txt = t.read_text()
    except Exception: continue
    for m in re.finditer(r'retread-drop-deps\s*=\s*\[([^\]]*)\]', txt):
        for tok in m.group(1).split(","):
            tok = tok.strip().strip('"\'')
            if tok: drop_dep_names.add(tok.lower().replace("_", "-"))

for a in PACKS.glob("*/retread-audit-*.json"):
    try: d = json.loads(a.read_text())
    except Exception: continue
    for w in d.get("wheels", []):
        n = (w.get("name") or "").lower().replace("_", "-")
        if n: pack_names.add(n); pack_of[n].add(a.parent.name)

# --- walk the manifest, tracking which [section] we are in ----------------
DEP = re.compile(r'^([A-Za-z0-9_.\-]+)\s*=\s*(.+?)\s*$')
sec = None; rows = []; prev_comment = ""
for i, line in enumerate(MAN.read_text().splitlines(), 1):
    s = line.strip()
    if s.startswith("["): sec = s.strip("[]"); prev_comment = ""; continue
    if s.startswith("#"): prev_comment = (prev_comment + " " + s).strip(); continue
    m = DEP.match(s)
    if not m or sec is None or "dependencies" not in (sec or ""):
        if s: prev_comment = ""
        continue
    name, spec = m.group(1), m.group(2)
    key = name.lower().replace("_", "-")
    exact = bool(re.match(r'^"?==', spec))
    # TESTIMONY comes in THREE spellings, not one. Keying only on "intent"
    # produced false ORPHANs for both `# retread:pin` (9 in the manifest, a
    # machine-readable marker) and for pins paired with a pack `retread-drop-deps`
    # entry (e.g. prettytable: the hover pack DROPS IsaacLab's <3.4 so the root
    # feature's ==3.18.0 is the deliberate provider). Deleting either class is a
    # regression, not a cleanup.
    inline = spec  # the raw RHS still carries any trailing comment
    intent = ("intent" in prev_comment.lower()
              or "retread:pin" in inline.lower()
              or "retread:pin" in prev_comment.lower())
    in_pack = key in pack_names
    # STRUCTURAL pins are the interpreter/platform/toolchain contract. They are
    # NOT removal candidates and labelling them ORPHAN is a false positive that
    # would send someone to delete `python = "==3.11"`. Measured: 22 of the 34
    # first-pass "orphans" were structural.
    STRUCTURAL = {"python", "cuda-version", "pip", "setuptools", "wheel", "toml"}
    STRUCT_PREFIX = ("cuda-", "sysroot", "gcc", "gxx", "binutils", "cxx-", "libstdcxx")
    structural = key in STRUCTURAL or key.startswith(STRUCT_PREFIX)
    if key in drop_dep_names: cls = "DROP-DEP-PROVIDER"
    elif structural:      cls = "STRUCTURAL"
    elif intent:          cls = "INTENT-PINNED"
    elif in_pack:         cls = "PACK-EDGE"
    else:                 cls = "ORPHAN"
    rows.append(dict(line=i, section=sec, name=name, spec=spec[:48],
                     exact=exact, cls=cls, packs=sorted(pack_of.get(key, []))[:2]))
    prev_comment = ""

exact_rows = [r for r in rows if r["exact"]]
print(f"### manifest dep lines: {len(rows)}   exact-pinned: {len(exact_rows)}")
for c in ("ORPHAN", "PACK-EDGE", "INTENT-PINNED", "STRUCTURAL", "DROP-DEP-PROVIDER"):
    sub = [r for r in exact_rows if r["cls"] == c]
    print(f"###   {c:14s} exact pins: {len(sub)}")
print()
for r in sorted([x for x in exact_rows if x["cls"] not in ("STRUCTURAL",)], key=lambda r: (r["cls"], r["name"])):
    p = ("<- " + ",".join(r["packs"])) if r["packs"] else ""
    print(f"  {r['cls']:14s} :{r['line']:<5} {r['name']:28s} {r['spec']:22s} {p}")
json.dump(rows, open(sys.argv[3], "w"), indent=None)
