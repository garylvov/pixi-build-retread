#!/usr/bin/env python3
"""Per-ENV per-package VERSION delta between two pixi.lock files.

CHECK 1 of the two-checks doctrine, done properly: b2_attribute.sh measures
whole-file OCCURRENCE counts, which cannot see "same count, different version"
(the mujoco/gym/tensordict trap) and cannot attribute a change to an env.
This walks the `environments:` section of each lock and reports, per env, every
package whose selected version moved.

Usage: env_version_delta.py <baseline.lock> <new.lock> [focus-package ...]
Focus packages are printed FIRST and always, changed or not.
Cross-day index drift makes unrelated rows move; read the rows, do not read a
nonzero row count as a failure.
"""
import re, sys, collections

def parse(path):
    lines = open(path).read().splitlines()
    try:
        pi = next(i for i, l in enumerate(lines) if l.startswith("packages:"))
    except StopIteration:
        pi = len(lines)
    env = collections.defaultdict(dict)
    cur = None
    for l in lines[:pi]:
        m = re.match(r'^  ([A-Za-z0-9_.\-]+):\s*$', l)
        if m:
            cur = m.group(1); continue
        m = re.match(r'^      - (conda|pypi): (\S+)', l)
        if not (m and cur): continue
        fn = m.group(2).rsplit("/", 1)[-1]
        mm = re.match(r'^([A-Za-z0-9_.\+\-]+?)-([0-9][^-]*)-', fn)
        if not mm:
            mm = re.match(r'^([A-Za-z0-9_.\+\-]+?)-([0-9][^-]*?)\.(tar\.gz|zip)$', fn)
        if not mm: continue
        env[cur][mm.group(1).lower().replace("_", "-")] = mm.group(2)
    for junk in ("depends", "constrains", "purls", "requires_dist", "run_exports",
                 "variants", "virtual-packages", "track-features", "build-packages",
                 "host-packages", "packages"):
        env.pop(junk, None)
    return env

def main():
    b, n = parse(sys.argv[1]), parse(sys.argv[2])
    focus = [f.lower().replace("_", "-") for f in sys.argv[3:]]
    print("### baseline=%s\n### new     =%s" % (sys.argv[1], sys.argv[2]))
    print("### envs: baseline=%d new=%d" % (len(b), len(n)))
    if focus:
        print("\n### FOCUS PACKAGES (printed whether or not they moved)")
        print("%-24s %-16s %-16s %s" % ("env", "baseline", "new", "verdict"))
        for e in sorted(set(b) | set(n)):
            for f in focus:
                vb, vn = b.get(e, {}).get(f), n.get(e, {}).get(f)
                if vb is None and vn is None: continue
                v = ("SAME" if vb == vn else
                     "GONE -> SOLE-PROVIDER" if vn is None else
                     "APPEARED" if vb is None else "VERSION MOVED")
                print("%-24s %-16s %-16s %s  [%s]" % (e, vb or "-", vn or "-", v, f))
    print("\n### FULL PER-ENV VERSION DRIFT")
    total = 0
    for e in sorted(set(b) | set(n)):
        eb, en = b.get(e, {}), n.get(e, {})
        rows = []
        for p in sorted(set(eb) | set(en)):
            if eb.get(p) != en.get(p):
                rows.append((p, eb.get(p, "-"), en.get(p, "-")))
        total += len(rows)
        print("  %-24s moved=%d  (baseline pkgs=%d new pkgs=%d)" % (e, len(rows), len(eb), len(en)))
        for p, x, y in rows[:40]:
            print("      %-38s %-18s -> %s" % (p, x, y))
        if len(rows) > 40:
            print("      ... %d more" % (len(rows) - 40))
    print("\n### total moved rows across all envs: %d" % total)

if __name__ == "__main__":
    main()
