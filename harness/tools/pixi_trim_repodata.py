"""Emit a repodata.json holding ONLY the records a lock references, and say how
big that is.  MEASUREMENT ONLY -- the frozen mirror does NOT ship a trimmed
universe: a trimmed document answers "no such package" to retread's route
probes, which is the p6ac / C22-3 divergence mechanism, not a saving.

    trim_repodata.py <src repodata.json> <dst repodata.json> <lock> <url prefix>

Full json.load, deliberately: a line/brace scanner desynchronises on the first
brace inside a string value (measured on conda-forge linux-64 -- it found 73 of
1265 records before this was rewritten).
"""
import sys, json, time

src, dst, lock, prefix = sys.argv[1:5]
want = set()
for line in open(lock):
    s = line.strip()
    if s.startswith("- conda: " + prefix):
        want.add(s[len("- conda: " + prefix):])

t0 = time.time()
doc = json.load(open(src))
t1 = time.time()
out = {"info": doc.get("info", {}), "repodata_version": doc.get("repodata_version", 1)}
kept = 0
for group in ("packages", "packages.conda"):
    g = {k: v for k, v in doc.get(group, {}).items() if k in want}
    kept += len(g)
    out[group] = g
json.dump(out, open(dst, "w"), indent=2, sort_keys=True)
print("kept %d of %d wanted (source records %d) load=%.1fs total=%.1fs"
      % (kept, len(want),
         len(doc.get("packages", {})) + len(doc.get("packages.conda", {})),
         t1 - t0, time.time() - t0))
