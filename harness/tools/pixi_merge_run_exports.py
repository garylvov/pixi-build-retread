"""Merge `run_exports` from pixi's sharded cache into a classic repodata.json.

WHY THIS EXISTS (measured, p6af): prefix.dev publishes `run_exports` ONLY in the
sharded protocol -- `<subdir>/run_exports.json` is a 404 and retread's own
`retread-repodata/*.json` snapshots carry zero `run_exports` keys -- while the
certified lock carries 2432 of them. A frozen file mirror built from the classic
document alone therefore produces a lock that differs from a network lock in
exactly the run_exports blocks. pixi DOES honour a `run_exports` key found in a
classic repodata.json record (measured with a marker value), so the fix is to
carry the field across at freeze time.

    merge_run_exports.py <src repodata.json> <dst repodata.json> \
                         <pixi repodata cache dir> <base_url prefix>
                         [--allow-missing-index]

`--allow-missing-index` (p6af-2, 2026-09-04). The shard cache only holds an
index for a (channel, subdir) pair some solve on this machine actually FETCHED.
The canonical workspace declares 7 channels x 3 subdirs = 21 pairs and the
shared cache carried 19 indices, so a strict run aborts the whole freeze on the
two pairs no solve ever touched -- pairs that, by construction, contribute no
records to the lock either. With the flag such a pair is copied through with NO
run_exports and says so on its census line (`index=absent`), which is the reader
for it: a pair that silently lost its run_exports would otherwise be invisible.
Strict remains the DEFAULT so the existing guard's behaviour is unchanged.
"""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pixi_shard_cache import load_index, load_shard


def run_exports_map(cache_dir, base_url, allow_missing=False):
    idx_path = None
    for p in glob.glob(os.path.join(cache_dir, "*.shards-cache-v1")):
        try:
            _, idx = load_index(p)
        except Exception:
            continue
        if idx.get("info", {}).get("base_url", "").rstrip("/") == base_url.rstrip("/"):
            idx_path = p
            break
    if idx_path is None:
        if allow_missing:
            sys.stderr.write("run_exports_map base_url=%s index=absent shards_present=0 "
                             "shards_absent=0 records=0\n" % base_url)
            return None
        raise SystemExit("no shard index for base_url %s under %s" % (base_url, cache_dir))
    _, idx = load_index(idx_path)
    shard_dir = os.path.join(cache_dir, "shards-v1")
    out, have, miss = {}, 0, 0
    for name, sha in idx["shards"].items():
        h = sha.hex() if isinstance(sha, bytes) else sha
        shard = load_shard(shard_dir, h)
        if shard is None:
            miss += 1
            continue
        have += 1
        for group in ("packages", "packages.conda"):
            for fn, rec in shard.get(group, {}).items():
                if "run_exports" in rec:
                    out[fn] = rec["run_exports"]
    sys.stderr.write("run_exports_map base_url=%s names=%d shards_present=%d shards_absent=%d records=%d\n"
                     % (base_url, len(idx["shards"]), have, miss, len(out)))
    return out


def main():
    args = [a for a in sys.argv[1:] if a != "--allow-missing-index"]
    allow_missing = "--allow-missing-index" in sys.argv[1:]
    src, dst, cache_dir, base_url = args[:4]
    rmap = run_exports_map(cache_dir, base_url, allow_missing)
    if rmap is None:
        rmap = {}
    n = 0
    with open(src) as f, open(dst, "w") as o:
        pending = None
        for line in f:
            o.write(line)
            s = line.strip()
            if pending is not None:
                o.write('      "run_exports": %s,\n' % json.dumps(pending))
                pending = None
                n += 1
                continue
            if s.endswith(": {") and s.startswith('"'):
                fn = s[1:s.index('":')]
                if fn in rmap:
                    pending = rmap[fn]
    print("injected", n, "of", len(rmap))


if __name__ == "__main__":
    main()
