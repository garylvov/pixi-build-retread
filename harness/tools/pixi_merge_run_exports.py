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
                         [<channel host>] [--allow-missing-index]

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
from pixi_shard_cache import load_index, load_shard, shard_hex


def _host(url):
    u = url.split("://", 1)[-1]
    return u.split("/", 1)[0]


def run_exports_map(cache_dir, base_url, channel_host="", allow_missing=False):
    """The shard index for one (channel, subdir), or None when there is none.

    THE HOST CHECK IS LOAD-BEARING, measured 2026-09-04. An index's `base_url`
    is a PATH -- `/pytorch/linux-64` -- and this workspace declares TWO channels
    with that path, `https://prefix.dev/pytorch` and
    `https://conda.anaconda.org/pytorch`. Matching on the path alone hands
    prefix.dev's run_exports to anaconda.org's mirror document, which is the same
    silent wrong-channel read `retread_freeze_channel_mirror` already refuses in
    its DIRECTORY naming. An index that publishes shards names them in
    `shards_base_url` (`https://shards.prefix.dev/<channel>`), so the rule is:
    the shards host must be the channel's host or `shards.<channel host>`.
    anaconda.org publishes no shards at all, so its pairs correctly come back as
    index=absent and are mirrored with no run_exports -- which is exactly what
    the network would have given them.
    """
    idx_path = None
    for p in glob.glob(os.path.join(cache_dir, "*.shards-cache-v1")):
        try:
            _, idx = load_index(p)
        except Exception:
            continue
        info = idx.get("info", {})
        if info.get("base_url", "").rstrip("/") != base_url.rstrip("/"):
            continue
        if channel_host:
            sh = _host(info.get("shards_base_url", "") or "")
            if sh and sh != channel_host and sh != "shards." + channel_host:
                continue
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
        h = shard_hex(sha)
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
    channel_host = args[4] if len(args) > 4 else ""
    rmap = run_exports_map(cache_dir, base_url, channel_host, allow_missing)
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
