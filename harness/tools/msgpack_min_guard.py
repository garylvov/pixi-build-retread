#!/usr/bin/env python3
"""Guard for `msgpack_min.py`: decode LIVE pixi cache files both ways and diff.

    msgpack_min_guard.py <PIXI_CACHE_DIR>/repodata [max shards per index]

It is not a fixture test. It reads the shard-cache indexes this campaign's
freeze actually reads, decodes each one with the real `msgpack` and with
`msgpack_min`, and refuses on ANY difference in the decoded objects -- then does
the same for a sample of the `shards-v1/*.msgpack` bodies, which are where the
`run_exports` records the freeze carries across actually live.

It REFUSES (exit 2) if the real `msgpack` is not importable, because then it is
comparing the fallback with itself and would pass no matter what.

MUTATION-CHECKED 2026-09-04: with the `0xC4` (bin8) branch of `msgpack_min`
made to return `None`, this exits 1 naming the first index whose `shards` map
differs; with the negative-fixint branch off by one, likewise.
"""
import sys, os, glob, struct

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import msgpack as real
except ModuleNotFoundError:
    sys.stderr.write("GUARD REFUSE: the real `msgpack` is not importable here, so this "
                     "guard would compare msgpack_min with itself. Run it where msgpack "
                     "is on the path (e.g. with the real HOME).\n")
    raise SystemExit(2)

import msgpack_min as mini
from pixi_shard_cache import shard_hex

MAGIC = b"SHARD-CACHE-V1"


def split_index(path):
    b = open(path, "rb").read()
    assert b[:len(MAGIC)] == MAGIC, path
    hlen = struct.unpack("<I", b[len(MAGIC):len(MAGIC) + 4])[0]
    off = len(MAGIC) + 4
    return b[off:off + hlen], b[off + hlen:]


def main():
    root = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    idxs = sorted(glob.glob(os.path.join(root, "*.shards-cache-v1")))
    if not idxs:
        sys.stderr.write("GUARD REFUSE: no *.shards-cache-v1 under %s -- nothing to compare\n" % root)
        raise SystemExit(2)
    shard_dir = os.path.join(root, "shards-v1")
    bad = 0
    n_idx = n_shard = 0
    for p in idxs:
        head, body = split_index(p)
        for label, blob in (("meta", head), ("index", body)):
            a = real.unpackb(blob, raw=False, strict_map_key=False)
            b = mini.unpackb(blob, raw=False, strict_map_key=False)
            if a != b:
                print("GUARD FAIL %s %s: msgpack and msgpack_min disagree" % (os.path.basename(p), label))
                bad += 1
        n_idx += 1
        idx = real.unpackb(body, raw=False, strict_map_key=False)
        for i, (_name, sha) in enumerate(sorted(idx.get("shards", {}).items())):
            if i >= cap:
                break
            h = shard_hex(sha)
            sp = os.path.join(shard_dir, h + ".msgpack")
            if not os.path.exists(sp):
                continue
            blob = open(sp, "rb").read()
            if real.unpackb(blob, raw=False, strict_map_key=False) != \
               mini.unpackb(blob, raw=False, strict_map_key=False):
                print("GUARD FAIL shard %s: msgpack and msgpack_min disagree" % h)
                bad += 1
            n_shard += 1
    print("msgpack_min guard: indexes=%d shards=%d mismatches=%d" % (n_idx, n_shard, bad))
    if bad:
        raise SystemExit(1)
    print("GUARD PASS")


if __name__ == "__main__":
    main()
