"""Decode pixi's <PIXI_CACHE_DIR>/repodata/ sharded-index cache.

MEASURED layout of `<8hex>.shards-cache-v1`:
    b"SHARD-CACHE-V1" + u32le(header_len) + msgpack(cached HTTP meta)
    + msgpack(the DECOMPRESSED repodata_shards index)
The body is NOT the wire bytes: pixi stores the index decompressed, and
`shards-v1/<sha256>.msgpack` likewise holds the DECOMPRESSED shard while its
name is the sha256 of the ZSTD-COMPRESSED shard as published.
"""
import sys, struct, glob, os, io

# The real `msgpack` when it is importable, otherwise the dependency-free
# decoder beside this file. `msgpack` lives in the USER site, which python
# derives from $HOME, and every relock harness here runs under a JOB-LOCAL HOME
# -- so inside a job the import fails and the whole channel-mirror freeze aborts
# on its first channel (measured, job 5849657). `msgpack_min_guard.py` checks the
# two against each other on live cache files.
try:
    import msgpack
except ModuleNotFoundError:  # pragma: no cover -- exercised by every batch job
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import msgpack_min as msgpack

MAGIC = b"SHARD-CACHE-V1"

def load_index(path):
    b = open(path, "rb").read()
    assert b[:len(MAGIC)] == MAGIC, path
    hlen = struct.unpack("<I", b[len(MAGIC):len(MAGIC)+4])[0]
    off = len(MAGIC) + 4
    meta = msgpack.unpackb(b[off:off+hlen], raw=False, strict_map_key=False)
    idx = msgpack.unpackb(b[off+hlen:], raw=False, strict_map_key=False)
    return meta, idx

def shard_hex(sha):
    """The `shards` map's value -> the `shards-v1/<name>.msgpack` stem.

    MEASURED 2026-09-04 on the live cache: most indexes store the sha256 as
    msgpack BIN (python `bytes`), but `/pytorch/linux-64` and `/pytorch/noarch`
    store it as a msgpack ARRAY of ints (python `list`). The old inline
    `sha.hex() if isinstance(sha, bytes) else sha` handed that list straight to
    `os.path.join`, which raises `TypeError: can only concatenate list (not
    "str") to list` -- p6af never saw it because its probe manifest used
    conda-forge alone.
    """
    if isinstance(sha, (bytes, bytearray)):
        return bytes(sha).hex()
    if isinstance(sha, (list, tuple)):
        return bytes(bytearray(sha)).hex()
    return str(sha)


def load_shard(shard_dir, sha_hex):
    p = os.path.join(shard_dir, sha_hex + ".msgpack")
    if not os.path.exists(p):
        return None
    return msgpack.unpackb(open(p, "rb").read(), raw=False, strict_map_key=False)

if __name__ == "__main__":
    d = sys.argv[1]
    for p in sorted(glob.glob(d + "/*.shards-cache-v1")):
        try:
            meta, idx = load_index(p)
        except Exception as e:
            print(os.path.basename(p), "ERR", e); continue
        sh = idx.get("shards", {})
        print("%-28s file=%-8d shards=%-6d info=%s"
              % (os.path.basename(p), os.path.getsize(p), len(sh), idx.get("info")))
