"""Decode pixi's <PIXI_CACHE_DIR>/repodata/ sharded-index cache.

MEASURED layout of `<8hex>.shards-cache-v1`:
    b"SHARD-CACHE-V1" + u32le(header_len) + msgpack(cached HTTP meta)
    + msgpack(the DECOMPRESSED repodata_shards index)
The body is NOT the wire bytes: pixi stores the index decompressed, and
`shards-v1/<sha256>.msgpack` likewise holds the DECOMPRESSED shard while its
name is the sha256 of the ZSTD-COMPRESSED shard as published.
"""
import sys, struct, glob, os, io
import msgpack

MAGIC = b"SHARD-CACHE-V1"

def load_index(path):
    b = open(path, "rb").read()
    assert b[:len(MAGIC)] == MAGIC, path
    hlen = struct.unpack("<I", b[len(MAGIC):len(MAGIC)+4])[0]
    off = len(MAGIC) + 4
    meta = msgpack.unpackb(b[off:off+hlen], raw=False, strict_map_key=False)
    idx = msgpack.unpackb(b[off+hlen:], raw=False, strict_map_key=False)
    return meta, idx

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
