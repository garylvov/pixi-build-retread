"""A dependency-free msgpack DECODER, enough for pixi's shard cache.

WHY THIS EXISTS, measured 2026-09-04 on job 5849657. `pixi_shard_cache.py`
imported the third-party `msgpack`, which is installed under
`/users/glvov/.local/lib/python3.X/site-packages` -- i.e. it resolves through
the USER SITE, which python derives from `$HOME`. Every relock harness on this
campaign sets a JOB-LOCAL `HOME` (that is the same mechanism p6af's
`[mirrors]` config depends on), so inside a job the import fails:

    File ".../pixi_shard_cache.py", line 11, in <module>
        import msgpack
    ModuleNotFoundError: No module named 'msgpack'

and `retread_freeze_channel_mirror` aborted on the first channel. Pinning a
PYTHONPATH at the real home would make the harness depend on the very variable
the harness deliberately moves; decoding msgpack here removes the dependency
instead. Encoding is NOT implemented -- nothing in this campaign writes msgpack.

THE READER: `pixi_shard_cache.py` prefers the real `msgpack` when it is
importable and falls back to this, and `msgpack_min_guard.py` decodes the SAME
live shard index and the SAME shards with both implementations and refuses on
any difference. So the fallback is checked against the reference rather than
trusted, and a divergence is a red guard, not a wrong lock three hours later.
"""
import struct

_U8 = struct.Struct(">B")
_U16 = struct.Struct(">H")
_U32 = struct.Struct(">I")
_U64 = struct.Struct(">Q")
_I8 = struct.Struct(">b")
_I16 = struct.Struct(">h")
_I32 = struct.Struct(">i")
_I64 = struct.Struct(">q")
_F32 = struct.Struct(">f")
_F64 = struct.Struct(">d")


class _Reader(object):
    __slots__ = ("b", "i")

    def __init__(self, b):
        self.b = b
        self.i = 0

    def take(self, n):
        i = self.i
        j = i + n
        if j > len(self.b):
            raise ValueError("msgpack: truncated (want %d at %d of %d)" % (n, i, len(self.b)))
        self.i = j
        return self.b[i:j]

    def u(self, s):
        return s.unpack(self.take(s.size))[0]

    def value(self):
        c = self.u(_U8)
        if c <= 0x7F:                      # positive fixint
            return c
        if c >= 0xE0:                      # negative fixint
            return c - 0x100
        if 0x80 <= c <= 0x8F:              # fixmap
            return self.map(c & 0x0F)
        if 0x90 <= c <= 0x9F:              # fixarray
            return self.array(c & 0x0F)
        if 0xA0 <= c <= 0xBF:              # fixstr
            return self.take(c & 0x1F).decode("utf-8")
        if c == 0xC0:
            return None
        if c == 0xC2:
            return False
        if c == 0xC3:
            return True
        if c == 0xC4:
            return self.take(self.u(_U8))
        if c == 0xC5:
            return self.take(self.u(_U16))
        if c == 0xC6:
            return self.take(self.u(_U32))
        if c in (0xC7, 0xC8, 0xC9):        # ext 8/16/32
            n = self.u({0xC7: _U8, 0xC8: _U16, 0xC9: _U32}[c])
            t = self.u(_I8)
            return (t, self.take(n))
        if c == 0xCA:
            return self.u(_F32)
        if c == 0xCB:
            return self.u(_F64)
        if c == 0xCC:
            return self.u(_U8)
        if c == 0xCD:
            return self.u(_U16)
        if c == 0xCE:
            return self.u(_U32)
        if c == 0xCF:
            return self.u(_U64)
        if c == 0xD0:
            return self.u(_I8)
        if c == 0xD1:
            return self.u(_I16)
        if c == 0xD2:
            return self.u(_I32)
        if c == 0xD3:
            return self.u(_I64)
        if 0xD4 <= c <= 0xD8:              # fixext 1,2,4,8,16
            n = 1 << (c - 0xD4)
            t = self.u(_I8)
            return (t, self.take(n))
        if c == 0xD9:
            return self.take(self.u(_U8)).decode("utf-8")
        if c == 0xDA:
            return self.take(self.u(_U16)).decode("utf-8")
        if c == 0xDB:
            return self.take(self.u(_U32)).decode("utf-8")
        if c == 0xDC:
            return self.array(self.u(_U16))
        if c == 0xDD:
            return self.array(self.u(_U32))
        if c == 0xDE:
            return self.map(self.u(_U16))
        if c == 0xDF:
            return self.map(self.u(_U32))
        raise ValueError("msgpack: unknown type byte 0x%02X at %d" % (c, self.i - 1))

    def array(self, n):
        return [self.value() for _ in range(n)]

    def map(self, n):
        out = {}
        for _ in range(n):
            k = self.value()
            if isinstance(k, (bytes, bytearray)):
                k = bytes(k)
            elif isinstance(k, list):
                k = tuple(k)
            out[k] = self.value()
        return out


def unpackb(data, raw=False, strict_map_key=False):
    """Decode ONE msgpack value from `data`. Trailing bytes are an error.

    The `raw` / `strict_map_key` keywords exist only so this is a drop-in for
    the calls `pixi_shard_cache.py` already makes; `raw=True` is refused rather
    than silently ignored, because it would change what every string becomes.
    """
    if raw:
        raise NotImplementedError("msgpack_min.unpackb: raw=True is not implemented")
    # Plain bytes, not a memoryview: `take` feeds `.decode("utf-8")` directly and
    # a memoryview has no `.decode`. The indexes are a couple of MB; the copies
    # slicing makes are not worth a second code path.
    r = _Reader(bytes(data))
    v = r.value()
    if r.i != len(r.b):
        raise ValueError("msgpack: %d trailing bytes" % (len(r.b) - r.i))
    return _plain(v)


def _plain(v):
    if isinstance(v, memoryview):
        return v.tobytes()
    if isinstance(v, dict):
        return dict((_plain(k), _plain(x)) for k, x in v.items())
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], (bytes, bytearray, memoryview)):
        return (v[0], _plain(v[1]))
    return v
