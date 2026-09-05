#!/usr/bin/env python3
"""p6af-2g: the SHARDED half of the frozen channel mirror.

WHY THIS EXISTS.  p6af built a loopback mirror of the CLASSIC `repodata.json`
documents and proved byte-identity on a 48-package probe.  p6af-2a took it to
the canonical 27-environment workspace and `env_version_delta` moved 16 rows;
p6af-2e then ran the same-window control -- one network lock and one mirror
lock in ONE job, the mirror frozen from that job's OWN shard cache -- and the
16 rows did not move, which KILLED the shard-cache-staleness explanation.  What
survived is protocol shape: pixi speaks the SHARDED protocol and the mirror
answers with the CLASSIC document, so pixi is served a universe whose
`run_exports` exist only for the names the freeze happened to hold.

THE PROTOCOL, READ OUT OF THE SOURCE, NOT REMEMBERED
(`rattler_repodata_gateway-0.25.5`, `rattler_conda_types-0.42.2` -- the crates
in this box's cargo registry; pixi 0.73.0's own pin is not knowable here and
this file does not pretend otherwise):

  * `gateway::sharded_subdir::tokio::REPODATA_SHARDS_FILENAME` =
    `"repodata_shards.msgpack.zst"`, fetched from `<channel>/<subdir>/` joined
    with that name (`index::fetch_index`).
  * The body is ZSTD over msgpack of
    `rattler_conda_types::repo_data::sharded::ShardedRepodata`:
    `{info: {subdir, base_url, shards_base_url, created_at}, shards: {name -> sha256}}`.
  * `ShardedSubdir::new` resolves BOTH `info.shards_base_url` and
    `info.base_url` RELATIVE TO the index base url and appends a trailing slash
    (`add_trailing_slash`).
  * `fetch_package_records` builds the shard url as
    `shards_base_url.join("{shard:x}.msgpack.zst")`, and the sha in the index is
    the sha256 of the COMPRESSED shard (measured on every live pair, probe
    5870870: `SHA-IS-OF-COMPRESSED=True`).
  * `parse_records` sets each record's `url = base_url.join(file_name)`, so
    `info.base_url` is WHAT THE LOCK RECORDS.
  * On disk pixi keeps the index DECOMPRESSED inside `<8hex>.shards-cache-v1`
    and each shard DECOMPRESSED at `shards-v1/<sha of the COMPRESSED shard>.msgpack`
    (`SHARDS_CACHE_SUFFIX`, `write_cache`).  This file never reads that cache:
    the mirror is frozen from the CHANNEL, so its provenance is the channel.

AND THE ONE FACT THAT DECIDES THE WHOLE DESIGN, measured, not assumed
(probe 5870870):

    prefix.dev publishes  shards_base_url = "https://shards.prefix.dev/<channel>/"

-- an ABSOLUTE url on a DIFFERENT HOST.  `rattler_networking::mirror_middleware`
maps requests by STRING PREFIX against the configured channel key, so a shard
url that no longer begins with `https://prefix.dev/<channel>` never reaches the
loopback mirror at all: pixi would fetch shards straight from
`shards.prefix.dev`, and `shards.prefix.dev` is NOT in the deny list p6af-2a/2e
armed.  A shard mirror that did nothing about this would be silently, and
invisibly, not a mirror.

TWO WAYS TO FIX IT AND WHY THIS ONE.  The index could be REWRITTEN (relative
`shards_base_url`) and re-encoded -- but then the served index is no longer the
document the channel published, its sha256 is ours rather than upstream's, and
a msgpack ENCODER has to be written and trusted for `channel_relations`, for
the two `/pytorch` pairs that store their shas as msgpack ARRAYS, and for the
`packages.whl` key rattler 0.25.5 does not even know about.  Instead the index
is served VERBATIM and `https://shards.prefix.dev/<channel>` is added to pixi's
`[mirrors]` map as a key of its own.  The frozen index is then byte-identical
to the channel's, its sha256 is the channel's sha256, and there is no encoder.

WHAT THIS BUYS, both halves measured before this file was written:
  classic documents, all 21 pairs   1 229 375 784 B   (p6af, 1.15 GiB)
  shard indexes,    all 21 pairs        ~2.1 MiB      (probe 5870870, per-pair
                                                       bytes in INDEX-STATE.json)
and the shards themselves are passed through LAZILY, each one verified against
the sha256 the frozen index names -- so coverage is complete BY CONSTRUCTION
rather than by whatever a warm cache happened to hold, which is precisely the
defect p6af-1 named and p6af-2e failed to cure by warming the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from urllib.parse import urljoin

SCHEMA = "retread-shard-mirror-v1"
INDEX_FILENAME = "repodata_shards.msgpack.zst"
STATE_FILENAME = "INDEX-STATE.json"
UA = {"User-Agent": "retread-channel-mirror/1.0 (+p6af-2g)", "Accept": "*/*"}


# ---------------------------------------------------------------- codecs ----
def zstd_decompress(blob: bytes) -> bytes:
    """`zstandard` when it imports, `/usr/bin/zstd` otherwise.

    Same reason `pixi_shard_cache.py` carries `msgpack_min`: python derives the
    user site from $HOME and every relock harness here runs under a JOB-LOCAL
    HOME, so a module that lives in the user site is simply absent inside a job.
    `zstd` is a system binary and cannot go missing that way.
    """
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(blob, max_output_size=1 << 31)
    except ImportError:
        pass
    except Exception:
        # A framed stream zstandard refuses without a size hint still decodes
        # through the CLI; fall through rather than refuse.
        pass
    proc = subprocess.run(["zstd", "-dc"], input=blob, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("zstd -dc rc=%d: %s" % (proc.returncode,
                                                   proc.stderr[:200].decode("utf-8", "replace")))
    return proc.stdout


def _msgpack():
    try:
        import msgpack
        return msgpack
    except ModuleNotFoundError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import msgpack_min
        return msgpack_min


def sha_hex(value) -> str:
    """The index's shard value -> a 64-char hex string.

    MEASURED (probe 5870870, and `pixi_shard_cache.shard_hex` before it): most
    channels store the sha as msgpack BIN, but BOTH `prefix.dev/pytorch` pairs
    store it as a msgpack ARRAY of ints.  A rule that handled only `bytes` would
    work on conda-forge and break on the channel nobody probes.
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return bytes(bytearray(value)).hex()
    return str(value)


def decode_index(raw: bytes):
    """The published `repodata_shards.msgpack.zst` bytes -> its parsed object."""
    return _msgpack().unpackb(zstd_decompress(raw), raw=False, strict_map_key=False)


def decode_shard(raw: bytes):
    """A published `<sha>.msgpack.zst` shard's bytes -> its parsed object."""
    return _msgpack().unpackb(zstd_decompress(raw), raw=False, strict_map_key=False)


# ------------------------------------------------------------- key rules ----
def slug_of(url: str) -> str:
    """`https://prefix.dev/conda-forge` -> `prefix.dev__conda-forge`.

    The SAME rule `retread_freeze_channel_mirror` and `retread_pixi_mirror_config`
    already use: scheme stripped, host AND path kept, `/` -> `__`.  Host and
    path both, never the last segment: this workspace declares BOTH
    `prefix.dev/pytorch` and `conda.anaconda.org/pytorch`, and a last-segment
    key silently merges two different channels into one directory.
    """
    u = url.split("://", 1)[-1].rstrip("/")
    return u.replace("/", "__")


def fetch(url: str, timeout: int = 300):
    """GET, returning (status, bytes).  404 is a RESULT here, not an error."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


# ----------------------------------------------------------- the freeze -----
def freeze(root: str, pairs, log=sys.stderr) -> dict:
    """Fetch and store one frozen shard index per (channel, subdir).

    `pairs` is an iterable of (channel url without trailing slash, subdir).
    Returns the state dict and writes it to `<root>/INDEX-STATE.json`.

    A pair whose index is a 404, or whose index carries ZERO shards (measured:
    `prefix.dev/robostack-humble/noarch` publishes an index with an empty
    `shards` map and a 174-byte classic document), is recorded as `classic` and
    its plain `repodata.json` is fetched instead.  Both modes are named in the
    state file, so a pair that quietly lost its sharded protocol can never be
    silent -- it is a row, and the server refuses to serve a classic document
    for any pair the state calls `sharded`.
    """
    os.makedirs(root, exist_ok=True)
    state = {"schema": SCHEMA, "root": os.path.abspath(root), "pairs": [],
             "mirror_keys": [], "totals": {}}
    channels = []
    idx_bytes = cls_bytes = 0
    n_sharded = n_classic = 0
    for chan, subdir in pairs:
        chan = chan.rstrip("/")
        if chan not in channels:
            channels.append(chan)
        slug = slug_of(chan)
        pair_dir = os.path.join(root, slug, subdir)
        os.makedirs(pair_dir, exist_ok=True)
        index_base = "%s/%s/" % (chan, subdir)
        index_url = index_base + INDEX_FILENAME
        status, raw = fetch(index_url)
        row = {"channel": chan, "subdir": subdir, "slug": slug,
               "index_url": index_url, "index_status": status}
        if status == 200:
            info = decode_index(raw).get("info") or {}
            shards = decode_index(raw).get("shards") or {}
        else:
            info, shards = {}, {}
        if status == 200 and shards:
            sbase = urljoin(index_base, info.get("shards_base_url") or "")
            if not sbase.endswith("/"):
                sbase += "/"
            pbase = urljoin(index_base, info.get("base_url") or "")
            if not pbase.endswith("/"):
                pbase += "/"
            with open(os.path.join(pair_dir, INDEX_FILENAME), "wb") as fh:
                fh.write(raw)
            row.update(mode="sharded",
                       index_sha256=hashlib.sha256(raw).hexdigest(),
                       index_bytes=len(raw), n_shards=len(shards),
                       shards_base_url=sbase, base_url=pbase)
            idx_bytes += len(raw)
            n_sharded += 1
        else:
            # No sharded protocol for this pair.  Serve what the channel does
            # publish, and SAY which pair lost the protocol.
            cstatus, cblob = fetch(index_base + "repodata.json")
            if cstatus != 200:
                raise RuntimeError(
                    "shard_mirror.freeze: %s/%s has neither a shard index (%s) "
                    "nor a repodata.json (%s)" % (chan, subdir, status, cstatus))
            with open(os.path.join(pair_dir, "repodata.json"), "wb") as fh:
                fh.write(cblob)
            row.update(mode="classic", classic_bytes=len(cblob),
                       classic_sha256=hashlib.sha256(cblob).hexdigest(),
                       n_shards=len(shards))
            cls_bytes += len(cblob)
            n_classic += 1
        state["pairs"].append(row)
        log.write("### shard_mirror %-46s %s\n" % (slug + "/" + subdir,
                  " ".join("%s=%s" % (k, row[k]) for k in
                           ("mode", "index_bytes", "n_shards", "classic_bytes",
                            "index_sha256") if k in row)))
        log.flush()

    # THE MIRROR KEY SET.  Every channel the lock declares, plus every distinct
    # shards_base_url that does not already sit under one of them -- that second
    # group is the whole point of this file, and it is derived from the frozen
    # indexes rather than configured, so a channel that moves its shards to a
    # new host is picked up by the freeze and not by a human remembering to.
    keys = [{"url": c, "slug": slug_of(c), "kind": "channel"} for c in channels]
    for row in state["pairs"]:
        if row["mode"] != "sharded":
            continue
        sbase = row["shards_base_url"]
        if any(sbase.startswith(k["url"].rstrip("/") + "/") for k in keys):
            row["shards_local_dir"] = _local_dir_for(sbase, keys)
            continue
        k = sbase.rstrip("/")
        if not any(x["url"] == k for x in keys):
            keys.append({"url": k, "slug": slug_of(k), "kind": "shards"})
        row["shards_local_dir"] = _local_dir_for(sbase, keys)
    state["mirror_keys"] = keys
    # Every directory a shard can be served out of must EXIST before the server
    # starts, empty or not: `retread_pixi_mirror_config`'s reader curls each
    # mirror base and refuses a base the freeze never wrote, and a lazily-filled
    # directory that only appears after the first shard would fail that check
    # before a single shard had been asked for.
    for row in state["pairs"]:
        if row.get("shards_local_dir"):
            os.makedirs(os.path.join(root, row["shards_local_dir"]), exist_ok=True)
    state["totals"] = {"pairs": len(state["pairs"]), "sharded": n_sharded,
                       "classic": n_classic, "index_bytes": idx_bytes,
                       "classic_bytes": cls_bytes}
    with open(os.path.join(root, STATE_FILENAME), "w") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)
    return state


def _local_dir_for(url: str, keys) -> str:
    """The mirror-local directory a url is served out of, under the longest key.

    This is the inverse of what `MirrorMiddleware::handle` does: it strips the
    matching key from the front of the url and joins the remainder onto the
    mirror base, so the mirror must hold exactly that remainder under the key's
    slug.  Longest key first, for the same reason the middleware sorts its keys
    by path length.
    """
    best = None
    for k in sorted(keys, key=lambda k: -len(k["url"])):
        pfx = k["url"].rstrip("/") + "/"
        if url.startswith(pfx):
            best = (k["slug"], url[len(pfx):])
            break
    if best is None:
        raise RuntimeError("shard_mirror: no mirror key covers %s" % url)
    return os.path.join(best[0], best[1].strip("/")) if best[1].strip("/") else best[0]


def load_state(root: str) -> dict:
    with open(os.path.join(root, STATE_FILENAME)) as fh:
        state = json.load(fh)
    if state.get("schema") != SCHEMA:
        raise RuntimeError("shard_mirror: %s is schema %r, want %r"
                           % (STATE_FILENAME, state.get("schema"), SCHEMA))
    return state


def load_frozen_shards(root: str, state: dict):
    """Rebuild the allow-map from the FROZEN INDEX FILES, not from a side table.

    The index is the authority on which shards exist; a second copy of that
    truth in the state file would be a second thing to keep in step.  Every
    index is re-verified against the sha256 the freeze recorded before it is
    trusted, so an index that moved on disk between the freeze and the serve is
    a refusal and not a quietly different universe.

    Returns (shard_upstream, shard_local, name_shard):
      shard_upstream[sha]        -> the real url the shard is fetched from
      shard_local[sha]           -> the mirror-relative directory it is served from
      name_shard[(slug,subdir)]  -> {package name: sha}
    """
    shard_upstream, shard_local, name_shard = {}, {}, {}
    for row in state["pairs"]:
        if row["mode"] != "sharded":
            continue
        path = os.path.join(root, row["slug"], row["subdir"], INDEX_FILENAME)
        with open(path, "rb") as fh:
            raw = fh.read()
        got = hashlib.sha256(raw).hexdigest()
        if got != row["index_sha256"]:
            raise RuntimeError("shard_mirror: frozen index %s moved: sha %s != %s"
                               % (path, got, row["index_sha256"]))
        shards = decode_index(raw).get("shards") or {}
        table = {}
        for name, value in shards.items():
            hexsha = sha_hex(value)
            table[name] = hexsha
            shard_upstream[hexsha] = row["shards_base_url"] + hexsha + ".msgpack.zst"
            # A SET, not a string: the same sha can legitimately be reachable
            # under two pairs of one channel (both subdirs share
            # `shards.prefix.dev/<channel>/`), and collapsing that to the last
            # writer would make a correct request look misplaced.
            shard_local.setdefault(hexsha, set()).add(row["shards_local_dir"])
        name_shard[(row["slug"], row["subdir"])] = table
    return shard_upstream, shard_local, name_shard


# ------------------------------------------------------------------ cli -----
def _pairs_from_lock(lock: str):
    """The channels and subdirs of a pixi lock, by the rules already in force.

    Channels come from the `- url: https://…` lines; subdirs come from the
    RECORD urls plus noarch, never from the `platforms:` block -- a lock whose
    platform name equals its subdir omits the `subdir:` key entirely, and a
    `subdir:`-only rule once built a mirror with noarch and nothing else.
    """
    chans, subdirs = [], set(["noarch"])
    with open(lock) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("- url: https://"):
                u = s[len("- url: "):].strip().rstrip("/")
                if u not in chans:
                    chans.append(u)
            elif s.startswith("- conda: http"):
                u = s[len("- conda: "):].strip()
                parts = u.split("://", 1)[-1].split("/")
                if len(parts) >= 2:
                    subdirs.add(parts[-2])
    if not chans:
        raise RuntimeError("shard_mirror: no channel urls in %s" % lock)
    return [(c, s) for c in chans for s in sorted(subdirs)]


def main(argv) -> int:
    if len(argv) < 3:
        sys.stderr.write("usage: shard_mirror.py freeze <lock> <mirror root>\n"
                         "       shard_mirror.py keys <mirror root>\n")
        return 2
    verb = argv[1]
    if verb == "freeze":
        lock, root = argv[2], argv[3]
        state = freeze(root, _pairs_from_lock(lock))
        t = state["totals"]
        print("### shard_mirror frozen root=%s pairs=%d sharded=%d classic=%d "
              "index_bytes=%d (%.2f MiB) classic_bytes=%d keys=%d"
              % (root, t["pairs"], t["sharded"], t["classic"], t["index_bytes"],
                 t["index_bytes"] / 1048576.0, t["classic_bytes"],
                 len(state["mirror_keys"])))
        for k in state["mirror_keys"]:
            print("###   mirror key %-8s %-52s -> %s" % (k["kind"], k["url"], k["slug"]))
        return 0
    if verb == "keys":
        state = load_state(argv[2])
        for k in state["mirror_keys"]:
            print("%s\t%s\t%s" % (k["kind"], k["url"], k["slug"]))
        return 0
    if verb == "probe":
        # What `p6af2g_shard_mirror_guard.sh` needs in order to replay pixi's
        # OWN request pattern against the mirror: the pair's frozen index sha,
        # one real shard of it, where that shard is served from, and one real
        # package file name inside that shard.  A guard that made these up
        # would be testing its own arithmetic.
        root, slug, subdir = argv[2], argv[3], argv[4]
        state = load_state(root)
        row = next(r for r in state["pairs"]
                   if r["slug"] == slug and r["subdir"] == subdir)
        print("mode\t%s" % row["mode"])
        if row["mode"] != "sharded":
            return 0
        path = os.path.join(root, slug, subdir, INDEX_FILENAME)
        with open(path, "rb") as fh:
            raw = fh.read()
        idx = decode_index(raw)
        print("index_path\t%s" % os.path.join(slug, subdir, INDEX_FILENAME))
        print("index_sha256\t%s" % row["index_sha256"])
        print("shards_local_dir\t%s" % row["shards_local_dir"])
        print("shards_base_url\t%s" % row["shards_base_url"])
        shards = idx["shards"]
        # A SMALL shard, so the guard's package arm downloads a small archive.
        name = sorted(shards)[0]
        hexsha = sha_hex(shards[name])
        print("shard_name\t%s" % name)
        print("shard_sha\t%s" % hexsha)
        print("shard_path\t%s" % os.path.join(row["shards_local_dir"],
                                              hexsha + ".msgpack.zst"))
        return 0
    if verb == "shard-files":
        # The package file names inside one already-served shard, smallest
        # first, so the guard's package arm can name a real archive.
        root, local = argv[2], argv[3]
        with open(os.path.join(root, local), "rb") as fh:
            shard = decode_shard(fh.read())
        rows = []
        for key in ("packages.conda", "packages"):
            for fname, rec in (shard.get(key) or {}).items():
                rows.append((rec.get("size") or 0, fname))
        for size, fname in sorted(rows)[:5]:
            print("%s\t%s" % (size, fname))
        return 0
    if verb == "set-shards-base":
        # MUTATION SUPPORT FOR THE GUARD, and nothing else calls it.  Repointing
        # a pair's shards_base_url at a server that answers with the WRONG BYTES
        # is the only way to exercise the sha256 refusal on the shard path with
        # no network and no forged msgpack.
        root, slug, subdir, newbase = argv[2], argv[3], argv[4], argv[5]
        state = load_state(root)
        for row in state["pairs"]:
            if row["slug"] == slug and row["subdir"] == subdir:
                row["shards_base_url"] = newbase if newbase.endswith("/") else newbase + "/"
        with open(os.path.join(root, STATE_FILENAME), "w") as fh:
            json.dump(state, fh, indent=1, sort_keys=True)
        print("set-shards-base %s/%s -> %s" % (slug, subdir, newbase))
        return 0
    sys.stderr.write("shard_mirror: unknown verb %r\n" % verb)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
