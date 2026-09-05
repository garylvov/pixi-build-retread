#!/usr/bin/env python3
"""p6af-2h STEP 1+2: measure what a canonical 27-env lock reads from PyPI, and
what a verbatim snapshot of those simple-index pages would COST.

Two independent sources, both named in the output so neither can be confused for
the other:

  (A) ON-DISK, the read set.  pixi 0.73.0's embedded uv writes its simple-index
      cache into  <PIXI_CACHE_DIR>/uv-cache/simple-v21/  which p6af2g_arm.sh
      symlinks back to the SHARED persist root, so it survived the cleanup that
      reaped the job roots.  Layout, verified not quoted:
        simple-v21/pypi/<project>.rkyv          -- the DEFAULT index (pypi.org)
        simple-v21/index/<16 hex>/<project>.rkyv -- one bucket per OTHER index
      Each file is rkyv; the rkyv root sits at the TAIL, and the tail carries the
      request URL, the ETag and the request headers.  We recover the URL by
      scanning the tail for an ascii http(s) url, and the etag as the quoted
      token that follows it.  This is a READ of the bytes, not a parse of uv's
      schema -- stated as the limit it is.

  (B) LIVE, the cost.  For every URL recovered in (A) we re-request the page with
      the EXACT Accept and Accept-Encoding uv sent (also recovered from the tail
      of the cache file, printed) and record: status, content-type, the bytes ON
      THE WIRE (compressed, which is what a snapshot stores if it stores the
      served representation) and the bytes decoded, plus etag and cache-control.
      The sum of the wire bytes IS the sizing this lane owes.

  (C) The lock's own pypi artefact URLs, per host, with a HEAD for content-length
      -- so the artefact half is sized too, even though the design passes it
      through rather than freezing it.
"""
import gzip
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter, defaultdict

SIMPLE21 = "/oscar/data/stellex/glvov/agrescap/cache/retread/pixi/uv-cache/simple-v21"
SIMPLE24 = "/oscar/data/stellex/glvov/agrescap/cache/retread/uv/simple-v24"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

URL_RE = re.compile(rb"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")


def tail_url_and_etag(path, nbytes=4096):
    """Recover (url, etag, accept, accept_encoding) from the rkyv tail."""
    sz = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.seek(max(0, sz - nbytes))
        tail = fh.read()
    urls = URL_RE.findall(tail)
    if not urls:
        return None, None, None, None, sz
    # the cache-policy url is the LAST url in the tail
    url = urls[-1].decode("ascii", "replace")
    # The rkyv tail packs the header names straight after the url with no
    # separator ("…/mujoco/accept-encodinggzip, deflate, zstd"), so the regex
    # over-reads. Cut the url at the project segment: every uv index url ends
    # `<base>/<project name>/`, and the project name is the CACHE FILE NAME.
    name = os.path.basename(path)[:-5]
    key = "/" + name + "/"
    i = url.find(key)
    if i >= 0:
        url = url[: i + len(key)]
    else:
        for marker in ('"', "accept", "application/", "gzip"):
            j = url.find(marker)
            if j > 8:
                url = url[:j]
    after = tail[tail.rfind(urls[-1]) + len(urls[-1]):]
    m = re.search(rb'"([^"]{4,80})"', after)
    etag = m.group(1).decode("ascii", "replace") if m else None
    am = re.search(rb"(application/vnd\.pypi\.simple\.v1\+json, application/vnd\.pypi\.simple\.v1\+html;q=0\.2, text/html;q=0\.01)", after)
    accept = am.group(1).decode("ascii", "replace") if am else None
    # uv sends "gzip, deflate, zstd"; this reader cannot decode zstd, so it asks
    # for "gzip, deflate" and the wire bytes it measures are therefore an UPPER
    # BOUND on what uv actually pulled (zstd is smaller than gzip on these
    # documents) -- and they are the right figure for a snapshot that stores the
    # gzip representation. Stated, not hidden.
    aenc = "gzip, deflate"
    return url, etag, accept, aenc, sz


def inventory(root, label):
    rows = []
    if not os.path.isdir(root):
        return rows
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".rkyv"):
                continue
            p = os.path.join(dirpath, fn)
            url, etag, accept, aenc, sz = tail_url_and_etag(p)
            rows.append({
                "cache": label,
                "bucket": os.path.relpath(dirpath, root),
                "project": fn[:-5],
                "path": p,
                "rkyv_bytes": sz,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                       time.localtime(os.path.getmtime(p))),
                "url": url,
                "etag": etag,
                "accept": accept,
                "accept_encoding": aenc,
            })
    rows.sort(key=lambda r: (r["cache"], r["bucket"], r["project"]))
    return rows


CTX = ssl.create_default_context()


def fetch(url, accept, aenc, method="GET", timeout=60):
    req = urllib.request.Request(url, method=method)
    req.add_header("Accept", accept or "application/vnd.pypi.simple.v1+json, "
                   "application/vnd.pypi.simple.v1+html;q=0.2, text/html;q=0.01")
    req.add_header("Accept-Encoding", aenc or "gzip, deflate, zstd")
    req.add_header("User-Agent", "uv/0.12.5")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
            raw = resp.read() if method == "GET" else b""
            hdr = {k.lower(): v for k, v in resp.headers.items()}
            enc = hdr.get("content-encoding", "")
            wire = len(raw)
            if method == "HEAD":
                wire = int(hdr.get("content-length", 0) or 0)
                dec = wire
            elif enc == "gzip":
                try:
                    dec = len(gzip.decompress(raw))
                except Exception:
                    dec = wire
            elif enc == "deflate":
                try:
                    dec = len(zlib.decompress(raw))
                except Exception:
                    dec = wire
            else:
                dec = wire
            return {
                "status": resp.status, "wire_bytes": wire, "decoded_bytes": dec,
                "content_type": hdr.get("content-type"),
                "content_encoding": enc or None,
                "etag": hdr.get("etag"), "cache_control": hdr.get("cache-control"),
                "last_modified": hdr.get("last-modified"),
                "wall_ms": int((time.time() - t0) * 1000),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        # a 404 from an extra index is a RESULT, not a failure: the frozen
        # snapshot has to record it or the mirror invents a package.
        body = b""
        try:
            body = exc.read()
        except Exception:
            pass
        h = {k.lower(): v for k, v in (exc.headers or {}).items()}
        return {"status": exc.code, "wire_bytes": len(body),
                "decoded_bytes": len(body),
                "content_type": h.get("content-type"), "content_encoding": None,
                "etag": h.get("etag"), "cache_control": h.get("cache-control"),
                "last_modified": h.get("last-modified"),
                "wall_ms": int((time.time() - t0) * 1000), "error": None}
    except Exception as exc:  # noqa: BLE001 -- every failure is a row, never a default
        return {"status": None, "wire_bytes": 0, "decoded_bytes": 0,
                "content_type": None, "content_encoding": None, "etag": None,
                "cache_control": None, "last_modified": None,
                "wall_ms": int((time.time() - t0) * 1000),
                "error": f"{type(exc).__name__}: {exc}"}


def host_of(url):
    m = re.match(r"https?://([^/]+)", url or "")
    return m.group(1) if m else "?"


def main():
    inv = inventory(SIMPLE21, "pixi-embedded-uv/simple-v21")
    inv += inventory(SIMPLE24, "retread-uv-0.12.5/simple-v24")
    with open(os.path.join(OUT, "cache-inventory.json"), "w") as fh:
        json.dump(inv, fh, indent=1)

    print("### CACHE INVENTORY (on disk, the read set)")
    per = defaultdict(lambda: [0, 0])
    for r in inv:
        k = (r["cache"], host_of(r["url"]))
        per[k][0] += 1
        per[k][1] += r["rkyv_bytes"]
    for (cache, host), (n, b) in sorted(per.items()):
        print(f"###   cache={cache} host={host} pages={n} rkyv_bytes={b}")
    mt = Counter(r["mtime"][:16] for r in inv)
    print("### CACHE MTIMES (minute buckets, so the p6af-2g arm window is visible)")
    for k in sorted(mt):
        print(f"###   {k} {mt[k]}")

    # one full example of the recovered request headers -- the evidence for (2)
    for r in inv:
        if r["accept"]:
            print(f"### RECOVERED REQUEST HEADERS from {r['path']}")
            print(f"###   url={r['url']}")
            print(f"###   etag={r['etag']}")
            print(f"###   accept={r['accept']}")
            print(f"###   accept-encoding={r['accept_encoding']}")
            break

    print("### LIVE FETCH of every page in the read set")
    live = []
    seen = set()
    for r in inv:
        if not r["url"] or r["url"] in seen:
            continue
        seen.add(r["url"])
        res = fetch(r["url"], r["accept"], r["accept_encoding"])
        res.update(url=r["url"], host=host_of(r["url"]), project=r["project"],
                   cache=r["cache"], rkyv_bytes=r["rkyv_bytes"],
                   cached_etag=r["etag"])
        live.append(res)
    with open(os.path.join(OUT, "live-pages.json"), "w") as fh:
        json.dump(live, fh, indent=1)

    print("### PER-HOST PAGE TABLE (live, wire bytes = what a verbatim snapshot stores)")
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for r in live:
        a = agg[r["host"]]
        a[0] += 1
        a[1] += r["wire_bytes"]
        a[2] += r["decoded_bytes"]
        if r["status"] != 200:
            a[3] += 1
    tot_wire = tot_dec = tot_pages = 0
    for host, (n, w, d, bad) in sorted(agg.items()):
        print(f"###   host={host} pages={n} wire_bytes={w} decoded_bytes={d} non200={bad}")
        tot_pages += n
        tot_wire += w
        tot_dec += d
    print(f"### FROZEN PyPI INDEX SIZE pages={tot_pages} wire_bytes={tot_wire} "
          f"decoded_bytes={tot_dec} wire_MiB={tot_wire/1048576:.2f} "
          f"decoded_MiB={tot_dec/1048576:.2f}")
    ct = Counter((r["host"], str(r["content_type"])) for r in live)
    for (h, c), n in sorted(ct.items()):
        print(f"### CONTENT-TYPE host={h} type={c} n={n}")
    cc = Counter((r["host"], str(r["cache_control"])) for r in live)
    for (h, c), n in sorted(cc.items()):
        print(f"### CACHE-CONTROL host={h} value={c} n={n}")
    et = sum(1 for r in live if r["etag"])
    print(f"### ETAG present on {et}/{len(live)} live pages; "
          f"agreeing with the cached etag on "
          f"{sum(1 for r in live if r['etag'] and r['cached_etag'] and r['etag'].strip(chr(34)) == r['cached_etag'].strip(chr(34)))}")
    for r in live:
        if r["error"] or r["status"] != 200:
            print(f"### PAGE NON-200 url={r['url']} status={r['status']} error={r['error']}")

    # ---- (B2) THE EXTRA INDEXES. The canonical manifest declares
    #      [pypi-options] extra-index-urls = ["https://pypi.org/simple",
    #      "https://pypi.nvidia.com", "https://py.mujoco.org"], so a frozen
    #      snapshot must also record what those hosts answer for every project
    #      name -- INCLUDING the 404s, because a mirror that answers 200 where
    #      the real index answered 404 (or the reverse) changes the resolution.
    names = sorted({r["project"] for r in inv})
    print(f"### EXTRA-INDEX PROBE over {len(names)} project names "
          f"(the manifest's extra-index-urls, minus pypi.org which is above)")
    extra = []
    for base in ("https://pypi.nvidia.com", "https://py.mujoco.org"):
        for nm in names:
            u = f"{base}/{nm}/"
            res = fetch(u, None, "gzip, deflate")
            res.update(url=u, host=host_of(u), project=nm)
            extra.append(res)
    with open(os.path.join(OUT, "extra-index.json"), "w") as fh:
        json.dump(extra, fh, indent=1)
    eagg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in extra:
        a = eagg[r["host"]][str(r["status"])]
        a[0] += 1
        a[1] += r["wire_bytes"]
    extra_200_bytes = 0
    extra_200_pages = 0
    for host in sorted(eagg):
        for st in sorted(eagg[host]):
            n, b = eagg[host][st]
            print(f"###   extra-index host={host} status={st} pages={n} wire_bytes={b}")
            if st == "200":
                extra_200_pages += n
                extra_200_bytes += b
    print(f"### EXTRA-INDEX 200s pages={extra_200_pages} wire_bytes={extra_200_bytes}")
    print(f"### FROZEN PyPI SNAPSHOT TOTAL (default index pages + extra-index 200s) "
          f"pages={tot_pages + extra_200_pages} "
          f"wire_bytes={tot_wire + extra_200_bytes} "
          f"MiB={(tot_wire + extra_200_bytes)/1048576:.2f} "
          f"(plus {len(extra) - extra_200_pages} recorded NEGATIVE answers, "
          f"which cost a name each, not a page)")

    # ---- (C) the lock's artefact URLs
    lockurls = sys.argv[2] if len(sys.argv) > 2 else None
    if lockurls and os.path.exists(lockurls):
        urls = []
        for line in open(lockurls):
            line = line.strip()
            if not line or line == "./":
                continue
            u = line.split("+", 1)[1] if line.startswith("direct+") else line
            u = u.split("#", 1)[0]
            if u.startswith("http"):
                urls.append(u)
        print(f"### LOCK ARTEFACT URLS n={len(urls)} (from {lockurls})")
        aagg = defaultdict(lambda: [0, 0, 0])
        arts = []
        for u in urls:
            res = fetch(u, "*/*", "identity", method="HEAD")
            res["url"] = u
            res["host"] = host_of(u)
            arts.append(res)
            a = aagg[res["host"]]
            a[0] += 1
            a[1] += res["wire_bytes"]
            if res["status"] not in (200, 302, None) or res["error"]:
                a[2] += 1
        with open(os.path.join(OUT, "lock-artefacts.json"), "w") as fh:
            json.dump(arts, fh, indent=1)
        ta = tb = 0
        for host, (n, b, bad) in sorted(aagg.items()):
            print(f"###   artefact host={host} urls={n} bytes={b} problems={bad}")
            ta += n
            tb += b
        print(f"### LOCK ARTEFACT TOTAL urls={ta} bytes={tb} MiB={tb/1048576:.1f}")
    print("### MEASURE DONE rc=0")


if __name__ == "__main__":
    main()
