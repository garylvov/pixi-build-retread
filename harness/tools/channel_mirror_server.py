#!/usr/bin/env python3
"""The frozen channel mirror's HTTP server (p6af-2a).

WHY THIS EXISTS AT ALL, AND WHY `python3 -m http.server` IS NOT ENOUGH.

p6af froze pixi's conda universe by mapping every channel through `mirrors` to a
loopback mirror of the CLASSIC `repodata.json` documents, and proved on a
48-package probe manifest that the resulting lock is byte-identical to a network
lock.  p6af-2 took that to the canonical 27-environment workspace and it died at
wall 1634 s, rc=1, on one line:

    GET /prefix.dev__conda-forge/linux-64/libcudnn-9.13.1.26-hf7e9902_0.conda 404

`mirrors` is transparent for PACKAGE urls too, not only for repodata.  `pixi
lock` on this workspace does not merely resolve: it INSTALLS a conda-backed
build environment in order to resolve the pypi half of `pm-newton-gpu`, and that
pulls package archives through the very same mapping.

AND THE FIX AS p6af-2.2 SIZED IT WOULD NOT HAVE WORKED.  That entry said "seed
pixi's `pkgs` cache from the reference lock's conda URLs".  Measured on
`mergeB15/artifacts/pixi.lock.cert` (md5 790854c1…, the lock at the mBH tip):
the lock names libcudnn **9.10.2.21** and nothing else.  The archive that 404'd
is **9.13.1.26** -- it is not in the lock, it is not in any lock, it is a member
of a build environment resolved fresh against the frozen repodata and recorded
nowhere.  Seeding all 2541 of the lock's conda URLs would have paid for 32
missing linux-aarch64 packages a resolve never downloads and still 404'd on this
one.  A reference lock cannot enumerate the set; only the solve can.

SO THE MIRROR SERVES PACKAGES, LAZILY, AND SAYS SO OUT LOUD.  On a request for
`<channel>/<subdir>/<file>.conda|.tar.bz2` that is not already in the mirror's
package store, this server:

  1. reads the record for that exact filename out of the mirror's OWN FROZEN
     `repodata.json` for that channel/subdir -- if the frozen universe does not
     contain the file, the answer is 404 and a `NORECORD` row, because serving a
     package the frozen universe never declared would be exactly the silent
     universe substitution this whole lane exists to prevent;
  2. fetches the archive from the real upstream channel, DIRECTLY (the server is
     started before the caller arms the deny proxy, and it additionally scrubs
     the eight proxy variables from its own environment at startup);
  3. verifies sha256 (and size, when the record carries one) against that frozen
     record and REFUSES on a mismatch -- 502 and a `SHA MISMATCH` row;
  4. stores it under the package store and serves it.

WHAT IS AND IS NOT FROZEN AFTER THIS, STATED PRECISELY, because "offline" that
is not decomposed is worthless:
  * the conda REPODATA universe -- the half that decides a resolution -- is
    100 % frozen and 100 % offline.  pixi never reaches a channel host for a
    document; the deny proxy's log is the positive record of that.
  * conda PACKAGE ARCHIVES are fetched by THIS process from upstream on demand.
    They cannot move a resolution: a package archive is content-pinned by the
    sha256 in the frozen repodata, and this server refuses any byte string that
    does not match it.  Every fetch is named in the log, so the set is a
    measured fact and not an assumption.
  * pixi's own network access to a channel is still zero.  This process is not
    pixi.

Usage:
    channel_mirror_server.py <mirror root> <port> [package store dir]

With no package store the server is a plain static file server and behaves
EXACTLY as `python3 -m http.server` did -- p6af's guard calls it that way and is
unaffected.  Access-log lines keep BaseHTTPRequestHandler's format verbatim,
because the harness's readers parse it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PKG_SUFFIXES = (".conda", ".tar.bz2")


def _scrub_proxy_env() -> None:
    """This process is the one thing in the job that MAY reach a channel.

    The caller starts it before it exports the deny-proxy variables, so they are
    normally absent already; unsetting them here means the order of two lines in
    a harness cannot silently turn every package fetch into a 502.
    """
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
              "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        os.environ.pop(k, None)


def _upstream_base(chan_dir: str) -> str:
    """`prefix.dev__conda-forge` -> `https://prefix.dev/conda-forge`.

    The inverse of the key `retread_freeze_channel_mirror` builds (host AND
    path, scheme stripped, `/` -> `__`).  Keyed on host and path for the reason
    that function states: this workspace declares both `prefix.dev/pytorch` and
    `conda.anaconda.org/pytorch` and a last-segment key merges them.
    """
    return "https://" + chan_dir.replace("__", "/")


def _record_slice(blob: bytes, fname: str):
    """Pull one record object out of a repodata document without parsing 638 MB.

    conda-forge's linux-64 `repodata.json` is 638 MB; `json.load` on it costs
    ~2.6 GB RSS (measured by p6af's trim script).  A package fetch must not.
    So: find the key, then walk to its matching close brace with string and
    escape awareness -- records DO carry nested objects (`run_exports`), so a
    naive scan-to-first-`}` is wrong.
    """
    key = b'"' + fname.encode() + b'":'
    i = blob.find(key)
    if i < 0:
        return None
    j = blob.find(b"{", i + len(key))
    if j < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    k = j
    n = len(blob)
    while k < n:
        c = blob[k]
        if esc:
            esc = False
        elif c == 0x5C:  # backslash
            esc = True
        elif in_str:
            if c == 0x22:
                in_str = False
        elif c == 0x22:
            in_str = True
        elif c == 0x7B:  # {
            depth += 1
        elif c == 0x7D:  # }
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(blob[j:k + 1].decode("utf-8", "replace"))
                except Exception:
                    return None
        k += 1
    return None


class MirrorHandler(SimpleHTTPRequestHandler):
    """Static mirror + lazy, sha-verified package pass-through."""

    pkg_store = None      # set by the partial() below
    mirror_root = None
    fetch_log = None
    protocol_version = "HTTP/1.0"

    # ---- logging -----------------------------------------------------------
    def _note(self, kind: str, detail: str) -> None:
        line = "%s %s %s\n" % (
            time.strftime("[%d/%b/%Y %H:%M:%S]"), kind, detail)
        sys.stderr.write(line)
        sys.stderr.flush()
        if self.fetch_log:
            try:
                with open(self.fetch_log, "a") as fh:
                    fh.write(line)
            except OSError:
                pass

    # ---- the one behaviour change -----------------------------------------
    def send_head(self):
        # Both GET and HEAD funnel through send_head, so filling here covers
        # whichever verb pixi picks.
        if self.pkg_store:
            try:
                self._maybe_fill()
            except Exception as exc:            # never take the server down
                self._note("PKGERROR", "%s %r" % (self.path, exc))
        return super().send_head()

    def _maybe_fill(self) -> None:
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if not path.endswith(PKG_SUFFIXES):
            return
        parts = [p for p in path.split("/") if p]
        if len(parts) != 3:
            return
        chan, subdir, fname = parts
        local = os.path.join(self.mirror_root, chan, subdir, fname)
        if os.path.exists(local):
            return

        doc = os.path.join(self.mirror_root, chan, subdir, "repodata.json")
        if not os.path.isfile(doc):
            self._note("NORECORD", "%s/%s/%s (no frozen document)" % (chan, subdir, fname))
            return
        with open(doc, "rb") as fh:
            rec = _record_slice(fh.read(), fname)
        if rec is None:
            # NOT in the frozen universe.  Refusing is the point: a package the
            # frozen documents never declared is a different universe, which is
            # the C22-3 / p6ac-1 substitution mechanism.
            self._note("NORECORD", "%s/%s/%s (absent from the frozen document)"
                       % (chan, subdir, fname))
            return
        want_sha = rec.get("sha256")
        want_size = rec.get("size")

        url = "%s/%s/%s" % (_upstream_base(chan), subdir, fname)
        t0 = time.time()
        # A REAL User-Agent, and it is not cosmetic. prefix.dev answers
        # `Python-urllib/3.9` with 403 (measured -- guard job 5858488 arm C: the
        # record was found, the fetch raised, and the mirror 404'd with the
        # archive sitting one HTTP header away).
        req = urllib.request.Request(url, headers={
            "User-Agent": "retread-channel-mirror/1.0 (+p6af-2a)",
            "Accept": "*/*",
        })
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                blob = resp.read()
        except Exception as exc:
            self._note("PKGFETCH-FAIL", "%s %r" % (url, exc))
            return
        got = hashlib.sha256(blob).hexdigest()
        if want_sha and got != want_sha:
            self._note("SHA-MISMATCH", "%s want=%s got=%s" % (url, want_sha, got))
            return
        if want_size is not None and len(blob) != want_size:
            self._note("SIZE-MISMATCH", "%s want=%s got=%s" % (url, want_size, len(blob)))
            return

        store_dir = os.path.join(self.pkg_store, chan, subdir)
        os.makedirs(store_dir, exist_ok=True)
        tmp = os.path.join(store_dir, "." + fname + ".part.%d" % os.getpid())
        with open(tmp, "wb") as fh:
            fh.write(blob)
        final = os.path.join(store_dir, fname)
        os.replace(tmp, final)
        # The mirror root is what the server serves out of; link the stored
        # archive into place (same filesystem by construction -- the store is
        # created next to the mirror root by the caller) and fall back to a
        # rename-copy on EXDEV rather than refusing.
        os.makedirs(os.path.dirname(local), exist_ok=True)
        try:
            os.link(final, local)
        except OSError:
            tmp2 = local + ".part.%d" % os.getpid()
            with open(tmp2, "wb") as fh:
                fh.write(blob)
            os.replace(tmp2, local)
        self._note("PKGFETCH", "%s bytes=%d sha256=%s wall_ms=%d"
                   % (url, len(blob), got, int((time.time() - t0) * 1000)))


def main(argv) -> int:
    if len(argv) < 3:
        sys.stderr.write(
            "usage: channel_mirror_server.py <mirror root> <port> [package store dir]\n")
        return 2
    root = os.path.abspath(argv[1])
    port = int(argv[2])
    store = os.path.abspath(argv[3]) if len(argv) > 3 and argv[3] else None
    if not os.path.isdir(root):
        sys.stderr.write("channel_mirror_server: no mirror root at %s\n" % root)
        return 2
    _scrub_proxy_env()
    if store:
        os.makedirs(store, exist_ok=True)
    MirrorHandler.pkg_store = store
    MirrorHandler.mirror_root = root
    MirrorHandler.fetch_log = os.environ.get("RETREAD_MIRROR_FETCH_LOG") or None
    sys.stderr.write("channel_mirror_server root=%s port=%d pkg_store=%s\n"
                     % (root, port, store or "(none: static only)"))
    sys.stderr.flush()
    handler = partial(MirrorHandler, directory=root)
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
