#!/usr/bin/env python3
"""A loopback HTTP proxy that REFUSES a named set of hosts and passes the rest.

    conda_deny_proxy.py <port> <deny host>[,<deny host>...] [log path]

WHY, measured 2026-09-04/05 across jobs 5851478, 5853300, 5853901, 5854363 and
5855234. p6af blocked pixi's network with a DEAD PROXY plus a `NO_PROXY`
allowlist, which is exact on a 48-package conda-only probe manifest and is the
wrong instrument for the canonical workspace: everything pixi legitimately needs
on the PyPI side has to be enumerated in advance, and each miss costs a whole
job. The list grew four times in one night -- `conda-mapping.prefix.dev` (the
conda->PyPI name mapping, not a channel), `miropsota.github.io` (a find-links
page for pytorch3d), then the release asset that page points at on github.com,
which redirects to a `*.githubusercontent.com` host that was not on the list
either. Each one killed a run after the CONDA half had already resolved.

The experiment only ever wanted one thing: pixi must not reach a CONDA CHANNEL.
So state that instead of its complement. This proxy denies exactly the channel
hosts and tunnels everything else, and its log is a POSITIVE record of every
host pixi asked for -- strictly better evidence than "no error appeared".

It is CONNECT-only plus a refusal for absolute-form plain HTTP, which is all a
rust HTTP client behind `HTTPS_PROXY` ever sends for https:// URLs.

THE READERS, both run by the harness before the lock:
  * a request to a denied host must FAIL (non-vacuity),
  * a request to an allowed host must SUCCEED (not over-blocking).
Either one alone is passable by a broken proxy; the pair is not.
"""
import os
import select
import socket
import sys
import threading

DENY = set()
LOG_LOCK = threading.Lock()
LOG = None


def log(line):
    with LOG_LOCK:
        LOG.write(line + "\n")
        LOG.flush()


def pump(a, b):
    try:
        while True:
            r, _, _ = select.select([a, b], [], [], 300)
            if not r:
                return
            for s in r:
                d = s.recv(65536)
                if not d:
                    return
                (b if s is a else a).sendall(d)
    except OSError:
        return


def handle(conn):
    conn.settimeout(60)
    try:
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = conn.recv(4096)
            if not d:
                return
            buf += d
            if len(buf) > 65536:
                return
        head = buf.split(b"\r\n", 1)[0].decode("latin-1")
        parts = head.split()
        if len(parts) < 2:
            return
        method, target = parts[0], parts[1]
        if method != "CONNECT":
            host = target.split("://", 1)[-1].split("/", 1)[0].split(":")[0]
            log("DENY-NON-CONNECT %s %s" % (method, host))
            conn.sendall(b"HTTP/1.1 501 Not Implemented\r\n\r\n")
            return
        host, _, port = target.partition(":")
        port = int(port or 443)
        if host in DENY:
            log("DENY %s:%d" % (host, port))
            conn.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        try:
            up = socket.create_connection((host, port), timeout=30)
        except OSError as e:
            log("FAIL %s:%d %s" % (host, port, e))
            conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        log("ALLOW %s:%d" % (host, port))
        conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        conn.settimeout(None)
        up.settimeout(None)
        pump(conn, up)
        up.close()
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def main():
    global LOG
    port = int(sys.argv[1])
    for h in sys.argv[2].split(","):
        h = h.strip()
        if h:
            DENY.add(h)
    LOG = open(sys.argv[3], "a") if len(sys.argv) > 3 else sys.stderr
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(256)
    log("proxy listening on 127.0.0.1:%d deny=%s pid=%d" % (port, ",".join(sorted(DENY)), os.getpid()))
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
