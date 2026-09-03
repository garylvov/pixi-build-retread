#!/usr/bin/env python3
"""Read an instrumented relock's logs and answer three questions.

  (a) per-environment uv candidate counts -- the top-10 packages by candidates
      tried, which is the table that does not exist anywhere in this campaign;
  (b) the timeline of the `isaac-pack-latest` uv closure call -- the single
      496.7 s `uv closure: resolving via uv` invocation that is 35 % of the
      conda phase in job 5618074;
  (c) the `pace` pypi resolve breakdown -- the ~1000 s frontend tail that is one
      environment, one resolver call, under `[concurrency] solves = 1`.

It is written to run against logs that DO NOT have the instrumentation too:
against 5618074 / 5611846 it must say "no uv_resolver rows" and keep going,
because that is the before-picture and the reason p6b exists.

Three traps this file is built around, all of them paid for already:

  * `resolved ... for environment 'X' ... in <t>` is elapsed since the PHASE
    started, not that environment's cost. Summing 27 of them gave 5360.9 s
    inside a 1412 s phase. This script never quotes those values as durations;
    it differences consecutive completion TIMES instead, and prints the
    sum-vs-wall check that would have caught the original misread.
  * a gap labelled with the row that PRECEDES it is not that row's cost. Gaps
    are printed with BOTH neighbours, always.
  * the frontend's rows carry no timestamp; the backend's carry a UTC `Z`
    stamp; the stamper adds a LOCAL one. Those are different clocks and the
    script keeps them apart, measuring the offset from rows that carry both.

Python 3.9 on this box.
"""
import argparse
import gzip
import io
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

# "2026-09-02T18:00:00.123456 " prepended by p6b_stamp.py (LOCAL time).
OUTER = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}) (.*)$", re.S)
# "2026-09-02T13:49:27.251821Z  INFO target: msg" -- the backend's own row (UTC).
INNER = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z\s+"
    r"(TRACE|DEBUG|INFO|WARN|ERROR)\s+([A-Za-z0-9_:]+):\s?(.*)$",
    re.S,
)
# " INFO pixi_core::lock_file::update: msg" -- a frontend row, no time of its own.
FRONT = re.compile(r"^\s*(TRACE|DEBUG|INFO|WARN|ERROR)\s+([A-Za-z0-9_:]+):\s?(.*)$", re.S)

ENV_DONE = re.compile(
    r"resolved (pypi packages|conda environment) for environment '([^']+)' '([^']+)' in (.*)$"
)


def parse_ts(text):
    try:
        return datetime.strptime(text[:26], "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return None


class Row(object):
    __slots__ = ("n", "local", "utc", "level", "target", "msg", "raw")

    def __init__(self, n, local, utc, level, target, msg, raw):
        self.n = n
        self.local = local
        self.utc = utc
        self.level = level
        self.target = target
        self.msg = msg
        self.raw = raw

    def when(self):
        """The row's own clock, whichever it has. Never mix the two in maths."""
        return self.utc if self.utc is not None else self.local

    def clock(self):
        return "utc" if self.utc is not None else ("local" if self.local else "none")


def read_rows(path):
    if path is None:
        return []
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rb") as fh:
        for n, raw in enumerate(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), 1):
            line = ANSI.sub("", raw.rstrip("\n"))
            if not line.strip():
                continue
            local = None
            body = line
            m = OUTER.match(line)
            if m:
                local = parse_ts(m.group(1))
                body = m.group(2)
            utc = None
            level = target = None
            msg = body
            m = INNER.match(body)
            if m:
                utc = parse_ts(m.group(1))
                level, target, msg = m.group(2), m.group(3), m.group(4)
            else:
                m = FRONT.match(body)
                if m:
                    level, target, msg = m.group(1), m.group(2), m.group(3)
            rows.append(Row(n, local, utc, level, target, msg, body))
    return rows


def secs(a, b):
    return (b - a).total_seconds()


def fmt(t):
    return t.strftime("%H:%M:%S.%f")[:-3] if t is not None else "--:--:--"


def clip(s, n=110):
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------
# uv rows
# --------------------------------------------------------------------------
UV_TARGET = re.compile(r"^uv_")

# uv's resolver messages, most specific first. Each yields a package name.
# Anything that matches none of these is reported as an UNMATCHED SHAPE rather
# than dropped -- an undercount that hides itself is worse than a gap in a table.
CANDIDATE_PATTERNS = [
    ("searching", re.compile(r"Searching for a (?:compatible )?version of ([A-Za-z0-9._-]+)")),
    ("selecting", re.compile(r"Selecting(?: candidate for)?:?\s+([A-Za-z0-9._-]+)")),
    ("skipping", re.compile(r"Skipping [^ ]+ of(?: package)? ([A-Za-z0-9._-]+)")),
    ("candidate", re.compile(r"Found candidate(?: for)?:?\s+([A-Za-z0-9._-]+)")),
    ("prefetch", re.compile(r"(?:Prefetching|Batch prefetch)[^A-Za-z0-9]*([A-Za-z0-9._-]+)")),
    ("fetching", re.compile(r"(?:Fetching|Requesting|No cache entry for)[^A-Za-z0-9]*([A-Za-z0-9._-]+)")),
    ("solving", re.compile(r"Solving[^A-Za-z0-9]*([A-Za-z0-9._-]+)")),
    ("adding", re.compile(r"Adding (?:transitive |direct )?dependency[^A-Za-z0-9]*([A-Za-z0-9._-]+)")),
]
# Shapes that count as "a candidate was tried" rather than as bookkeeping.
TRIED = ("searching", "selecting", "skipping", "candidate")


def classify_uv(msg):
    for kind, pat in CANDIDATE_PATTERNS:
        m = pat.search(msg)
        if m:
            return kind, m.group(1)
    return None, None


def env_segments(rows):
    """Serialized-phase attribution.

    `[concurrency] solves = 1` serializes the resolves, so a uv row belongs to
    the environment whose completion row comes next. Segment i therefore runs
    from completion i-1 to completion i. This is an ATTRIBUTION, not a label
    uv itself writes; it is only sound while solves = 1, and the caller prints
    that caveat.
    """
    done = []
    for r in rows:
        m = ENV_DONE.search(r.msg or "")
        if m:
            done.append((r, m.group(1), m.group(2), m.group(3), m.group(4)))
    return done


def section_a(out, lock_rows, backend_rows):
    out("=" * 78)
    out("(a) PER-ENV uv CANDIDATE COUNTS")
    out("=" * 78)
    uv_rows = [r for r in lock_rows + backend_rows if r.target and UV_TARGET.match(r.target)]
    if not uv_rows:
        out("no uv_resolver rows in these logs.")
        out("")
        out("  That is the BEFORE picture, not a bug: `pixi lock -v` emits pixi_core::*")
        out("  only. uv's resolver tracing needs `-vvv` (pixi 0.73: -v warn, -vv info,")
        out("  -vvv debug) and/or RUST_LOG naming uv_resolver / uv_client, which is what")
        out("  p6b_relock.sh sets. No per-package candidate table can be produced from")
        out("  these logs and none should be quoted from them.")
        out("")
        by_target = Counter(r.target for r in lock_rows + backend_rows if r.target)
        out("  targets present, most frequent first:")
        for target, n in by_target.most_common(12):
            out("    %-58s %8d" % (target, n))
        return
    out("uv rows: %d  (targets: %s)"
        % (len(uv_rows), ", ".join("%s=%d" % kv for kv in
                                   Counter(r.target for r in uv_rows).most_common())))
    segs = env_segments(lock_rows)
    out("")
    out("ATTRIBUTION: uv rows carry no environment of their own. With")
    out("`[concurrency] solves = 1` the resolves are serialized, so each row is")
    out("attributed to the environment whose completion row comes next. If that")
    out("setting ever changes this attribution is void.")
    out("")
    per_env = defaultdict(Counter)
    per_env_kind = defaultdict(Counter)
    unmatched = Counter()
    idx = 0
    order = [(r.n, r) for r in uv_rows]
    bounds = [(d[0].n, d[2]) for d in segs]
    for n, r in order:
        env = "(after the last env)"
        for line_no, name in bounds:
            if n <= line_no:
                env = name
                break
        kind, pkg = classify_uv(r.msg or "")
        if kind is None:
            unmatched[clip((r.msg or "").split(" ")[0], 40)] += 1
            continue
        per_env_kind[env][kind] += 1
        if kind in TRIED:
            per_env[env][pkg] += 1
        idx += 1
    for env in sorted(per_env, key=lambda e: -sum(per_env[e].values())):
        total = sum(per_env[env].values())
        out("-- %-40s candidates tried: %d" % (env, total))
        out("   kinds: %s" % ", ".join("%s=%d" % kv for kv in per_env_kind[env].most_common()))
        for pkg, n in per_env[env].most_common(10):
            out("     %-46s %8d" % (pkg, n))
    if unmatched:
        out("")
        out("UNMATCHED uv message shapes (extend CANDIDATE_PATTERNS; these are NOT counted):")
        for shape, n in unmatched.most_common(8):
            out("   %-52s %8d" % (shape, n))


def gaps_of(rows, threshold, out, limit=25):
    """Print inter-row silences with BOTH neighbours. A gap belongs to neither."""
    timed = [r for r in rows if r.when() is not None]
    found = []
    for a, b in zip(timed, timed[1:]):
        d = secs(a.when(), b.when())
        if d >= threshold:
            found.append((d, a, b))
    found.sort(key=lambda t: -t[0])
    if not found:
        out("   (no silences >= %.0fs)" % threshold)
        return 0.0
    for d, a, b in found[:limit]:
        out("   %8.1fs  %s -> %s" % (d, fmt(a.when()), fmt(b.when())))
        out("             before: %s" % clip(a.msg or a.raw))
        out("             after : %s" % clip(b.msg or b.raw))
    return sum(d for d, _, _ in found)


def section_b(out, backend_rows, bundle, threshold):
    out("")
    out("=" * 78)
    out("(b) TIMELINE OF THE `%s` uv CLOSURE CALL" % bundle)
    out("=" * 78)
    if not backend_rows:
        out("no backend log rows.")
        return
    tagged = [r for r in backend_rows if bundle in (r.msg or "")]
    out("backend rows naming %s: %d   (clock: UTC, the backend's own stamp)"
        % (bundle, len(tagged)))
    if not tagged:
        out("  the bundle is not named in this log; nothing to time.")
        return
    closure = [r for r in tagged if "resolving via uv" in (r.msg or "")]
    out("`uv closure: resolving via uv` invocations for it: %d" % len(closure))
    timed = [r for r in backend_rows if r.when() is not None]
    for r in closure:
        nxt = None
        for other in timed:
            if other.n > r.n:
                nxt = other
                break
        out("   at %s  -> next backend row %s  (%s)"
            % (fmt(r.when()),
               fmt(nxt.when()) if nxt else "none",
               ("%.1fs of silence" % secs(r.when(), nxt.when())) if nxt and r.when() else "?"))
        if nxt:
            out("      the row that ENDS the silence: %s" % clip(nxt.msg or nxt.raw))
    out("")
    out("bench spans this bundle emitted:")
    for r in tagged:
        if (r.msg or "").startswith("bench:"):
            out("   %s  %s" % (fmt(r.when()), clip(r.msg)))
    out("")
    out("silences >= %.0fs between consecutive rows of this bundle:" % threshold)
    total = gaps_of(tagged, threshold, out)
    if tagged and tagged[0].when() and tagged[-1].when():
        wall = secs(tagged[0].when(), tagged[-1].when())
        out("")
        out("   SANITY (the check that catches a misread): silences total %.1fs inside a"
            % total)
        out("   %.1fs window for this bundle -- %.0f%%. A per-item sum larger than its own"
            % (wall, 100.0 * total / wall if wall else 0.0))
        out("   phase wall means the items are offsets, not durations.")


def section_c(out, lock_rows, backend_rows, env, threshold):
    out("")
    out("=" * 78)
    out("(c) `%s` RESOLVE BREAKDOWN" % env)
    out("=" * 78)
    segs = env_segments(lock_rows)
    if not segs:
        out("no `resolved ... for environment ...` rows in the lock log.")
        return
    out("environment completion rows: %d" % len(segs))
    stamped = [d for d in segs if d[0].local is not None]
    if not stamped:
        out("")
        out("the lock log is NOT stamped -- pixi's own frontend rows carry no time, so")
        out("these rows can only be ordered, not timed. The `in <t>` value on each row is")
        out("elapsed since the PHASE started (a completion OFFSET), never that")
        out("environment's cost; summing 27 of them gave 5360.9s inside a 1412s phase on")
        out("job 5618074. Run through p6b_stamp.py to get real times.")
        for row, kind, name, plat, dur in segs:
            if name == env:
                out("   %-8s %-28s offset-as-printed: %s" % (kind.split()[0], name, dur.strip()))
        return
    out("")
    out("per-environment cost by DIFFERENCING consecutive completion times")
    out("(the printed `in <t>` is an offset and is shown only for comparison):")
    prev = None
    rows_out = []
    for row, kind, name, plat, dur in segs:
        t = row.local
        cost = secs(prev, t) if prev is not None else 0.0
        rows_out.append((cost, kind, name, plat, dur.strip(), row))
        prev = t
    for cost, kind, name, plat, dur, row in sorted(rows_out, key=lambda r: -r[0])[:15]:
        out("   %9.1fs  %-6s %-28s %-26s printed-offset=%s"
            % (cost, kind.split()[0], name, plat, dur))
    total = sum(r[0] for r in rows_out)
    first, last = segs[0][0].local, segs[-1][0].local
    wall = secs(first, last)
    out("")
    out("   SANITY: differenced costs sum to %.1fs against a %.1fs completion-row window."
        % (total, wall))
    out("   These must agree. Printed offsets summed would not.")
    target = [r for r in rows_out if r[2] == env]
    if not target:
        out("   environment '%s' has no completion row in this log." % env)
        return
    for cost, kind, name, plat, dur, row in target:
        out("")
        out("   %s / %s: %.1fs (printed offset %s)" % (name, kind.split()[0], cost, dur))
        start_n = None
        for c2, k2, n2, p2, d2, r2 in rows_out:
            if r2.n < row.n:
                start_n = r2.n
        window = [r for r in lock_rows if (start_n or 0) < r.n <= row.n]
        by_target = Counter(r.target for r in window if r.target)
        out("   rows inside that window: %d" % len(window))
        for tgt, n in by_target.most_common(10):
            out("      %-52s %8d" % (tgt, n))
        uv_window = [r for r in window if r.target and UV_TARGET.match(r.target)]
        if not uv_window:
            out("      no uv rows inside the window -- nothing to attribute the time to.")
            out("      This is precisely the hole p6b's RUST_LOG is meant to fill.")
        else:
            pkgs = Counter()
            for r in uv_window:
                kind2, pkg = classify_uv(r.msg or "")
                if kind2 in TRIED:
                    pkgs[pkg] += 1
            out("      top packages by candidates tried:")
            for pkg, n in pkgs.most_common(10):
                out("        %-48s %8d" % (pkg, n))
        out("   silences >= %.0fs inside the window:" % threshold)
        gaps_of(window, threshold, out, limit=10)
    tail = [r for r in lock_rows if "Updated lock file" in (r.raw or "")]
    last_backend = None
    for r in lock_rows:
        if r.utc is not None:
            last_backend = r
    if tail and last_backend is not None and tail[-1].local and last_backend.local:
        out("")
        out("   TAIL: last backend row -> `Updated lock file` = %.1fs with no backend row"
            % secs(last_backend.local, tail[-1].local))
        out("   in it (the window in which the frontend is alone).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock", required=True, help="lock log (pixi frontend + interleaved backend)")
    ap.add_argument("--backend", default=None, help="backend stderr log (timestamped)")
    ap.add_argument("--bundle", default="isaac-pack-latest")
    ap.add_argument("--env", default="pace")
    ap.add_argument("--gap", type=float, default=20.0, help="silence threshold, seconds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    lock_rows = read_rows(args.lock)
    backend_rows = read_rows(args.backend) if args.backend else []

    buf = []

    def out(line=""):
        buf.append(line)

    out("p6b extract -- lock=%s (%d rows)  backend=%s (%d rows)"
        % (args.lock, len(lock_rows), args.backend, len(backend_rows)))
    both = [r for r in lock_rows if r.local is not None and r.utc is not None]
    if both:
        off = secs(both[0].local, both[0].utc)
        out("clock offset measured on a row carrying BOTH stamps: utc - local = %.0fs"
            % off)
    else:
        out("no row carries both a stamper time and a backend UTC time; clocks are kept apart.")
    out("")
    section_a(out, lock_rows, backend_rows)
    section_b(out, backend_rows or lock_rows, args.bundle, args.gap)
    section_c(out, lock_rows, backend_rows, args.env, args.gap)

    text = "\n".join(buf) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
