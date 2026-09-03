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
# "2026-09-02T13:49:27.251821Z  INFO <prefix>: msg" -- the backend's own row (UTC).
INNER = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z\s+"
    r"(TRACE|DEBUG|INFO|WARN|ERROR)\s+(.*)$",
    re.S,
)
# " INFO <prefix>: msg" -- a row with no time of its own (pixi's frontend).
FRONT = re.compile(r"^\s*(TRACE|DEBUG|INFO|WARN|ERROR)\s+(.*)$", re.S)

# THE DEFECT THIS REPLACES (LANE-SPEED-LOG 23:10 debt item 2): the old regexes
# read the target as a single `[A-Za-z0-9_:]+` token straight after the level.
# tracing's fmt layer does not print that. It prints the enclosing SPANS first:
#
#   DEBUG solve: uv_resolver::resolver: Adding direct dependency: mujoco>=3.3.3
#   TRACE resolve_pypi{group=gpu platform=linux-64-cuda-12}: uv_resolver::resolver: ...
#
# so every one of the 42,563 `uv_resolver` rows in job 5656622 was filed under
# target `solve` (or, once a span carried `{fields}`, under NO target at all,
# because `{` is not in that character class). `UV_TARGET.match` then matched
# nothing and the script reported "no uv_resolver rows" on a log full of them.
#
# The prefix is a chain of `segment: ` pieces, and the LAST one is the target.
# But a message may itself open with a space-free word and a colon (the
# backend's own `bench: conda_outputs ...` rows do), so taking the last segment
# unconditionally would eat it. Rule that survives both: consume the chain,
# then prefer the last segment that contains `::` -- a crate path -- and fall
# back to the last segment only when none does. `bench` has no `::`, so
# `pixi_build_retread::uv_closure: bench: ...` keeps target and message intact.
# Spans NEST, and tracing joins them with a bare `:` and no space:
#   resolve_pypi{group=gpu ...}:process_request{request=Prefetch mujoco *}: uv_resolver::…
# so the separator is a colon followed by a space OR immediately by the next
# segment. Requiring the space dropped 11,250 of the 42,563 uv rows in job
# 5656622 -- every prefetch row, which is exactly the fetch-vs-think evidence.
SEGMENT = re.compile(r"^([A-Za-z0-9_]+(?:::[A-Za-z0-9_]+)*)(\{[^}]*\})?:(?:[ ]|(?=[A-Za-z0-9_]))")
SPAN_FIELD = re.compile(r"(\w+)=([^ }]+)")


def split_prefix(body):
    """-> (spans, target, fields, message). Never raises; worst case all None."""
    segs = []
    rest = body
    while True:
        m = SEGMENT.match(rest)
        if not m:
            break
        segs.append((m.group(1), m.group(2), rest))
        rest = rest[m.end():]
    if not segs:
        return [], None, {}, body
    pick = len(segs) - 1
    for i, (name, _fields, _r) in enumerate(segs):
        if "::" in name:
            pick = i
    name, fields, at = segs[pick]
    # The message is everything after the CHOSEN segment's colon-space.
    consumed = SEGMENT.match(at)
    msg = at[consumed.end():]
    spans = [seg[0] + (seg[1] or "") for seg in segs[:pick]]
    parsed_fields = {}
    for seg_name, seg_fields, _r in segs[:pick + 1]:
        if seg_fields:
            for k, v in SPAN_FIELD.findall(seg_fields):
                parsed_fields[k] = v
    return spans, name, parsed_fields, msg

ENV_DONE = re.compile(
    r"resolved (pypi packages|conda environment) for environment '([^']+)' '([^']+)' in (.*)$"
)


def parse_ts(text):
    try:
        return datetime.strptime(text[:26], "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return None


class Row(object):
    __slots__ = ("n", "local", "utc", "level", "target", "msg", "raw", "spans", "fields")

    def __init__(self, n, local, utc, level, target, msg, raw, spans=None, fields=None):
        self.n = n
        self.local = local
        self.utc = utc
        self.level = level
        self.target = target
        self.msg = msg
        self.raw = raw
        self.spans = spans or []
        self.fields = fields or {}

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
            spans, fields = [], {}
            m = INNER.match(body)
            if m:
                utc = parse_ts(m.group(1))
                level = m.group(2)
                spans, target, fields, msg = split_prefix(m.group(3))
            else:
                m = FRONT.match(body)
                if m:
                    level = m.group(1)
                    spans, target, fields, msg = split_prefix(m.group(2))
            rows.append(Row(n, local, utc, level, target, msg, body, spans, fields))
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
    # Shapes measured against job 5656622's 42,563 uv rows, most specific first.
    ("selecting", re.compile(r"Selecting candidate for ([A-Za-z0-9._-]+)")),
    ("searching", re.compile(r"Searching for a compatible version of ([A-Za-z0-9._-]+)")),
    ("found", re.compile(r"Found candidate for package ([A-Za-z0-9._-]+)")),
    ("returning", re.compile(r"Returning candidate for package ([A-Za-z0-9._-]+)")),
    ("chosen", re.compile(r"Selecting: ([A-Za-z0-9._-]+)==")),
    ("skipping", re.compile(r"Skipping [^ ]+ of(?: package)? ([A-Za-z0-9._-]+)")),
    ("adding", re.compile(r"Adding (?:transitive |direct )?dependency(?: for)?:? ([A-Za-z0-9._-]+)")),
    ("metadata", re.compile(r"Received (?:package|built distribution|source distribution) metadata for: ([A-Za-z0-9._-]+)")),
    ("conflict", re.compile(r"Recording unit propagation conflict of ([A-Za-z0-9._-]+)")),
    ("decision", re.compile(r"Chose package for decision: ([A-Za-z0-9._-]+)")),
    ("edge", re.compile(r"Resolution edge: [^ ]+ -> ([A-Za-z0-9._-]+)")),
    ("prefetch", re.compile(r"(?:Prefetching|Batch prefetch)[^A-Za-z0-9]*([A-Za-z0-9._-]+)")),
    ("fetching", re.compile(r"(?:Fetching|Requesting|No cache entry for)[^A-Za-z0-9]*([A-Za-z0-9._-]+)")),
]
# Shapes that count as "a candidate was tried" rather than as bookkeeping.
# `selecting` is the candidate_selector actually being asked for a version of a
# package; `found`/`returning` are that same ask answered, so counting all three
# would triple every package. Only the ask is counted.
TRIED = ("selecting", "searching", "skipping")


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

    # ---- the sound half: uv rows that carry their own group -----------------
    exact = Counter()
    exact_pkgs = defaultdict(Counter)
    for r in uv_rows:
        g = r.fields.get("group")
        if not g:
            continue
        key = "%s %s" % (g, r.fields.get("platform", ""))
        exact[key] += 1
        kind, pkg = classify_uv(r.msg or "")
        if kind in TRIED:
            exact_pkgs[key][pkg] += 1
    spanned = sum(exact.values())
    out("ATTRIBUTION, and it is the finding of this section as much as the counts:")
    out("only %d of %d uv rows (%.0f%%) carry a group of their own. Those come from"
        % (spanned, len(uv_rows), 100.0 * spanned / len(uv_rows)))
    out("pixi's `resolve_pypi{group=... platform=...}` span. The other %d sit under a"
        % (len(uv_rows) - spanned))
    out("bare `solve:` span: uv runs its resolver on its own task, which does not")
    out("inherit pixi's span, so uv's OWN rows say nothing about which environment")
    out("asked for them.")
    switches = 0
    prev = None
    for r in lock_rows:
        g = r.fields.get("group")
        if g and g != prev:
            switches += 1
            prev = g
    out("")
    out("And the ordering fallback this script used to apply -- attribute a row to")
    out("the environment whose completion row comes next -- is NOT sound here: the")
    out("group span switches %d times across %d groups, so the resolves INTERLEAVE."
        % (switches, len(exact)))
    out("An ordering attribution over interleaved work parks thousands of rows on")
    out("whichever environment happens to finish next. No per-environment table is")
    out("printed for those rows, and none should be quoted from this run.")
    out("")
    out("uv rows per group, from uv's OWN span field (no attribution, no guess):")
    for name, n in exact.most_common(40):
        out("   %-46s %8d" % (name, n))

    # ---- whole-lock package counts: no attribution needed --------------------
    pkgs = Counter()
    kinds = Counter()
    unmatched = Counter()
    for r in uv_rows:
        kind, pkg = classify_uv(r.msg or "")
        if kind is None:
            unmatched[clip((r.msg or "").split(" ")[0], 40)] += 1
            continue
        kinds[kind] += 1
        if kind in TRIED:
            pkgs[pkg] += 1
    out("")
    out("TOP PACKAGES BY CANDIDATES TRIED, WHOLE LOCK (a `Selecting candidate for X`")
    out("or `Searching for a compatible version of X` row; the matching `Found` and")
    out("`Returning` answers are NOT counted again, or every package would triple):")
    for pkg, n in pkgs.most_common(10):
        out("   %-46s %8d" % (pkg, n))
    out("")
    out("row kinds: %s" % ", ".join("%s=%d" % kv for kv in kinds.most_common()))
    out("total candidates tried: %d over %d distinct packages"
        % (sum(pkgs.values()), len(pkgs)))
    if unmatched:
        out("")
        out("UNMATCHED uv message shapes (NOT counted; extend CANDIDATE_PATTERNS):")
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
    # The biggest of those gaps is the question this section exists to answer:
    # while the child uv runs, what is written? Not "by this bundle" -- by
    # ANYONE. Fourteen backends share this log, so a window that is busy with
    # other bundles and empty of this one says the child is untraced, not idle.
    timed_t = [r for r in tagged if r.when() is not None]
    big = None
    for a, b in zip(timed_t, timed_t[1:]):
        d = secs(a.when(), b.when())
        if big is None or d > big[0]:
            big = (d, a, b)
    if big and big[0] >= threshold:
        d, a, b = big
        inside = [r for r in backend_rows
                  if r.when() is not None and a.when() < r.when() < b.when()]
        mine = [r for r in inside if bundle in (r.msg or "")]
        out("")
        out("   INSIDE THE LARGEST GAP (%.1fs, %s -> %s), across ALL bundles:"
            % (d, fmt(a.when()), fmt(b.when())))
        out("     backend rows of any kind: %d" % len(inside))
        out("     backend rows naming %s: %d" % (bundle, len(mine)))
        by_bundle = Counter()
        for r in inside:
            m = re.search(r"bundle=([A-Za-z0-9._-]+)", r.msg or "")
            if m:
                by_bundle[m.group(1)] += 1
        for name, n in by_bundle.most_common(8):
            out("       bundle=%-34s %8d" % (name, n))
        out("     Read that as: the backend and its SIBLINGS are logging normally")
        out("     throughout. What emits nothing is the one thing the gap is made")
        out("     of -- the child `uv` process this bundle spawned. Its stdout and")
        out("     stderr are piped and read only at exit, and no verbosity flag or")
        out("     RUST_LOG reaches it, so the whole call is a black box by")
        out("     construction, not by accident.")
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
        # Is a silence the FRONTEND thinking, or the whole system idle? The
        # backend runs in its own process with its own (UTC) clock, so the
        # answer is a row count in the same wall-clock interval, converted by
        # the offset measured at the top of this report.
        timed_w = [r for r in window if r.when() is not None]
        biggest = None
        for a, b in zip(timed_w, timed_w[1:]):
            d = secs(a.when(), b.when())
            if biggest is None or d > biggest[0]:
                biggest = (d, a, b)
        if biggest and backend_rows and biggest[0] >= threshold:
            d, a, b = biggest
            off = None
            for r in backend_rows:
                if r.utc is not None:
                    off = 0
                    break
            lo = a.when() + (b.when() - b.when())  # keep type
            lo = a.when()
            hi = b.when()
            # local -> UTC using the offset the header measured (whole hours).
            shift = None
            both = [r for r in lock_rows if r.local is not None and r.utc is not None]
            if both:
                shift = secs(both[0].local, both[0].utc)
            n_backend = None
            if shift is not None:
                import datetime as _dt
                lo_u = lo + _dt.timedelta(seconds=shift)
                hi_u = hi + _dt.timedelta(seconds=shift)
                n_backend = sum(1 for r in backend_rows
                                if r.utc is not None and lo_u <= r.utc <= hi_u)
            out("")
            out("   IS THAT SILENCE IDLE, OR JUST UNTRACED? backend rows in the same")
            if n_backend is None:
                out("   wall-clock interval: unknown -- no row carried both clocks, so the")
                out("   two logs cannot be aligned. Say nothing more than that.")
            else:
                out("   wall-clock interval (%s -> %s local, +%.0fs to UTC): %d"
                    % (fmt(lo), fmt(hi), shift, n_backend))
                if n_backend == 0:
                    out("   ZERO. Both processes are silent for the whole gap, at TRACE for")
                    out("   every target either filter names. So the work in it belongs to a")
                    out("   target NOT in the filter -- see the report's closing note.")
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

    out("")
    out("=" * 78)
    out("(d) WHAT THE NEXT INSTRUMENTED RUN MUST SET")
    out("=" * 78)
    out("Two lumps, two DIFFERENT uv processes, and only one of them is reachable")
    out("from the frontend at all.")
    out("")
    out("1. THE FRONTEND's in-process uv (pixi's own pypi resolve; the `pace`")
    out("   window above). Its resolver rows are here in bulk, but `uv_client` and")
    out("   `uv_distribution` produced ZERO rows in this run, and that is why the")
    out("   silence cannot be called fetching, resolving or idle. The cause is")
    out("   mechanical: pixi's `-v` flags BUILD their own filter -- the template in")
    out("   the binary is `apple_codesign=off,pixi=<l>,pixi_command_dispatcher=<l>,")
    out("   pixi_core=<l>,rattler_upload=<l>,uv_resolver=<l>,resolvo=<l>` -- which")
    out("   names neither, and RUST_LOG did not survive alongside it (uv_resolver")
    out("   rows arrived at TRACE though RUST_LOG asked for debug). So: DROP")
    out("   `-vvv` and drive the filter from RUST_LOG alone, adding")
    out("   `uv_client=debug,uv_distribution=debug` (registry_client's request rows")
    out("   are what separate FETCHING from THINKING, and `uv_distribution::source`")
    out("   is what an sdist build looks like). Verify the filter took effect by")
    out("   asserting a nonzero uv_client row count before trusting any timing.")
    out("")
    out("2. THE BACKEND's CHILD uv (`uv closure`, the 497 s call). No pixi flag and")
    out("   no frontend RUST_LOG can ever reach it: it is a separate executable")
    out("   spawned with stdout/stderr piped and collected only at exit, and the")
    out("   stderr shim deliberately unsets RUST_LOG before exec'ing the backend.")
    out("   Reaching inside it is a BACKEND CHANGE, not a harness setting: pass")
    out("   `-v` to the child and set `RUST_LOG` in its environment, and stream its")
    out("   stderr line by line into the backend's own tracing so each line gets a")
    out("   timestamp. Buffering it to the end would show WHAT it did and still not")
    out("   show WHEN, which is the entire question.")
    text = "\n".join(buf) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
