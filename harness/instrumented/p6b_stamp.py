#!/usr/bin/env python3
"""Stamp every stdin line with a local ISO timestamp, line-buffered.

Why this exists: `pixi lock`'s own tracing rows carry NO time of their own --
only the backend's rows do, because those come from the backend's tracing
through the stderr shim. Every timeline this campaign has built came off
backend rows for exactly that reason, and the frontend's uv resolution (the
~1000 s `pace` tail) has no backend rows in it at all. Stamping at the pipe is
what makes that window measurable.

Kept deliberately dumb: no parsing, no filtering, no buffering beyond a line.
A multi-GB stream goes through it, so it must not accumulate anything.

Python 3.9 on this box: no tomllib, no fancy typing.
"""
import sys
import time


def main():
    out = sys.stdout
    # Read as bytes and write as bytes: log lines carry ANSI and occasionally
    # invalid UTF-8, and a decode error mid-relock would lose the rest of the
    # stream. errors='replace' on a text wrapper would corrupt bytes silently.
    stdin = sys.stdin.buffer
    stdout = out.buffer
    for line in stdin:
        t = time.time()
        stamp = "%s.%06d " % (
            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)),
            int((t % 1) * 1e6),
        )
        stdout.write(stamp.encode("ascii"))
        stdout.write(line)
        stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass
