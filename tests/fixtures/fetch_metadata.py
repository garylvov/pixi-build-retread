#!/usr/bin/env python3
"""Capture wheel METADATA fixtures used by the retread integration tests.

Downloads each wheel listed in WHEELS, extracts `*.dist-info/METADATA`, and
writes it to `<output>.METADATA.txt` plus a sibling `<output>.json` with the
URL, sha256, and the filename. Re-run this whenever upstream republishes a
wheel or the test fixtures need refreshing -- the diff in the generated files
becomes the audit trail.

Uses only the Python standard library so it runs anywhere a Python 3.9+ is
available (pixi's default env satisfies this).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Wheel:
    """A wheel to capture, plus where to write its fixture files."""

    url: str
    slug: str  # output filename stem under tests/fixtures/


# Add more wheels here as new test scenarios appear. Pick wheels that exercise
# the conflicts described in https://github.com/prefix-dev/pixi/issues/5230 --
# isaacsim is the canonical case because its `Requires-Dist` pins (numpy,
# torch, pillow, scipy) overlap with conda-side packages other workspace
# features pull in.
WHEELS: list[Wheel] = [
    Wheel(
        url=(
            "https://pypi.nvidia.com/isaacsim-kernel/"
            "isaacsim_kernel-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
        ),
        slug="isaacsim_kernel",
    ),
    Wheel(
        url=(
            "https://pypi.nvidia.com/isaacsim-core/"
            "isaacsim_core-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
        ),
        slug="isaacsim_core",
    ),
]


def fetch(url: str) -> bytes:
    print(f"  GET {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def extract_metadata(wheel_bytes: bytes) -> str:
    # The wheel's own METADATA lives at `<name>-<version>.dist-info/METADATA`
    # at the zip root. Other vendored packages may ship their own .dist-info
    # nested under data/ or under package subtrees; ignore those.
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as zf:
        candidates = [
            n
            for n in zf.namelist()
            if n.endswith(".dist-info/METADATA")
            and n.count("/") == 1  # root-level only
        ]
        if not candidates:
            raise RuntimeError("no root-level .dist-info/METADATA inside wheel")
        if len(candidates) > 1:
            raise RuntimeError(f"ambiguous root-level METADATA entries: {candidates}")
        with zf.open(candidates[0]) as f:
            return f.read().decode("utf-8")


def main() -> int:
    here = Path(__file__).resolve().parent
    for wheel in WHEELS:
        print(f"==> {wheel.slug}", file=sys.stderr)
        try:
            data = fetch(wheel.url)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1

        sha256 = hashlib.sha256(data).hexdigest()
        metadata = extract_metadata(data)
        filename = wheel.url.rsplit("/", 1)[-1]

        metadata_path = here / f"{wheel.slug}.METADATA.txt"
        manifest_path = here / f"{wheel.slug}.json"
        metadata_path.write_text(metadata, encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {"url": wheel.url, "filename": filename, "sha256": sha256},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {metadata_path.name} ({len(metadata)} bytes)", file=sys.stderr)
        print(f"  wrote {manifest_path.name} (sha256={sha256[:16]}...)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
