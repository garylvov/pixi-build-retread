#!/usr/bin/env python3
"""Fail when the release version sources disagree."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE_NAME = "pixi-build-retread"


def quoted_value(block: str, key: str) -> str:
    match = re.search(rf'(?m)^{re.escape(key)}\s*=\s*"([^"]+)"\s*$', block)
    if match is None:
        raise ValueError(f"missing literal {key!r}")
    return match.group(1)


def cargo_manifest_version() -> str:
    manifest = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    package = re.search(r"(?ms)^\[package\]\s*\n(.*?)(?=^\[|\Z)", manifest)
    if package is None:
        raise ValueError("Cargo.toml has no [package] table")
    return quoted_value(package.group(1), "version")


def cargo_lock_version() -> str:
    lockfile = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    packages = re.split(r"(?m)^\[\[package\]\]\s*$", lockfile)[1:]
    matches = []
    for package in packages:
        block = re.split(r"(?m)^\[", package, maxsplit=1)[0]
        try:
            name = quoted_value(block, "name")
        except ValueError:
            continue
        if name == PACKAGE_NAME:
            matches.append(quoted_value(block, "version"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {PACKAGE_NAME!r} package in Cargo.lock, "
            f"found {len(matches)}"
        )
    return matches[0]


def recipe_version() -> str:
    recipe = (ROOT / "recipe" / "recipe.yaml").read_text(encoding="utf-8")
    context = re.search(r"(?m)^context:\s*\n((?:^[ \t]+.*\n?)*)", recipe)
    if context is None:
        raise ValueError("recipe/recipe.yaml has no top-level context block")
    version = re.search(r"(?m)^\s+version:\s*['\"]?([^'\"#\s]+)", context.group(1))
    if version is None:
        raise ValueError("recipe/recipe.yaml context has no literal version")
    return version.group(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        help="release tag to validate (must be v<version>); omit for workflow_dispatch",
    )
    args = parser.parse_args()

    versions = {
        "Cargo.toml": cargo_manifest_version(),
        "Cargo.lock": cargo_lock_version(),
        "recipe/recipe.yaml": recipe_version(),
    }
    expected = versions["Cargo.toml"]
    mismatches = {
        source: version for source, version in versions.items() if version != expected
    }

    if args.tag:
        if not re.fullmatch(r"v[0-9][0-9A-Za-z.+-]*", args.tag):
            raise ValueError(f"release tag must have the form v<version>, got {args.tag!r}")
        tag_version = args.tag[1:]
        if tag_version != expected:
            mismatches[f"tag {args.tag}"] = tag_version

    if mismatches:
        details = ", ".join(f"{source}={version}" for source, version in versions.items())
        if args.tag:
            details += f", tag={args.tag[1:]}"
        print(f"release version mismatch: {details}", file=sys.stderr)
        return 1

    suffix = f" and tag {args.tag}" if args.tag else ""
    print(f"release versions agree at {expected}{suffix}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"release version validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
