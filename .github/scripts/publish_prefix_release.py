#!/usr/bin/env python3
"""Publish and verify one exact conda package per supported Linux subdir."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any


REQUIRED_SUBDIRS = ("linux-64", "linux-aarch64")
DEFAULT_POLL_ATTEMPTS = 60


class PublishError(RuntimeError):
    """A release cannot be proven complete and byte-identical."""


@dataclasses.dataclass(frozen=True)
class Package:
    subdir: str
    path: pathlib.Path
    filename: str
    sha256: str
    size: int


@dataclasses.dataclass(frozen=True)
class RemoteState:
    kind: str
    detail: str


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_packages(root: pathlib.Path) -> dict[str, Package]:
    packages: dict[str, Package] = {}
    for subdir in REQUIRED_SUBDIRS:
        candidates = sorted((root / subdir).glob("*.conda"))
        if len(candidates) != 1 or not candidates[0].is_file():
            raise PublishError(
                f"expected exactly one {subdir} .conda package under {root}, "
                f"found {len(candidates)}"
            )
        path = candidates[0]
        packages[subdir] = Package(
            subdir=subdir,
            path=path,
            filename=path.name,
            sha256=file_sha256(path),
            size=path.stat().st_size,
        )
    return packages


class RepodataClient:
    def __init__(self, server_url: str, channel: str) -> None:
        self.server_url = server_url.rstrip("/")
        self.channel = channel

    def package_record(self, package: Package) -> Mapping[str, Any] | None:
        channel = urllib.parse.quote(self.channel, safe="")
        subdir = urllib.parse.quote(package.subdir, safe="")
        cache_buster = time.time_ns()
        url = (
            f"{self.server_url}/{channel}/{subdir}/repodata.json"
            f"?release-verify={cache_buster}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "pixi-build-retread-release-verifier/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                repodata = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise PublishError(
                f"failed to read {package.subdir} repodata: HTTP {error.code}"
            ) from error
        except (OSError, ValueError, urllib.error.URLError) as error:
            raise PublishError(
                f"failed to read {package.subdir} repodata: {error}"
            ) from error

        if not isinstance(repodata, dict):
            raise PublishError(f"{package.subdir} repodata is not an object")
        records = repodata.get("packages.conda")
        if not isinstance(records, dict):
            raise PublishError(
                f"{package.subdir} repodata has no packages.conda mapping"
            )
        record = records.get(package.filename)
        if record is None:
            return None
        if not isinstance(record, dict):
            raise PublishError(
                f"{package.subdir} record for {package.filename} is not an object"
            )
        return record


def remote_state(
    package: Package,
    record: Mapping[str, Any] | None,
) -> RemoteState:
    if record is None:
        return RemoteState("missing", "not present in repodata")

    expected = {
        "sha256": package.sha256,
        "size": package.size,
        "subdir": package.subdir,
    }
    mismatches = [
        f"{key}: remote={record.get(key)!r} local={value!r}"
        for key, value in expected.items()
        if record.get(key) != value
    ]
    if mismatches:
        return RemoteState("mismatch", "; ".join(mismatches))
    return RemoteState(
        "matching",
        f"sha256={package.sha256} size={package.size}",
    )


def default_upload(
    channel: str,
    package: Package,
    *,
    api_key: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PREFIX_API_KEY"] = api_key
    try:
        return subprocess.run(
            [
                "rattler-build",
                "upload",
                "prefix",
                "--log-style",
                "plain",
                "--channel",
                channel,
                str(package.path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    except OSError as error:
        raise PublishError(f"could not execute rattler-build: {error}") from error


class ReleasePublisher:
    def __init__(
        self,
        packages: Mapping[str, Package],
        client: RepodataClient,
        channel: str,
        *,
        upload: Callable[[str, Package], subprocess.CompletedProcess[str]],
        sleep: Callable[[float], None] = time.sleep,
        poll_attempts: int = DEFAULT_POLL_ATTEMPTS,
        poll_interval: float = 5.0,
    ) -> None:
        if set(packages) != set(REQUIRED_SUBDIRS):
            raise PublishError(
                f"release requires exactly these subdirs: {', '.join(REQUIRED_SUBDIRS)}"
            )
        if poll_attempts < 1:
            raise PublishError("poll attempts must be at least one")
        if poll_interval < 0:
            raise PublishError("poll interval cannot be negative")
        self.packages = packages
        self.client = client
        self.channel = channel
        self.upload = upload
        self.sleep = sleep
        self.poll_attempts = poll_attempts
        self.poll_interval = poll_interval

    def inspect(self, package: Package) -> RemoteState:
        return remote_state(package, self.client.package_record(package))

    def inspect_all(self) -> dict[str, RemoteState]:
        return {
            subdir: self.inspect(self.packages[subdir])
            for subdir in REQUIRED_SUBDIRS
        }

    @staticmethod
    def format_states(states: Mapping[str, RemoteState]) -> str:
        return ", ".join(
            f"{subdir}={states[subdir].kind} ({states[subdir].detail})"
            for subdir in REQUIRED_SUBDIRS
        )

    def wait_for_match(self, package: Package) -> None:
        state = RemoteState("missing", "not checked")
        for attempt in range(1, self.poll_attempts + 1):
            state = self.inspect(package)
            if state.kind == "matching":
                print(f"verified {package.subdir}: {state.detail}")
                return
            if state.kind == "mismatch":
                raise PublishError(
                    f"remote {package.subdir}/{package.filename} differs: "
                    f"{state.detail}"
                )
            if attempt < self.poll_attempts:
                self.sleep(self.poll_interval)
        raise PublishError(
            f"timed out waiting for {package.subdir}/{package.filename} in repodata "
            f"after {self.poll_attempts} checks: {state.detail}"
        )

    def publish(self) -> None:
        initial = self.inspect_all()
        mismatched = [
            subdir
            for subdir, state in initial.items()
            if state.kind == "mismatch"
        ]
        if mismatched:
            raise PublishError(
                "refusing to upload over different remote bytes: "
                + self.format_states(initial)
            )

        missing = [
            subdir
            for subdir in REQUIRED_SUBDIRS
            if initial[subdir].kind == "missing"
        ]
        if missing and len(missing) != len(REQUIRED_SUBDIRS):
            print(
                "::warning::partial multiarch release detected before upload: "
                + self.format_states(initial),
                file=sys.stderr,
            )
        if not missing:
            print("both platform packages already match exactly; nothing to upload")
            return

        for subdir in missing:
            package = self.packages[subdir]
            result = self.upload(self.channel, package)
            if result.stdout:
                print(result.stdout.rstrip())
            if result.returncode != 0:
                output = result.stdout or ""
                state = self.inspect(package)
                if state.kind == "matching":
                    print(
                        f"upload exited {result.returncode}, but exact remote bytes "
                        f"are present for {subdir}"
                    )
                    continue
                if state.kind == "mismatch":
                    raise PublishError(
                        f"upload failed and remote {subdir}/{package.filename} "
                        f"differs: {state.detail}"
                    )
                retryable_race = re.search(
                    r"(?:\b409\b|already exists|conflict|timed? out|"
                    r"connection (?:closed|reset|aborted)|broken pipe|"
                    r"unexpected eof|end of file)",
                    output,
                    re.I,
                )
                if retryable_race is None:
                    raise PublishError(
                        f"upload failed for {subdir} with exit {result.returncode}; "
                        f"remote state is {state.kind} ({state.detail}); "
                        f"output={output.strip()!r}"
                    )
                print(
                    f"upload reported a possible race for {subdir}; "
                    "verifying remote bytes"
                )
            self.wait_for_match(package)

        final = self.inspect_all()
        if any(state.kind != "matching" for state in final.values()):
            raise PublishError(
                "publication did not converge to a complete matching set: "
                + self.format_states(final)
            )
        print("release complete: " + self.format_states(final))


def report_failure_state(publisher: ReleasePublisher) -> None:
    try:
        states = publisher.inspect_all()
    except PublishError as error:
        print(f"::error::could not audit final remote state: {error}", file=sys.stderr)
        return
    print(
        "::error::remote release state: " + publisher.format_states(states),
        file=sys.stderr,
    )
    if any(state.kind == "mismatch" for state in states.values()):
        print(
            "::error::a remote filename has different bytes; never overwrite it. "
            "Publish a new build number or version instead.",
            file=sys.stderr,
        )
        return
    matching = sum(state.kind == "matching" for state in states.values())
    if matching == 1:
        print(
            "::error::partial multiarch publication detected; rerun this exact tag "
            "after resolving the reported failure. Matching artifacts are safe "
            "to reuse.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", required=True)
    parser.add_argument("--packages-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--server-url",
        default=os.environ.get("PREFIX_SERVER_URL", "https://prefix.dev"),
    )
    parser.add_argument(
        "--poll-attempts", type=int, default=DEFAULT_POLL_ATTEMPTS
    )
    parser.add_argument("--poll-interval", type=float, default=5.0)
    args = parser.parse_args()

    api_key = os.environ.pop("PREFIX_API_KEY", "")
    if not api_key:
        print(
            "::error::PREFIX_API_KEY is required for a tag release",
            file=sys.stderr,
        )
        return 1

    try:
        packages = discover_packages(args.packages_root)
        client = RepodataClient(args.server_url, args.channel)
        upload = lambda channel, package: default_upload(
            channel,
            package,
            api_key=api_key,
        )
        publisher = ReleasePublisher(
            packages,
            client,
            args.channel,
            upload=upload,
            poll_attempts=args.poll_attempts,
            poll_interval=args.poll_interval,
        )
        publisher.publish()
    except PublishError as error:
        print(f"::error::release publication failed: {error}", file=sys.stderr)
        if "publisher" in locals():
            report_failure_state(publisher)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
