#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Mapping
from typing import Any


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import publish_prefix_release as release


def matching_record(package: release.Package) -> dict[str, object]:
    return {
        "sha256": package.sha256,
        "size": package.size,
        "subdir": package.subdir,
    }


class FakeClient:
    def __init__(
        self,
        records: Mapping[str, list[Mapping[str, Any] | None]],
    ) -> None:
        self.records = {subdir: list(states) for subdir, states in records.items()}

    def package_record(
        self,
        package: release.Package,
    ) -> Mapping[str, Any] | None:
        states = self.records[package.subdir]
        if len(states) > 1:
            return states.pop(0)
        return states[0]


class PublishReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tempdir.name)
        for subdir, content in (
            ("linux-64", b"x86 package"),
            ("linux-aarch64", b"arm package"),
        ):
            directory = root / subdir
            directory.mkdir()
            (directory / f"retread-{subdir}.conda").write_bytes(content)
        self.packages = release.discover_packages(root)
        self.uploaded: list[str] = []
        self.sleeps: list[float] = []

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def upload_ok(
        self,
        channel: str,
        package: release.Package,
    ) -> subprocess.CompletedProcess[str]:
        self.assertEqual(channel, "test-channel")
        self.uploaded.append(package.subdir)
        return subprocess.CompletedProcess([], 0, "uploaded")

    def publisher(
        self,
        client: FakeClient,
        *,
        upload=None,
        attempts: int = 3,
    ) -> release.ReleasePublisher:
        return release.ReleasePublisher(
            self.packages,
            client,  # type: ignore[arg-type]
            "test-channel",
            upload=upload or self.upload_ok,
            sleep=self.sleeps.append,
            poll_attempts=attempts,
            poll_interval=0.25,
        )

    def test_matching_remote_set_is_idempotent(self) -> None:
        client = FakeClient(
            {
                subdir: [matching_record(package)]
                for subdir, package in self.packages.items()
            }
        )
        self.publisher(client).publish()
        self.assertEqual(self.uploaded, [])

    def test_mismatch_fails_before_any_upload(self) -> None:
        records = {
            subdir: [matching_record(package)]
            for subdir, package in self.packages.items()
        }
        records["linux-64"][0] = {
            **records["linux-64"][0],
            "sha256": "0" * 64,
        }
        with self.assertRaisesRegex(release.PublishError, "different remote bytes"):
            self.publisher(FakeClient(records)).publish()
        self.assertEqual(self.uploaded, [])

    def test_missing_packages_upload_and_reach_complete_state(self) -> None:
        client = FakeClient(
            {
                subdir: [None, matching_record(package)]
                for subdir, package in self.packages.items()
            }
        )
        self.publisher(client).publish()
        self.assertEqual(self.uploaded, ["linux-64", "linux-aarch64"])

    def test_partial_state_uploads_only_missing_platform(self) -> None:
        client = FakeClient(
            {
                "linux-64": [matching_record(self.packages["linux-64"])],
                "linux-aarch64": [
                    None,
                    matching_record(self.packages["linux-aarch64"]),
                ],
            }
        )
        self.publisher(client).publish()
        self.assertEqual(self.uploaded, ["linux-aarch64"])

    def test_conflict_is_accepted_only_after_digest_match(self) -> None:
        client = FakeClient(
            {
                "linux-64": [matching_record(self.packages["linux-64"])],
                "linux-aarch64": [
                    None,
                    None,
                    None,
                    matching_record(self.packages["linux-aarch64"]),
                ],
            }
        )

        def conflict(
            _channel: str,
            package: release.Package,
        ) -> subprocess.CompletedProcess[str]:
            self.uploaded.append(package.subdir)
            return subprocess.CompletedProcess([], 1, "HTTP 409 Conflict")

        self.publisher(client, upload=conflict).publish()
        self.assertEqual(self.uploaded, ["linux-aarch64"])
        self.assertEqual(self.sleeps, [0.25])

    def test_failed_upload_accepts_immediate_exact_remote_match(self) -> None:
        client = FakeClient(
            {
                "linux-64": [matching_record(self.packages["linux-64"])],
                "linux-aarch64": [
                    None,
                    matching_record(self.packages["linux-aarch64"]),
                ],
            }
        )

        def lost_response(
            _channel: str,
            package: release.Package,
        ) -> subprocess.CompletedProcess[str]:
            self.uploaded.append(package.subdir)
            return subprocess.CompletedProcess([], 1, "connection closed")

        self.publisher(client, upload=lost_response).publish()
        self.assertEqual(self.uploaded, ["linux-aarch64"])
        self.assertEqual(self.sleeps, [])

    def test_transport_race_polls_until_exact_remote_match(self) -> None:
        client = FakeClient(
            {
                "linux-64": [matching_record(self.packages["linux-64"])],
                "linux-aarch64": [
                    None,
                    None,
                    None,
                    matching_record(self.packages["linux-aarch64"]),
                ],
            }
        )

        def lost_response(
            _channel: str,
            package: release.Package,
        ) -> subprocess.CompletedProcess[str]:
            self.uploaded.append(package.subdir)
            return subprocess.CompletedProcess([], 1, "connection closed")

        self.publisher(client, upload=lost_response).publish()
        self.assertEqual(self.uploaded, ["linux-aarch64"])
        self.assertEqual(self.sleeps, [0.25])

    def test_non_conflict_failure_stops_and_reports_partial_state(self) -> None:
        client = FakeClient(
            {
                "linux-64": [None, matching_record(self.packages["linux-64"])],
                "linux-aarch64": [None],
            }
        )

        def fail_second(
            _channel: str,
            package: release.Package,
        ) -> subprocess.CompletedProcess[str]:
            self.uploaded.append(package.subdir)
            if package.subdir == "linux-aarch64":
                return subprocess.CompletedProcess([], 1, "authentication failed")
            return subprocess.CompletedProcess([], 0, "uploaded")

        publisher = self.publisher(client, upload=fail_second)
        with self.assertRaisesRegex(release.PublishError, "upload failed"):
            publisher.publish()
        states = publisher.inspect_all()
        self.assertEqual(states["linux-64"].kind, "matching")
        self.assertEqual(states["linux-aarch64"].kind, "missing")

    def test_polling_is_bounded(self) -> None:
        client = FakeClient(
            {
                "linux-64": [matching_record(self.packages["linux-64"])],
                "linux-aarch64": [None],
            }
        )
        with self.assertRaisesRegex(release.PublishError, "timed out"):
            self.publisher(client, attempts=3).publish()
        self.assertEqual(self.sleeps, [0.25, 0.25])


if __name__ == "__main__":
    unittest.main()
