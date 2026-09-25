#!/usr/bin/env python3
"""Tests for scripts/fetch_fleurs.py, with a Hugging Face stand-in on disk (file:// URLs)."""
import contextlib
import hashlib
import io
import os
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

import fetch_fleurs as fetch  # noqa: E402
from test_fetching import Folder, git_id  # noqa: E402


class WantedTests(unittest.TestCase):
    entries = [
        {"type": "file", "path": "data/en_us/train.tsv", "size": 10, "oid": "abc"},
        {"type": "file", "path": "data/en_us/dev.tsv", "size": 5, "oid": "def"},
        {"type": "file", "path": "data/en_us/audio/train.tar.gz", "size": 99, "oid": "pointer",
         "lfs": {"oid": "sha", "size": 99, "pointerSize": 134}},
    ]

    def test_takes_each_splits_transcript_and_archive(self):
        self.assertEqual(fetch.wanted(self.entries, "en_us", ["train"]), [
            ("data/en_us/audio/train.tar.gz", 99, "sha256", "sha"),
            ("data/en_us/train.tsv", 10, "git", "abc"),
        ])

    def test_a_split_the_hub_lacks_stops_the_run(self):
        with self.assertRaisesRegex(ValueError, "FLEURS has no data/en_us/audio/test.tar.gz, data/en_us/test.tsv"):
            fetch.wanted(self.entries, "en_us", ["test"])


class FetchTests(Folder):
    def test_fetches_checks_and_extracts_then_keeps_what_it_has(self):
        transcript = b"1\ta.wav\tHello.\thello\th e l l o |\t16000\tMALE\n"
        self.write("hub/data/en_us/train.tsv", transcript)
        archive = self.archive("hub/data/en_us/audio/train.tar.gz", [("train/a.wav", b"RIFF")])
        with open(archive, "rb") as data:
            archive_bytes = data.read()
        entries = [
            {"type": "file", "path": "data/en_us/train.tsv", "size": len(transcript), "oid": git_id(transcript)},
            {"type": "file", "path": "data/en_us/audio/train.tar.gz", "size": len(archive_bytes), "oid": "pointer",
             "lfs": {"oid": hashlib.sha256(archive_bytes).hexdigest(), "size": len(archive_bytes)}},
        ]
        out = self.path("fleurs")
        hub = Path(self.path("hub")).as_uri()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            fetch.fetch(out, ["en_us"], ["train"], True, list_files=lambda code: entries, files_url=hub)
        self.assertIn("fetching data/en_us/train.tsv", log.getvalue())
        with open(os.path.join(out, "data", "en_us", "audio", "train", "a.wav"), "rb") as data:
            self.assertEqual(data.read(), b"RIFF")

        with contextlib.redirect_stdout(io.StringIO()) as log:
            fetch.fetch(out, ["en_us"], ["train"], True, list_files=lambda code: entries, files_url=hub)
        self.assertEqual(log.getvalue().count("have "), 2)
        self.assertIn("extracted 0 recordings", log.getvalue())


if __name__ == "__main__":
    unittest.main()
