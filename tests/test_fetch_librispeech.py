#!/usr/bin/env python3
"""Tests for scripts/fetch_librispeech.py, with an OpenSLR stand-in on disk (file:// URLs)."""
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

import fetch_librispeech as fetch  # noqa: E402
from test_fetching import Folder  # noqa: E402


class FetchTests(Folder):
    def test_fetches_from_a_mirror_that_has_it_checks_and_extracts(self):
        archive = self.archive(f"mirror/{fetch.ARCHIVE}", [
            ("LibriSpeech", None),
            ("LibriSpeech/LICENSE.TXT", b"CC BY 4.0"),
            ("LibriSpeech/train-clean-100/19/198/19-198.trans.txt", b"19-198-0000 NORTHANGER ABBEY\n"),
            ("LibriSpeech/train-clean-100/19/198/19-198-0000.flac", b"fLaC"),
        ])
        with open(archive, "rb") as data:
            contents = data.read()
        mirrors = [Path(self.path("empty")).as_uri(), Path(self.path("mirror")).as_uri()]
        out = self.path("librispeech")
        options = dict(mirrors=mirrors, size=len(contents), md5=hashlib.md5(contents).hexdigest())
        with contextlib.redirect_stdout(io.StringIO()) as log, contextlib.redirect_stderr(io.StringIO()):
            fetch.fetch(out, True, **options)
        self.assertIn("fetching", log.getvalue())
        self.assertIn("extracted 3 files", log.getvalue())
        with open(os.path.join(out, "LibriSpeech", "train-clean-100", "19", "198", "19-198-0000.flac"), "rb") as data:
            self.assertEqual(data.read(), b"fLaC")

        with contextlib.redirect_stdout(io.StringIO()) as log:
            fetch.fetch(out, True, **options)
        self.assertIn(f"have {fetch.ARCHIVE}", log.getvalue())
        self.assertIn("extracted 0 files", log.getvalue())


if __name__ == "__main__":
    unittest.main()
