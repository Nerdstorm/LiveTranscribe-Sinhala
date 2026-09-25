#!/usr/bin/env python3
"""Tests for scripts/fetching.py, with downloads from files on disk (file:// URLs) and from a
local HTTP server that honours ranges."""
import contextlib
import hashlib
import http.server
import io
import os
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fetching  # noqa: E402


def git_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Folder(unittest.TestCase):
    """A temporary folder, with helpers to write files and archives into it."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def write(self, relative, data):
        path = self.path(relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as out:
            out.write(data)
        return path

    def archive(self, relative, members):
        """A .tar.gz of (name, bytes) files; bytes None makes a folder, a str a symbolic link."""
        path = self.path(relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with tarfile.open(path, "w:gz") as tar:
            for name, data in members:
                info = tarfile.TarInfo(name)
                if data is None:
                    info.type = tarfile.DIRTYPE
                    tar.addfile(info)
                elif isinstance(data, str):
                    info.type, info.linkname = tarfile.SYMTYPE, data
                    tar.addfile(info)
                else:
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        return path


class VerifyTests(Folder):
    def test_hashes_match_git_sha256_and_md5(self):
        path = self.write("hello.txt", b"hello\n")
        # `echo hello | git hash-object --stdin`
        self.assertEqual(fetching.digest(path, "git"), "ce013625030ba8dba906f756967f9e9ca394464a")
        self.assertEqual(fetching.digest(path, "sha256"), hashlib.sha256(b"hello\n").hexdigest())
        self.assertEqual(fetching.digest(path, "md5"), hashlib.md5(b"hello\n").hexdigest())
        with self.assertRaisesRegex(ValueError, "unknown hash"):
            fetching.digest(path, "crc32")

    def test_a_file_passes_only_with_its_size_and_hash(self):
        path = self.write("hello.txt", b"hello\n")
        self.assertTrue(fetching.verified(path, 6, "git", git_id(b"hello\n")))
        self.assertFalse(fetching.verified(path, 7, "git", git_id(b"hello\n")))
        self.assertFalse(fetching.verified(path, 6, "git", git_id(b"jello\n")))
        self.assertFalse(fetching.verified(self.path("missing.txt"), 6, "git", git_id(b"hello\n")))


class DownloadTests(Folder):
    md5 = hashlib.md5(b"hello\n").hexdigest()

    def test_keeps_a_file_that_matches(self):
        url = Path(self.write("mirror/a.tsv", b"hello\n")).as_uri()
        target = self.path("a.tsv")
        fetching.download([url], target, 6, "md5", self.md5)
        with open(target, "rb") as data:
            self.assertEqual(data.read(), b"hello\n")
        self.assertFalse(os.path.exists(target + ".part"))

    def test_tries_the_next_mirror(self):
        good = Path(self.write("mirror/a.tsv", b"hello\n")).as_uri()
        with contextlib.redirect_stderr(io.StringIO()) as log:
            fetching.download([Path(self.path("gone", "a.tsv")).as_uri(), good], self.path("a.tsv"), 6, "md5", self.md5)
        self.assertIn("attempt 1 of 3", log.getvalue())
        self.assertTrue(os.path.exists(self.path("a.tsv")))

    def test_refuses_a_file_that_doesnt_match(self):
        url = Path(self.write("mirror/a.tsv", b"hello\n")).as_uri()
        target = self.path("a.tsv")
        with contextlib.redirect_stderr(io.StringIO()) as log, self.assertRaisesRegex(RuntimeError, "doesn't match"):
            fetching.download([url], target, 6, "sha256", "0" * 64, attempts=2)
        self.assertIn("attempt 2 of 2", log.getvalue())
        self.assertFalse(os.path.exists(target) or os.path.exists(target + ".part"))

    def test_an_unreachable_url_fails(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(RuntimeError):
            fetching.download([Path(self.path("nowhere.tsv")).as_uri()], self.path("a.tsv"), 6, "git", "abc", attempts=1)

    def test_a_server_that_ignores_the_range_starts_the_file_over(self):
        # file:// URLs answer a range with the whole file, as a server without range support does.
        url = Path(self.write("mirror/a.tsv", b"hello\n")).as_uri()
        self.write("a.tsv.part", b"hel")
        fetching.download([url], self.path("a.tsv"), 6, "md5", self.md5)
        with open(self.path("a.tsv"), "rb") as data:
            self.assertEqual(data.read(), b"hello\n")

    def test_keeps_what_it_fetched_when_every_attempt_fails(self):
        self.write("a.tsv.part", b"hel")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, "carry on from the 3 bytes"):
            fetching.download([Path(self.path("nowhere.tsv")).as_uri()], self.path("a.tsv"), 6, "md5", self.md5, attempts=1)
        self.assertTrue(os.path.exists(self.path("a.tsv.part")))


class RangeServer(http.server.BaseHTTPRequestHandler):
    """Serves `data` at any path, honouring a Range header's start, and notes each Range asked for."""
    data = b""
    ranges = []

    def do_GET(self):
        asked = self.headers.get("Range")
        RangeServer.ranges.append(asked)
        start = int(re.fullmatch(r"bytes=(\d+)-", asked).group(1)) if asked else 0
        body = self.data[start:]
        self.send_response(206 if asked else 200)
        if asked:
            self.send_header("Content-Range", f"bytes {start}-{len(self.data) - 1}/{len(self.data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ResumeTests(Folder):
    md5 = hashlib.md5(b"hello\n").hexdigest()

    def setUp(self):
        super().setUp()
        RangeServer.data, RangeServer.ranges = b"hello\n", []
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeServer)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_address[1]}/a.tsv"

    def test_carries_on_from_a_broken_off_download(self):
        self.write("a.tsv.part", b"hel")
        fetching.download([self.url], self.path("a.tsv"), 6, "md5", self.md5)
        self.assertEqual(RangeServer.ranges, ["bytes=3-"])
        with open(self.path("a.tsv"), "rb") as data:
            self.assertEqual(data.read(), b"hello\n")

    def test_starts_over_when_the_carried_on_file_doesnt_match(self):
        self.write("a.tsv.part", b"jel")
        with contextlib.redirect_stderr(io.StringIO()) as log:
            fetching.download([self.url], self.path("a.tsv"), 6, "md5", self.md5)
        self.assertIn("doesn't match", log.getvalue())
        self.assertEqual(RangeServer.ranges, ["bytes=3-", None])
        with open(self.path("a.tsv"), "rb") as data:
            self.assertEqual(data.read(), b"hello\n")


class ExtractTests(Folder):
    def test_writes_the_files_under_their_folder(self):
        archive = self.archive("a.tar.gz", [("train", None), ("train/a.wav", b"aa"), ("train/x/y/b.wav", b"bbb")])
        self.assertEqual(fetching.extract(archive, self.path("audio"), "train"), 2)
        with open(self.path("audio", "train", "x", "y", "b.wav"), "rb") as data:
            self.assertEqual(data.read(), b"bbb")

    def test_skips_files_already_there_and_rewrites_cut_ones(self):
        archive = self.archive("a.tar.gz", [("train/a.wav", b"aa"), ("train/b.wav", b"bbb")])
        self.assertEqual(fetching.extract(archive, self.path("audio"), "train"), 2)
        self.assertEqual(fetching.extract(archive, self.path("audio"), "train"), 0)
        self.write(os.path.join("audio", "train", "b.wav"), b"b")
        self.assertEqual(fetching.extract(archive, self.path("audio"), "train"), 1)

    def test_refuses_entries_outside_the_folder_and_links(self):
        for name, data in (("../evil.wav", b"x"), ("train/../../evil.wav", b"x"), ("/evil.wav", b"x"),
                           ("test/a.wav", b"x"), ("train/link.wav", "/etc/passwd")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unexpected entry"):
                fetching.extract(self.archive("bad.tar.gz", [(name, data)]), self.path("audio"), "train")
        self.assertFalse(os.path.exists(self.path("evil.wav")))


if __name__ == "__main__":
    unittest.main()
