#!/usr/bin/env python3
"""Tests for scripts/pick_librispeech.py."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pick_librispeech as pick  # noqa: E402


def make_corpus(root, chapters):
    """A LibriSpeech train-clean-100 tree with {"<speaker>-<chapter>": utterance count}."""
    for chapter_id, count in chapters.items():
        speaker, chapter = chapter_id.split("-")
        folder = os.path.join(root, "LibriSpeech", "train-clean-100", speaker, chapter)
        os.makedirs(folder)
        with open(os.path.join(folder, f"{chapter_id}.trans.txt"), "w", encoding="utf-8") as transcripts:
            for number in range(count):
                utterance = f"{chapter_id}-{number:04d}"
                transcripts.write(f"{utterance} WORDS OF {utterance}\n")
                open(os.path.join(folder, f"{utterance}.flac"), "wb").close()


class PickTests(unittest.TestCase):
    files = [f"19-198-{number:04d}.flac" for number in range(10)]

    def test_the_same_seed_picks_the_same_in_name_order(self):
        first = pick.pick(self.files, 4, 52)
        self.assertEqual(first, pick.pick(list(reversed(self.files)), 4, 52))
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(set(first)), 4)
        self.assertNotEqual(first, pick.pick(self.files, 4, 53))

    def test_the_count_must_fit(self):
        for count in (0, 11):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "can't pick"):
                pick.pick(self.files, count, 52)


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.corpus = os.path.join(self.root, "librispeech")
        make_corpus(self.corpus, {"19-198": 3, "26-495": 2})
        self.out = os.path.join(self.root, "picked")

    def test_deals_the_picks_into_shards_of_links(self):
        files = ["19-198-0000.flac", "19-198-0002.flac", "26-495-0001.flac"]
        folders = pick.link(files, self.corpus, self.out, 2)
        self.assertEqual(sorted(os.listdir(folders[0])), ["19-198-0000.flac", "26-495-0001.flac"])
        self.assertEqual(os.listdir(folders[1]), ["19-198-0002.flac"])
        link = os.path.join(folders[0], "26-495-0001.flac")
        self.assertEqual(os.readlink(link), os.path.join(self.corpus, "LibriSpeech", "train-clean-100", "26", "495", "26-495-0001.flac"))
        self.assertEqual(pick.link(files, self.corpus, self.out, 2), folders)

    def test_refuses_a_folder_with_other_recordings(self):
        pick.link(["19-198-0000.flac"], self.corpus, self.out, 1)
        with self.assertRaisesRegex(ValueError, "weren't picked, such as 19-198-0000.flac"):
            pick.link(["19-198-0001.flac"], self.corpus, self.out, 1)

    def test_refuses_a_recording_the_corpus_lacks(self):
        with self.assertRaisesRegex(ValueError, "is missing"):
            pick.link(["19-198-0009.flac"], self.corpus, self.out, 1)

    def test_end_to_end(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "pick_librispeech.py"), "--librispeech", self.corpus, "--out", self.out,
             "--count", "4", "--seed", "52"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("picked 4 of 5 utterances, from 2 speakers", result.stdout)
        self.assertEqual(len(os.listdir(self.out)), 4)


if __name__ == "__main__":
    unittest.main()
