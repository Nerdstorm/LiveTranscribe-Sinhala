#!/usr/bin/env python3
"""Tests for scripts/make_split.py."""
import collections
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import make_split  # noqa: E402


class ReadSpeakersTests(unittest.TestCase):
    def test_counts_utterances_per_speaker_and_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "utt_spk_text.tsv")
            with open(path, "w", encoding="utf-8") as tsv:
                tsv.write("aa01\tspk1\tඅපි\n\naa02\tspk1\tයමු\naa03\tspk2\tගෙදර\n")
            self.assertEqual(make_split.read_speakers(path), {"spk1": 2, "spk2": 1})


class SplitSpeakersTests(unittest.TestCase):
    speakers = {f"spk{number:02d}": 1 for number in range(20)}

    def test_holds_out_the_asked_numbers_of_speakers(self):
        split = make_split.split_speakers(self.speakers, 52, test=4, dev=2)
        self.assertEqual(set(split), set(self.speakers))
        self.assertEqual(collections.Counter(split.values()), {"train": 14, "test": 4, "dev": 2})

    def test_the_same_seed_gives_the_same_split_whatever_the_input_order(self):
        backwards = dict(reversed(list(self.speakers.items())))
        self.assertEqual(make_split.split_speakers(self.speakers, 52, 4, 2),
                         make_split.split_speakers(backwards, 52, 4, 2))

    def test_the_seed_decides_the_split(self):
        self.assertNotEqual(make_split.split_speakers(self.speakers, 52, 4, 2),
                            make_split.split_speakers(self.speakers, 53, 4, 2))

    def test_refuses_to_leave_no_speakers_for_training(self):
        with self.assertRaises(ValueError):
            make_split.split_speakers(self.speakers, 52, 15, 5)


class EndToEndTests(unittest.TestCase):
    def test_writes_the_split_sorted_by_speaker_and_prints_counts(self):
        with tempfile.TemporaryDirectory() as root:
            transcripts = os.path.join(root, "utt_spk_text.tsv")
            with open(transcripts, "w", encoding="utf-8") as tsv:
                for number in range(6):
                    tsv.write(f"aa{number:02d}\tspk{5 - number}\tඅපි\n")
            out = os.path.join(root, "speaker-split.tsv")
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "make_split.py"), transcripts, out, "--test", "2", "--dev", "1"],
                capture_output=True, text=True, check=True,
            )
            with open(out, encoding="utf-8") as lines:
                rows = [line.rstrip("\n").split("\t") for line in lines]
            self.assertEqual([speaker for speaker, _ in rows], [f"spk{number}" for number in range(6)])
            self.assertEqual(collections.Counter(name for _, name in rows), {"train": 3, "test": 2, "dev": 1})
            self.assertIn("train: 3 speakers, 3 utterances", result.stdout)


if __name__ == "__main__":
    unittest.main()
