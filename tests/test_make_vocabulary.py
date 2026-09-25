#!/usr/bin/env python3
"""Tests for scripts/make_vocabulary.py."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import make_vocabulary  # noqa: E402

ZWJ = "\N{ZERO WIDTH JOINER}"


class CountWordsTests(unittest.TestCase):
    def count(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return make_vocabulary.count_words(handle.name)

    def test_counts_words_most_frequent_first(self):
        self.assertEqual(self.count("aa01\tspk1\tඅපි ගෙදර යමු\naa02\tspk2\tඅපි යමු, අපි!\n"),
                         [("අපි", 3), ("යමු", 2), ("ගෙදර", 1)])

    def test_leaves_out_latin_letters_digits_and_punctuation(self):
        self.assertEqual(self.count("aa01\tspk1\tFacebook 2018 එකේ (ෆේස්බුක්)\n"),
                         [("එකේ", 1), ("ෆේස්බුක්", 1)])

    def test_keeps_joiners_inside_words(self):
        self.assertEqual(self.count("aa01\tspk1\tශ්" + ZWJ + "රී ලංකාව\n"),
                         [("ශ්" + ZWJ + "රී", 1), ("ලංකාව", 1)])

    def test_ties_keep_the_order_words_first_appear_in(self):
        self.assertEqual([word for word, _ in self.count("aa01\tspk1\tයමු ගෙදර අපි\n")], ["යමු", "ගෙදර", "අපි"])

    def test_skips_lines_without_a_transcript(self):
        self.assertEqual(self.count("aa01\tspk1\n\naa02\tspk2\tඅපි\n"), [("අපි", 1)])


class EndToEndTests(unittest.TestCase):
    def test_writes_the_vocabulary_the_frequent_words_and_their_chunks(self):
        with tempfile.TemporaryDirectory() as root:
            transcripts = os.path.join(root, "utt_spk_text.tsv")
            with open(transcripts, "w", encoding="utf-8") as tsv:
                tsv.write("aa01\tspk1\tඅපි ගෙදර යමු හෙට උදේ රෑ\n")
                tsv.write("aa02\tspk2\tඅපි ගෙදර යමු හෙට උදේ\n")
            out = os.path.join(root, "vocab")
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "make_vocabulary.py"), transcripts, out,
                 "--min-count", "2", "--chunk-size", "2"],
                capture_output=True, text=True, check=True,
            )

            def read(*parts):
                with open(os.path.join(out, *parts), encoding="utf-8") as lines:
                    return [line.rstrip("\n").split("\t")[0] for line in lines]

            self.assertEqual(read("vocabulary.tsv"), ["අපි", "ගෙදර", "යමු", "හෙට", "උදේ", "රෑ"])
            self.assertEqual(read("frequent.tsv"), ["අපි", "ගෙදර", "යමු", "හෙට", "උදේ"])
            self.assertEqual(sorted(os.listdir(os.path.join(out, "chunks"))),
                             ["words-00.tsv", "words-01.tsv", "words-02.tsv"])
            self.assertEqual(read("chunks", "words-02.tsv"), ["උදේ"])
            self.assertIn("11 words, 6 distinct. 5 occur at least 2 times and cover 90.9%", result.stdout)


if __name__ == "__main__":
    unittest.main()
