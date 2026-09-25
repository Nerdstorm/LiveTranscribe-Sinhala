#!/usr/bin/env python3
"""Checks the committed files in data/, so an edit that would break a run fails here first."""
import collections
import sys
import unicodedata
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_replay  # noqa: E402
import make_split  # noqa: E402
import prepare_data  # noqa: E402

SINHALA = range(0x0D80, 0x0E00)


def read(name):
    with open(ROOT / "data" / name, encoding="utf-8") as lines:
        return [line.rstrip("\n").split("\t") for line in lines]


class SpeakerSplitTests(unittest.TestCase):
    def test_is_the_seed_52_split_of_openslr_52s_478_speakers(self):
        committed = dict(read("speaker-split.tsv"))
        self.assertEqual(len(committed), 478)
        self.assertEqual(collections.Counter(committed.values()), {"train": 442, "test": 24, "dev": 12})
        self.assertEqual(make_split.split_speakers(dict.fromkeys(committed, 1), 52, 24, 12), committed)


class LoanwordTableTests(unittest.TestCase):
    rows = read("loanwords.tsv")

    def test_every_row_is_well_formed(self):
        for number, row in enumerate(self.rows, 1):
            with self.subTest(line=number, row=row):
                self.assertEqual(len(row), 4)
                word, category, english, suffix = row
                self.assertIn(category, {"E", "N", "P"})
                self.assertEqual(unicodedata.normalize("NFC", word), word)
                self.assertTrue(english and all(character.isalpha() or character in " -" for character in english))
                self.assertFalse(any(ord(character) in SINHALA for character in english))
                self.assertTrue(word.endswith(suffix) and suffix != word)

    def test_each_word_is_listed_once(self):
        words = collections.Counter(row[0] for row in self.rows)
        self.assertEqual([word for word, count in words.items() if count > 1], [])

    def test_words_that_prepare_data_reads_as_one_agree(self):
        # A stray joiner at the start of a word makes a second row for it; cleaning merges them.
        seen = {}
        for word, category, english, suffix in self.rows:
            entry = (category, english, suffix.strip(prepare_data.JOINERS))
            with self.subTest(word=word):
                self.assertEqual(seen.setdefault(prepare_data.clean(word), entry), entry)


class ReplayTableTests(unittest.TestCase):
    rows = read("replay.tsv")
    # FLEURS's train utterances per language, at fetch_fleurs.REVISION, and pick_librispeech.py's
    # default count.
    utterances = {"en_us": 2602, "cmn_hans_cn": 3246, "es_419": 2796, "fr_fr": 3193, "de_de": 2987,
                  "librispeech": 16000}

    def test_has_every_fleurs_train_utterance_and_the_librispeech_picks_once(self):
        self.assertEqual(collections.Counter(row[0] for row in self.rows), self.utterances)
        self.assertEqual(len({(row[0], row[1]) for row in self.rows}), len(self.rows))

    def test_every_row_is_well_formed(self):
        for number, row in enumerate(self.rows, 1):
            with self.subTest(line=number, row=row):
                self.assertEqual(len(row), 4)
                code, file, error, label = row
                self.assertRegex(file, r"^[0-9]+\.wav$" if code != "librispeech" else r"^[0-9]+-[0-9]+-[0-9]{4}\.flac$")
                self.assertRegex(error, r"^[0-9]+\.[0-9]{3}$")
                self.assertEqual(label, label.strip())

    def test_is_in_source_then_file_order(self):
        order = [(list(make_replay.SOURCES).index(code), file) for code, file, _, _ in self.rows]
        self.assertEqual(order, sorted(order))

    def test_prepare_replay_can_read_it(self):
        self.assertEqual(len(make_replay.read_replay(ROOT / "data" / "replay.tsv")), len(self.rows))


if __name__ == "__main__":
    unittest.main()
