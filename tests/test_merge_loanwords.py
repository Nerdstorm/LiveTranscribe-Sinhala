#!/usr/bin/env python3
"""Tests for scripts/merge_loanwords.py."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import merge_loanwords  # noqa: E402

COUNTS = {"අපි": 90, "බ්ලොග්": 50, "පොලිසිය": 30, "ඉන්දියාව": 20, "කොම්පියුටර්": 10}


class MergeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def write(self, name, text):
        path = os.path.join(self.folder, name)
        with open(path, "w", encoding="utf-8") as tsv:
            tsv.write(text)
        return path

    def test_orders_rows_most_frequent_first_across_files(self):
        first = self.write("words-00.tsv", "කොම්පියුටර්\tE\tcomputer\t\n\nබ්ලොග්\tE\tblog\t\n")
        second = self.write("words-01.tsv", "පොලිසිය\tN\tpolice\tිය\n")
        rows, problems, duplicates = merge_loanwords.merge([first, second], COUNTS)
        self.assertEqual([row[0] for row in rows], ["බ්ලොග්", "පොලිසිය", "කොම්පියුටර්"])
        self.assertEqual((problems, duplicates), ([], 0))

    def test_restores_an_empty_suffix_whose_tab_was_dropped(self):
        path = self.write("words-00.tsv", "බ්ලොග්\tE\tblog\n")
        self.assertEqual(merge_loanwords.merge([path], COUNTS), ([["බ්ලොග්", "E", "blog", ""]], [], 0))

    def test_a_word_listed_twice_keeps_its_first_line(self):
        first = self.write("words-00.tsv", "බ්ලොග්\tE\tblog\t\n")
        second = self.write("words-01.tsv", "බ්ලොග්\tP\tBlog\t\n")
        self.assertEqual(merge_loanwords.merge([first, second], COUNTS), ([["බ්ලොග්", "E", "blog", ""]], [], 1))

    def test_reports_every_bad_line_with_its_place(self):
        path = self.write("words-00.tsv", "".join(line + "\n" for line in (
            "බ්ලොග්\tE",
            "බ්ලොග්\tX\tblog\t",
            "බ්ලොග්\tE\t\t",
            "ගෙදර\tE\thome\t",
            "බ්ලොග්\tE\tblog\t\textra",
        )))
        rows, problems, _ = merge_loanwords.merge([path], COUNTS)
        self.assertEqual(rows, [])
        self.assertEqual(problems, [
            f"{path}:1: expected 4 columns, got 2",
            f"{path}:2: unknown category 'X'",
            f"{path}:3: no English spelling",
            f"{path}:4: {'ගෙදර'!r} is not in the vocabulary",
            f"{path}:5: expected 4 columns, got 5",
        ])


class EndToEndTests(unittest.TestCase):
    def run_merge(self, classified):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        vocabulary = os.path.join(root.name, "vocabulary.tsv")
        with open(vocabulary, "w", encoding="utf-8") as tsv:
            tsv.write("".join(f"{word}\t{count}\n" for word, count in COUNTS.items()))
        folder = os.path.join(root.name, "out")
        os.makedirs(folder)
        for name, text in classified.items():
            with open(os.path.join(folder, name), "w", encoding="utf-8") as tsv:
                tsv.write(text)
        out = os.path.join(root.name, "loanwords.tsv")
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "merge_loanwords.py"), vocabulary, folder, out],
            capture_output=True, text=True,
        )
        return result, out

    def test_writes_the_table_and_the_share_of_each_category(self):
        result, out = self.run_merge({
            "words-00.tsv": "කොම්පියුටර්\tE\tcomputer\t\nබ්ලොග්\tE\tblog\t\n",
            "words-01.tsv": "පොලිසිය\tN\tpolice\tිය\nඉන්දියාව\tP\tIndia\tව\n",
            "notes.txt": "not a classifier file\n",
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(out, encoding="utf-8") as table:
            self.assertEqual(table.read(), "බ්ලොග්\tE\tblog\t\nපොලිසිය\tN\tpolice\tිය\n"
                                           "ඉන්දියාව\tP\tIndia\tව\nකොම්පියුටර්\tE\tcomputer\t\n")
        self.assertIn("E: 2 words, 30.00% of all words", result.stdout)
        self.assertIn("N: 1 words, 15.00% of all words", result.stdout)
        self.assertIn("4 words from 2 files, 0 duplicates skipped", result.stdout)

    def test_a_bad_line_stops_the_run_before_anything_is_written(self):
        result, out = self.run_merge({"words-00.tsv": "බ්ලොග්\tE\tblog\t\nබ්ලොග්\tX\tblog\t\n"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("words-00.tsv:2: unknown category 'X'", result.stderr)
        self.assertIn("1 lines need fixing; nothing written", result.stderr)
        self.assertFalse(os.path.exists(out))

    def test_a_folder_without_classifier_files_stops_the_run(self):
        result, out = self.run_merge({})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no words-*.tsv", result.stderr)
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
