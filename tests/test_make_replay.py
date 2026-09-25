#!/usr/bin/env python3
"""Tests for scripts/make_replay.py."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import make_replay as replay  # noqa: E402

HEADER = replay.LABELS_HEADER + "\n"


def fleurs_row(file, transcript):
    """A line of a FLEURS <split>.tsv; make_replay reads only the file and the transcript."""
    return f"1\t{file}\t{transcript}\tnormalised\tc h a r s |\t16000\tFEMALE\n"


class NormaliseTests(unittest.TestCase):
    def test_words_ignore_case_and_punctuation(self):
        self.assertEqual(replay.words("Sir Richard Branson's Virgin Group, rejected."),
                         ["sir", "richard", "branson's", "virgin", "group", "rejected"])

    def test_apostrophes_are_one_kind_and_stay_inside_words(self):
        self.assertEqual(replay.words("the bank\N{RIGHT SINGLE QUOTATION MARK}s nationalisation"),
                         replay.words("the bank's nationalisation"))
        self.assertEqual(replay.words("'quoted' l'homme"), ["quoted", "l'homme"])

    def test_separators_inside_numbers_are_dropped(self):
        for text in ("10,000 BCE", "10.000 BCE", "10 000 BCE", "10\N{NO-BREAK SPACE}000 BCE"):
            with self.subTest(text=text):
                self.assertEqual(replay.words(text), ["10000", "bce"])
        self.assertEqual(replay.words("3,5 km"), replay.words("3.5 km"))
        self.assertEqual(replay.words("in 2004 2005"), ["in", "2004", "2005"])

    def test_hyphens_separate_words(self):
        self.assertEqual(replay.words("low-pressure air"), ["low", "pressure", "air"])

    def test_letters_keep_their_accents_and_fold_their_case(self):
        self.assertEqual(replay.words("Straße ÉTÉ"), ["strasse", "été"])

    def test_characters_drop_spaces_and_punctuation(self):
        self.assertEqual(replay.characters("洛杉矶，警察局。 (LA)"), list("洛杉矶警察局la"))
        self.assertEqual(replay.characters("１０％"), replay.characters("10%"))


class ErrorRateTests(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(replay.edit_distance("kitten", "sitting"), 3)
        self.assertEqual(replay.edit_distance([], ["a"]), 1)
        self.assertEqual(replay.edit_distance(["a", "b"], []), 2)

    def test_word_error_rate(self):
        self.assertAlmostEqual(replay.error_rate("the cat sat on the mat", "The cat sat on a mat.", False), 1 / 6)
        self.assertEqual(replay.error_rate("the cat", "", False), 1.0)
        self.assertEqual(replay.error_rate("the cat", "the cat sat on it", False), 1.5)

    def test_chinese_counts_characters(self):
        self.assertEqual(replay.error_rate("要了解圣殿骑士。", "要了解圣殿骑士", True), 0.0)
        self.assertAlmostEqual(replay.error_rate("马英九主张", "马云主张", True), 2 / 5)

    def test_an_empty_reference_has_no_rate(self):
        self.assertIsNone(replay.error_rate("…", "text", False))
        self.assertIsNone(replay.label_error("…", "text", replay.SOURCES["en_us"]))


class GlossTests(unittest.TestCase):
    chinese = replay.SOURCES["cmn_hans_cn"]

    def test_latin_glosses_are_dropped_and_chinese_brackets_kept(self):
        self.assertEqual(replay.without_glosses("克里斯托弗·加西亚（Christopher Garcia）表示，萨基斯 (Sakis) 和"),
                         "克里斯托弗·加西亚 表示，萨基斯 和")
        self.assertEqual(replay.without_glosses("北大西洋公约组织（简称北约）"), "北大西洋公约组织（简称北约）")

    def test_a_chinese_label_is_scored_without_the_glosses_it_didnt_hear(self):
        transcript = "而且太平洋海啸预警中心（Also the Pacific Tsunami Warning Center）也表示并未发现海啸迹象。"
        self.assertEqual(replay.label_error(transcript, "而且，太平洋海啸预警中心也表示，并未发现海啸迹象。", self.chinese), 0.0)

    def test_or_with_them_when_they_were_read(self):
        self.assertEqual(replay.label_error("加西亚（Garcia）表示", "加西亚Garcia表示", self.chinese), 0.0)

    def test_other_languages_keep_what_is_in_brackets(self):
        self.assertAlmostEqual(replay.label_error("100 Fuß (30 m) vor dem Zaun", "100 Fuß vor dem Zaun", replay.SOURCES["de_de"]), 2 / 7)


class ReadTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_fleurs_rows_give_the_file_and_transcript(self):
        self.assertEqual(replay.read_fleurs(self.write(fleurs_row("a.wav", "Hello, world."))), [("a.wav", "Hello, world.")])

    def test_a_fleurs_row_without_seven_columns_stops_the_run(self):
        with self.assertRaisesRegex(ValueError, "expected 7 columns"):
            replay.read_fleurs(self.write("151\ta.wav\tHello\n"))

    def test_labels(self):
        path = self.write(HEADER + "a.wav\t1.50\t40\tHello, world.\nb.wav\t2.00\t50\t\n")
        self.assertEqual(replay.read_label_file(path, {}), {"a.wav": (1.5, "Hello, world."), "b.wav": (2.0, "")})

    def test_labels_need_the_header_whole_lines_and_one_label_a_file(self):
        for text, message in (("a.wav\t1.50\t40\tHello\n", "expected the header"),
                              (HEADER + "a.wav\t1.50\t40\tHel", "cut off"),
                              (HEADER + "a.wav\t1.50\tHello\n", "expected 4 columns"),
                              (HEADER + "a.wav\tlong\t40\tHello\n", "aren't a number"),
                              (HEADER + "a.wav\t1.50\t40\tHi\na.wav\t1.50\t40\tHi\n", "labelled twice")):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                replay.read_label_file(self.write(text), {})

    def test_a_sources_labels_can_be_in_shards(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        os.makedirs(os.path.join(folder, "librispeech"))
        for shard, line in (("1", "a.flac\t1.00\t10\tOne\n"), ("2", "b.flac\t2.00\t20\tTwo\n")):
            with open(os.path.join(folder, "librispeech", f"{shard}.tsv"), "w", encoding="utf-8") as out:
                out.write(HEADER + line)
        self.assertEqual(replay.read_labels(folder, "librispeech"), {"a.flac": (1.0, "One"), "b.flac": (2.0, "Two")})

    def test_librispeech_transcripts(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        folder = os.path.join(root, "LibriSpeech", "train-clean-100", "19", "198")
        os.makedirs(folder)
        with open(os.path.join(folder, "19-198.trans.txt"), "w", encoding="utf-8") as out:
            out.write("19-198-0001 NORTHANGER ABBEY\n19-198-0000 CHAPTER ONE\n")
        self.assertEqual(replay.read_librispeech(root),
                         [("19-198-0000.flac", "CHAPTER ONE"), ("19-198-0001.flac", "NORTHANGER ABBEY")])
        self.assertEqual(replay.audio_path("librispeech", "19-198-0000.flac", {"librispeech": root}),
                         os.path.join(folder, "19-198-0000.flac"))
        with self.assertRaisesRegex(ValueError, "no LibriSpeech transcripts"):
            replay.read_librispeech(os.path.join(root, "elsewhere"))

    def test_replay_rows(self):
        self.assertEqual(replay.read_replay(self.write("en_us\ta.wav\t0.125\tHello, world.\n")),
                         [("en_us", "a.wav", 0.125, "Hello, world.")])

    def test_bad_replay_rows_stop_the_run(self):
        for text, message in (("xx_yy\ta.wav\t0.1\tHi\n", "unknown source"),
                              ("en_us\ta.wav\tlow\tHi\n", "isn't a rate"),
                              ("en_us\ta.wav\tnan\tHi\n", "isn't a rate"),
                              ("en_us\ta.wav\t-0.1\tHi\n", "isn't a rate"),
                              ("en_us\ta.wav\t0.1\n", "expected 4 columns")):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                replay.read_replay(self.write(text))

    def test_sources(self):
        self.assertEqual(replay.parse_sources("en_us, librispeech"), ["en_us", "librispeech"])
        with self.assertRaisesRegex(ValueError, "unknown sources"):
            replay.parse_sources("en_us,si_lk")
        with self.assertRaisesRegex(ValueError, "unknown sources"):
            replay.parse_sources("librispeech", replay.FLEURS_CODES)
        with self.assertRaises(ValueError):
            replay.parse_sources("")


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)

    def write(self, relative, text):
        path = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as out:
            out.write(text)

    def run_script(self, sources="en_us,cmn_hans_cn", *options):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "make_replay.py"), "--fleurs", os.path.join(self.root, "fleurs"),
             "--labels", os.path.join(self.root, "labels"), "--out", os.path.join(self.root, "replay.tsv"),
             "--sources", sources, *options],
            capture_output=True, text=True,
        )

    def test_scores_every_utterance_in_language_and_file_order(self):
        self.write("fleurs/data/en_us/train.tsv",
                   fleurs_row("b.wav", "The cat sat on the mat.") + fleurs_row("a.wav", "Hello there, world."))
        self.write("fleurs/data/cmn_hans_cn/train.tsv", fleurs_row("c.wav", "马英九主张"))
        self.write("labels/en_us.tsv", HEADER + "b.wav\t2.00\t20\tThe cat sat on a mat.\na.wav\t1.00\t10\tHello there world\n")
        self.write("labels/cmn_hans_cn.tsv", HEADER + "c.wav\t3.00\t30\t马云主张。\n")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(os.path.join(self.root, "replay.tsv"), encoding="utf-8") as table:
            self.assertEqual(table.read(),
                             "en_us\ta.wav\t0.000\tHello there world\n"
                             "en_us\tb.wav\t0.167\tThe cat sat on a mat.\n"
                             "cmn_hans_cn\tc.wav\t0.400\t马云主张。\n")
        self.assertRegex(result.stdout, r"total\s+3\s")

    def test_a_missing_label_stops_the_run(self):
        self.write("fleurs/data/en_us/train.tsv", fleurs_row("a.wav", "Hello.") + fleurs_row("b.wav", "Bye."))
        self.write("fleurs/data/cmn_hans_cn/train.tsv", "")
        self.write("labels/en_us.tsv", HEADER + "a.wav\t1.00\t10\tHello.\n")
        self.write("labels/cmn_hans_cn.tsv", HEADER)
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("en_us: 1 utterances have no label, such as ['b.wav']", result.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.root, "replay.tsv")))

    def test_a_label_without_a_transcript_stops_the_run(self):
        self.write("fleurs/data/en_us/train.tsv", fleurs_row("a.wav", "Hello."))
        self.write("labels/en_us.tsv", HEADER + "a.wav\t1.00\t10\tHello.\nz.wav\t1.00\t10\tWho?\n")
        result = self.run_script("en_us")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("en_us: 1 labels are for recordings without a transcript, such as ['z.wav']", result.stderr)

    def test_scores_the_librispeech_utterances_that_were_labelled(self):
        folder = os.path.join("librispeech", "LibriSpeech", "train-clean-100", "19", "198")
        self.write(os.path.join(folder, "19-198.trans.txt"),
                   "19-198-0000 CHAPTER ONE\n19-198-0001 NORTHANGER ABBEY\n19-198-0002 NOT PICKED\n")
        self.write("labels/librispeech/1.tsv", HEADER + "19-198-0001.flac\t2.00\t20\tNorthanger Abbey.\n")
        self.write("labels/librispeech/2.tsv", HEADER + "19-198-0000.flac\t1.00\t10\tChapter 1.\n")
        result = self.run_script("librispeech", "--librispeech", os.path.join(self.root, "librispeech"))
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(os.path.join(self.root, "replay.tsv"), encoding="utf-8") as table:
            self.assertEqual(table.read(),
                             "librispeech\t19-198-0000.flac\t0.500\tChapter 1.\n"
                             "librispeech\t19-198-0001.flac\t0.000\tNorthanger Abbey.\n")

    def test_a_source_needs_its_corpus_folder(self):
        result = self.run_script("librispeech")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("give --librispeech for the sources librispeech", result.stderr)


if __name__ == "__main__":
    unittest.main()
