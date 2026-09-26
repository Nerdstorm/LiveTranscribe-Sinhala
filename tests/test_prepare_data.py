#!/usr/bin/env python3
"""Tests for scripts/prepare_data.py."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import prepare_data as prep  # noqa: E402

ZWJ = "\N{ZERO WIDTH JOINER}"


class CleanTests(unittest.TestCase):
    def test_keeps_joiners_inside_conjuncts(self):
        self.assertEqual(prep.clean("ශ්" + ZWJ + "රී ලංකාව"), "ශ්" + ZWJ + "රී ලංකාව")

    def test_drops_joiners_at_word_edges_and_collapses_spaces(self):
        self.assertEqual(prep.clean(ZWJ + "මියුසික්  වලින්" + ZWJ + " "), "මියුසික් වලින්")

    def test_normalises_to_nfc(self):
        decomposed = "\N{SINHALA VOWEL SIGN KOMBUVA}\N{SINHALA SIGN AL-LAKUNA}"  # NFC composes these two
        ka = "\N{SINHALA LETTER ALPAPRAANA KAYANNA}"
        self.assertEqual(prep.clean(ka + decomposed), ka + "\N{SINHALA VOWEL SIGN DIGA KOMBUVA}")


class RewriteTests(unittest.TestCase):
    table = {
        "කොම්පියුටර්": ("computer", ""),
        "ආමිඑක": ("army", "එක"),
        "කමෙන්ටුවක්": ("comment", "ුවක්"),
        "ෆයිල්වල": ("file", "වල"),
    }

    def test_bare_word_becomes_english(self):
        self.assertEqual(prep.rewrite("මගේ කොම්පියුටර් හොඳයි", self.table), "මගේ computer හොඳයි")

    def test_particle_follows_the_english_word(self):
        self.assertEqual(prep.rewrite("ආමිඑක ආවා", self.table), "army එක ආවා")
        self.assertEqual(prep.rewrite("ෆයිල්වල තියෙනවා", self.table), "file වල තියෙනවා")

    def test_other_inflected_forms_stay_sinhala(self):
        self.assertEqual(prep.rewrite("කමෙන්ටුවක් දාන්න", self.table), "කමෙන්ටුවක් දාන්න")

    def test_words_not_in_the_table_stay(self):
        self.assertEqual(prep.rewrite("අපි ගෙදර යමු", self.table), "අපි ගෙදර යමු")


class SentenceTests(unittest.TestCase):
    def test_key_ignores_case_punctuation_and_spacing(self):
        self.assertEqual(prep.sentence_key("මේ film එක බලන්න."), prep.sentence_key("මේ  Film එක, බලන්න"))
        self.assertEqual(prep.sentence_key("“අපි යමු!”"), "අපි යමු")

    def test_key_keeps_joiners(self):
        self.assertNotEqual(prep.sentence_key("ශ්" + ZWJ + "රී"), prep.sentence_key("ශ්රී"))

    def records(self, split, *texts):
        return [(prep.sentence_key(text), {"text": text, "split": split}) for text in texts]

    def test_new_sentences_are_those_not_in_train(self):
        records = {"train": self.records("train", "අපි යමු", "ගෙදර යමු"),
                   "dev": self.records("dev", "අපි යමු.", "වෙන දෙයක්"),
                   "test": self.records("test", "ගෙදර යමු", "අලුත් වාක්‍යයක්")}
        files, counts = prep.split_sentences(records, hold_out=False)
        self.assertEqual([r["text"] for r in files["train"]], ["අපි යමු", "ගෙදර යමු"])
        self.assertEqual([r["text"] for r in files["dev"]], ["අපි යමු.", "වෙන දෙයක්"])
        self.assertEqual([r["text"] for r in files["dev_new"]], ["වෙන දෙයක්"])
        self.assertEqual([r["text"] for r in files["test_new"]], ["අලුත් වාක්‍යයක්"])
        self.assertEqual(counts, {"train held out": 0})

    def test_holding_out_sentences_empties_train_of_dev_and_test_sentences(self):
        records = {"train": self.records("train", "අපි යමු", "ගෙදර යමු", "තව එකක්"),
                   "dev": self.records("dev", "අපි යමු."),
                   "test": self.records("test", "ගෙදර යමු")}
        files, counts = prep.split_sentences(records, hold_out=True)
        self.assertEqual([r["text"] for r in files["train"]], ["තව එකක්"])
        self.assertEqual(files["dev_new"], files["dev"])
        self.assertEqual(files["test_new"], files["test"])
        self.assertEqual(counts, {"train held out": 2})


class LoadTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_only_chosen_categories_are_loaded(self):
        path = self.write("කොම්පියුටර්\tE\tcomputer\t\nපොලිසිය\tN\tpolice\t\nඉන්දියාව\tP\tIndia\t\n")
        self.assertEqual(set(prep.load_loanwords(path, {"E"})), {"කොම්පියුටර්"})
        self.assertEqual(set(prep.load_loanwords(path, {"E", "P"})), {"කොම්පියුටර්", "ඉන්දියාව"})

    def test_bad_rows_stop_the_load(self):
        with self.assertRaises(ValueError):
            prep.load_loanwords(self.write("කොම්පියුටර්\tX\tcomputer\t\n"), {"E"})
        with self.assertRaises(ValueError):
            prep.load_loanwords(self.write("කොම්පියුටර්\tE\tcomputer\n"), {"E"})
        with self.assertRaises(ValueError):
            prep.load_split(self.write("00349\tvalidation\n"))


class EndToEndTests(unittest.TestCase):
    def test_writes_each_speaker_to_its_split(self):
        with tempfile.TemporaryDirectory() as root:
            data = os.path.join(root, "asr_sinhala")
            os.makedirs(os.path.join(data, "data", "aa"))
            with open(os.path.join(data, "utt_spk_text.tsv"), "w", encoding="utf-8") as tsv:
                tsv.write("aa01\tspk1\tමගේ කොම්පියුටර් හොඳයි\n")
                tsv.write("aa02\tspk2\tඅපි ගෙදර යමු\n")
                tsv.write("aa03\tspk3\t" + ZWJ + "\n")
            for utterance in ("aa01", "aa02", "aa03"):
                open(os.path.join(data, "data", "aa", f"{utterance}.flac"), "wb").close()
            split = os.path.join(root, "split.tsv")
            with open(split, "w", encoding="utf-8") as tsv:
                tsv.write("spk1\ttrain\nspk2\ttest\nspk3\tdev\n")
            loanwords = os.path.join(root, "loanwords.tsv")
            with open(loanwords, "w", encoding="utf-8") as tsv:
                tsv.write("කොම්පියුටර්\tE\tcomputer\t\n")
            out = os.path.join(root, "out")
            script = str(SCRIPTS / "prepare_data.py")
            result = subprocess.run(
                [sys.executable, script, "--data-dir", data, "--split", split, "--loanwords", loanwords,
                 "--out", out, "--check-audio"],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("train: 1", result.stdout)
            self.assertIn("test_new: 1", result.stdout)
            self.assertIn("empty: 1", result.stdout)

            def records(name):
                with open(os.path.join(out, f"{name}.jsonl"), encoding="utf-8") as lines:
                    return [json.loads(line) for line in lines]

            self.assertEqual(records("train"), [{
                "audio": os.path.join(data, "data", "aa", "aa01.flac"),
                "text": "language Sinhala<asr_text>මගේ computer හොඳයි",
            }])
            self.assertEqual(records("test")[0]["text"], "language Sinhala<asr_text>අපි ගෙදර යමු")
            self.assertEqual(records("test_new"), records("test"))
            self.assertEqual(records("dev"), [])
            self.assertEqual(records("dev_new"), [])
            with open(os.path.join(out, "rewrites.tsv"), encoding="utf-8") as report:
                self.assertEqual(report.read(), "කොම්පියුටර්\tcomputer\t1\n")

    def test_holding_out_sentences_keeps_a_test_sentence_out_of_train(self):
        with tempfile.TemporaryDirectory() as root:
            data = os.path.join(root, "asr_sinhala")
            os.makedirs(os.path.join(data, "data", "aa"))
            with open(os.path.join(data, "utt_spk_text.tsv"), "w", encoding="utf-8") as tsv:
                tsv.write("aa01\tspk1\tඅපි ගෙදර යමු\n")
                tsv.write("aa02\tspk1\tතව එකක්\n")
                tsv.write("aa03\tspk2\tඅපි ගෙදර යමු.\n")
            split = os.path.join(root, "split.tsv")
            with open(split, "w", encoding="utf-8") as tsv:
                tsv.write("spk1\ttrain\nspk2\ttest\n")
            loanwords = os.path.join(root, "loanwords.tsv")
            open(loanwords, "w").close()
            out = os.path.join(root, "out")
            command = [sys.executable, str(SCRIPTS / "prepare_data.py"), "--data-dir", data, "--split", split,
                       "--loanwords", loanwords, "--out", out]

            def texts(name):
                with open(os.path.join(out, f"{name}.jsonl"), encoding="utf-8") as lines:
                    return [json.loads(line)["text"].split("<asr_text>")[1] for line in lines]

            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn("test_new: 0", result.stdout)
            self.assertEqual(texts("train"), ["අපි ගෙදර යමු", "තව එකක්"])

            result = subprocess.run(command + ["--hold-out-sentences"], capture_output=True, text=True, check=True)
            self.assertIn("train held out: 1", result.stdout)
            self.assertEqual(texts("train"), ["තව එකක්"])
            self.assertEqual(texts("test_new"), ["අපි ගෙදර යමු."])

    def test_a_speaker_without_a_split_stops_the_run(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "data"))
            with open(os.path.join(root, "data", "utt_spk_text.tsv"), "w", encoding="utf-8") as tsv:
                tsv.write("aa01\tstranger\tඅපි\n")
            for name, text in (("split.tsv", "spk1\ttrain\n"), ("loanwords.tsv", "")):
                with open(os.path.join(root, name), "w", encoding="utf-8") as tsv:
                    tsv.write(text)
            script = str(SCRIPTS / "prepare_data.py")
            result = subprocess.run(
                [sys.executable, script, "--data-dir", os.path.join(root, "data"),
                 "--split", os.path.join(root, "split.tsv"), "--loanwords", os.path.join(root, "loanwords.tsv"),
                 "--out", os.path.join(root, "out")],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("1 speakers have no split", result.stderr)


if __name__ == "__main__":
    unittest.main()
