#!/usr/bin/env python3
"""Tests for scripts/prepare_replay.py."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import prepare_replay as prep  # noqa: E402


class RecordTests(unittest.TestCase):
    folders = {"fleurs": "/f", "librispeech": "/l"}

    def test_a_record_names_the_language_as_qwen_does(self):
        self.assertEqual(prep.record("cmn_hans_cn", "c.wav", "马云主张。", self.folders), {
            "audio": os.path.join("/f", "data", "cmn_hans_cn", "audio", "train", "c.wav"),
            "text": "language Chinese<asr_text>马云主张。",
        })
        self.assertEqual(prep.record("librispeech", "19-198-0000.flac", "Chapter one.", self.folders), {
            "audio": os.path.join("/l", "LibriSpeech", "train-clean-100", "19", "198", "19-198-0000.flac"),
            "text": "language English<asr_text>Chapter one.",
        })


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.fleurs = os.path.join(self.root, "fleurs")
        for code, file in (("en_us", "a.wav"), ("en_us", "b.wav"), ("cmn_hans_cn", "c.wav")):
            folder = os.path.join(self.fleurs, "data", code, "audio", "train")
            os.makedirs(folder, exist_ok=True)
            open(os.path.join(folder, file), "wb").close()
        self.replay = os.path.join(self.root, "replay.tsv")
        with open(self.replay, "w", encoding="utf-8") as table:
            table.write("en_us\ta.wav\t0.000\tHello there world\n"
                        "en_us\tb.wav\t0.400\tThe cat sat on a hat.\n"
                        "cmn_hans_cn\tc.wav\t0.300\t马云主张。\n"
                        "de_de\td.wav\t0.100\tHallo.\n")

    def run_script(self, *options):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "prepare_replay.py"), "--fleurs", self.fleurs, "--replay", self.replay,
             "--out", os.path.join(self.root, "jsonl"), *options],
            capture_output=True, text=True,
        )

    def records(self):
        with open(os.path.join(self.root, "jsonl", "replay.jsonl"), encoding="utf-8") as lines:
            return [json.loads(line) for line in lines]

    def test_keeps_labels_within_the_error_whose_audio_is_there(self):
        result = self.run_script("--check-audio")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.records(), [
            {"audio": os.path.join(self.fleurs, "data", "en_us", "audio", "train", "a.wav"),
             "text": "language English<asr_text>Hello there world"},
            {"audio": os.path.join(self.fleurs, "data", "cmn_hans_cn", "audio", "train", "c.wav"),
             "text": "language Chinese<asr_text>马云主张。"},
        ])
        self.assertIn("en_us: 1 kept, 1 over 0.3", result.stdout)
        self.assertIn("replay: 2", result.stdout)
        self.assertIn("missing audio: 1", result.stdout)

    def test_max_error_moves_the_cut(self):
        result = self.run_script("--max-error", "0.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([record["text"] for record in self.records()],
                         ["language English<asr_text>Hello there world", "language German<asr_text>Hallo."])

    def test_librispeech_rows_need_its_folder(self):
        with open(self.replay, "a", encoding="utf-8") as table:
            table.write("librispeech\t19-198-0000.flac\t0.000\tChapter one.\n")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("has librispeech utterances; give --librispeech", result.stderr)

    def test_a_negative_max_error_is_refused(self):
        result = self.run_script("--max-error", "-1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("can't be negative", result.stderr)


if __name__ == "__main__":
    unittest.main()
