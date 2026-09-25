#!/usr/bin/env python3
"""Tests for scripts/prepare_english_test.py."""
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

import prepare_english_test as prep  # noqa: E402


class EnglishTestTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.fleurs = os.path.join(self.root, "fleurs")
        folder = os.path.join(self.fleurs, "data", "en_us")
        os.makedirs(os.path.join(folder, "audio", "test"))
        with open(os.path.join(folder, "test.tsv"), "w", encoding="utf-8") as table:
            table.write("1\ta.wav\tThe cat sat.\tthe cat sat\tt h e\t16000\tFEMALE\n"
                        "2\tb.wav\tIt's 5 o'clock.\tit's five o'clock\ti t\t32000\tMALE\n")
        open(os.path.join(folder, "audio", "test", "a.wav"), "wb").close()

    def run_script(self, *options):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "prepare_english_test.py"), "--fleurs", self.fleurs,
             "--out", os.path.join(self.root, "jsonl"), *options],
            capture_output=True, text=True,
        )

    def written(self):
        with open(os.path.join(self.root, "jsonl", "fleurs_en_test.jsonl"), encoding="utf-8") as lines:
            return [json.loads(line) for line in lines]

    def test_each_recording_has_its_transcript_as_the_reference(self):
        found = prep.records(self.fleurs)
        self.assertEqual([record["text"] for record in found],
                         ["language English<asr_text>The cat sat.", "language English<asr_text>It's 5 o'clock."])
        self.assertEqual(found[0]["audio"], os.path.join(self.fleurs, "data", "en_us", "audio", "test", "a.wav"))

    def test_writes_every_recording(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.written()), 2)
        self.assertIn("fleurs_en_test: 2", result.stdout)

    def test_check_audio_skips_missing_recordings(self):
        result = self.run_script("--check-audio")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([os.path.basename(record["audio"]) for record in self.written()], ["a.wav"])
        self.assertIn("missing audio: 1", result.stdout)

    def test_the_dev_split_is_written_to_its_own_file(self):
        folder = os.path.join(self.fleurs, "data", "en_us")
        os.makedirs(os.path.join(folder, "audio", "dev"))
        with open(os.path.join(folder, "dev.tsv"), "w", encoding="utf-8") as table:
            table.write("3\tc.wav\tA dog ran.\ta dog ran\ta d\t16000\tMALE\n")
        result = self.run_script("--split", "dev")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fleurs_en_dev: 1", result.stdout)
        with open(os.path.join(self.root, "jsonl", "fleurs_en_dev.jsonl"), encoding="utf-8") as lines:
            written = [json.loads(line) for line in lines]
        self.assertEqual(written, [{"audio": os.path.join(folder, "audio", "dev", "c.wav"),
                                    "text": "language English<asr_text>A dog ran."}])

    def test_a_missing_transcript_file_is_an_error(self):
        os.remove(os.path.join(self.fleurs, "data", "en_us", "test.tsv"))
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prepare_english_test:", result.stderr)


if __name__ == "__main__":
    unittest.main()
