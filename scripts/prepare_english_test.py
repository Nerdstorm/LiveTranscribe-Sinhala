#!/usr/bin/env python3
r"""Writes the English check's records: FLEURS's English test split, each recording with its
FLEURS transcript as the reference.

    prepare_english_test.py --fleurs <FLEURS folder> --out <dir> [--check-audio]

Writes <dir>/fleurs_en_test.jsonl in the format of the other JSONL files, one line a recording:

    {"audio": "<the recording>", "text": "language English<asr_text><FLEURS's transcript>"}

The recordings are where `fetch_fleurs.py --languages en_us --splits test --extract` puts them.
None of the split's sentences is in the replay set, which is made from the train split. The
trainer's `transcribe` command scores a model on them (docs/training.md). Stdlib only.
"""
import argparse
import json
import os
import sys

from make_replay import read_fleurs

CODE = "en_us"
SPLIT = "test"


def records(fleurs):
    """The test split's recordings, in the transcript file's order."""
    folder = os.path.join(fleurs, "data", CODE)
    return [{"audio": os.path.join(folder, "audio", SPLIT, file), "text": f"language English<asr_text>{transcript}"}
            for file, transcript in read_fleurs(os.path.join(folder, f"{SPLIT}.tsv"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fleurs", required=True, help="the folder fetch_fleurs.py wrote, with the test audio extracted")
    parser.add_argument("--out", required=True, help="folder for fleurs_en_test.jsonl")
    parser.add_argument("--check-audio", action="store_true", help="skip recordings that are missing")
    args = parser.parse_args()

    try:
        found = records(os.path.abspath(args.fleurs))
    except (OSError, ValueError) as error:
        sys.exit(f"prepare_english_test: {error}")
    missing = [record for record in found if args.check_audio and not os.path.exists(record["audio"])]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fleurs_en_test.jsonl"), "w", encoding="utf-8") as out:
        for record in found:
            if record not in missing:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"fleurs_en_test: {len(found) - len(missing)}")
    print(f"missing audio: {len(missing)}")


if __name__ == "__main__":
    main()
