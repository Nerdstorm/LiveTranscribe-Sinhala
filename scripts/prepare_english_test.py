#!/usr/bin/env python3
r"""Writes the English check's records: a FLEURS English split, each recording with its FLEURS
transcript as the reference.

    prepare_english_test.py --fleurs <FLEURS folder> --out <dir> [--split test|dev] [--check-audio]

Writes <dir>/fleurs_en_<split>.jsonl in the format of the other JSONL files, one line a recording:

    {"audio": "<the recording>", "text": "language English<asr_text><FLEURS's transcript>"}

The recordings are where `fetch_fleurs.py --languages en_us --splits <split> --extract` puts them.
None of the sentences in the test or dev split is in the replay set, which is made from the train
split. The trainer checks English on the dev split while it trains, and its `transcribe` command
scores the final model on the test split (docs/training.md). Stdlib only.
"""
import argparse
import json
import os
import sys

from make_replay import read_fleurs

CODE = "en_us"
SPLITS = ("test", "dev")


def records(fleurs, split="test"):
    """The split's recordings, in the transcript file's order."""
    folder = os.path.join(fleurs, "data", CODE)
    return [{"audio": os.path.join(folder, "audio", split, file), "text": f"language English<asr_text>{transcript}"}
            for file, transcript in read_fleurs(os.path.join(folder, f"{split}.tsv"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fleurs", required=True, help="the folder fetch_fleurs.py wrote, with the split's audio extracted")
    parser.add_argument("--out", required=True, help="folder for fleurs_en_<split>.jsonl")
    parser.add_argument("--split", choices=SPLITS, default="test", help="the FLEURS split (default test)")
    parser.add_argument("--check-audio", action="store_true", help="skip recordings that are missing")
    args = parser.parse_args()

    try:
        found = records(os.path.abspath(args.fleurs), args.split)
    except (OSError, ValueError) as error:
        sys.exit(f"prepare_english_test: {error}")
    missing = [record for record in found if args.check_audio and not os.path.exists(record["audio"])]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"fleurs_en_{args.split}.jsonl"), "w", encoding="utf-8") as out:
        for record in found:
            if record not in missing:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"fleurs_en_{args.split}: {len(found) - len(missing)}")
    print(f"missing audio: {len(missing)}")


if __name__ == "__main__":
    main()
