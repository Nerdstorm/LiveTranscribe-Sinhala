#!/usr/bin/env python3
r"""Writes a FLEURS check's records: a split of one FLEURS language, each recording with its FLEURS
transcript as the reference.

    prepare_fleurs_test.py --fleurs <FLEURS folder> --out <dir> [--language en_us]
        [--split test|dev] [--check-audio]

Writes <dir>/fleurs_<language>_<split>.jsonl in the format of the other JSONL files, one line a
recording, and names the language as Qwen3-ASR does:

    {"audio": "<the recording>", "text": "language English<asr_text><FLEURS's transcript>"}

The file's language is two letters: fleurs_en_test.jsonl for en_us, fleurs_zh_test.jsonl for
cmn_hans_cn, and es, fr and de. Chinese references drop the bracketed glosses that FLEURS's
transcripts add after a transliterated name, "加西亚（Christopher Garcia）", as the speakers don't
read them. The recordings are where `fetch_fleurs.py --languages <code> --splits <split> --extract`
puts them. The replay set is made from the train split. The trainer checks English on the dev
split while it trains, and its `transcribe` command scores the final model on the test splits
(docs/training.md). Stdlib only.
"""
import argparse
import json
import os
import sys

from make_replay import FLEURS_CODES, SOURCES, read_fleurs, without_glosses

SPLITS = ("test", "dev")
# The two-letter language in the file's name, as in fleurs_en_test.jsonl.
SHORT_NAMES = {"en_us": "en", "cmn_hans_cn": "zh", "es_419": "es", "fr_fr": "fr", "de_de": "de"}


def records(fleurs, split="test", code="en_us"):
    """The split's recordings, in the transcript file's order."""
    source = SOURCES[code]
    folder = os.path.join(fleurs, "data", code)
    found = []
    for file, transcript in read_fleurs(os.path.join(folder, f"{split}.tsv")):
        reference = without_glosses(transcript) if source.glossed else transcript
        found.append({"audio": os.path.join(folder, "audio", split, file),
                      "text": f"language {source.language}<asr_text>{reference}"})
    return found


def name(code, split):
    """The records' file name, without .jsonl: fleurs_en_test for English's test split."""
    return f"fleurs_{SHORT_NAMES[code]}_{split}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fleurs", required=True, help="the folder fetch_fleurs.py wrote, with the split's audio extracted")
    parser.add_argument("--out", required=True, help="folder for fleurs_<language>_<split>.jsonl")
    parser.add_argument("--language", choices=FLEURS_CODES, default="en_us", help="the FLEURS language (default en_us)")
    parser.add_argument("--split", choices=SPLITS, default="test", help="the FLEURS split (default test)")
    parser.add_argument("--check-audio", action="store_true", help="skip recordings that are missing")
    args = parser.parse_args()

    try:
        found = records(os.path.abspath(args.fleurs), args.split, args.language)
    except (OSError, ValueError) as error:
        sys.exit(f"prepare_fleurs_test: {error}")
    missing = [record for record in found if args.check_audio and not os.path.exists(record["audio"])]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name(args.language, args.split)}.jsonl"), "w", encoding="utf-8") as out:
        for record in found:
            if record not in missing:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{name(args.language, args.split)}: {len(found) - len(missing)}")
    print(f"missing audio: {len(missing)}")


if __name__ == "__main__":
    main()
