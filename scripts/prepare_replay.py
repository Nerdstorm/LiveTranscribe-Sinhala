#!/usr/bin/env python3
r"""Writes replay.jsonl for Qwen's fine-tuning script from data/replay.tsv and the replay audio.

    prepare_replay.py --fleurs <FLEURS folder> --librispeech <LibriSpeech folder> \
        --replay data/replay.tsv --out <dir> [--max-error 0.3] [--check-audio]

Keeps the utterances whose label is at most --max-error from the corpus's own transcript (the
error make_replay.py measured, in words, or characters for Chinese) and writes one line for each,
in the format of Qwen's qwen3_asr_sft.py:

    {"audio": "<the recording>", "text": "language <Language><asr_text><label>"}

The recordings are where fetch_fleurs.py --extract and fetch_librispeech.py --extract put them.
Stdlib only.
"""
import argparse
import collections
import json
import os
import sys

from make_replay import DEFAULT_MAX_ERROR, SOURCES, audio_path, read_replay


def record(code, file, label, folders):
    return {"audio": audio_path(code, file, folders), "text": f"language {SOURCES[code].language}<asr_text>{label}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fleurs", help="the folder fetch_fleurs.py wrote, with the audio extracted")
    parser.add_argument("--librispeech", help="the folder fetch_librispeech.py wrote, extracted")
    parser.add_argument("--replay", required=True, help="data/replay.tsv")
    parser.add_argument("--out", required=True, help="folder for replay.jsonl")
    parser.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR,
                        help=f"the most a kept label may differ from the corpus's transcript (default {DEFAULT_MAX_ERROR})")
    parser.add_argument("--check-audio", action="store_true", help="skip utterances whose recording is missing")
    args = parser.parse_args()
    if args.max_error < 0:
        sys.exit("prepare_replay: --max-error can't be negative")

    try:
        rows = read_replay(args.replay)
    except (OSError, ValueError) as error:
        sys.exit(f"prepare_replay: {error}")
    folders = {corpus: os.path.abspath(folder) for corpus, folder in
               (("fleurs", args.fleurs), ("librispeech", args.librispeech)) if folder}
    needed = sorted({SOURCES[code].corpus for code, *_ in rows} - folders.keys())
    if needed:
        sys.exit(f"prepare_replay: {args.replay} has {' and '.join(needed)} utterances; give --{' and --'.join(needed)}")
    os.makedirs(args.out, exist_ok=True)

    kept = collections.Counter()
    dropped = collections.Counter()
    missing_audio = 0
    with open(os.path.join(args.out, "replay.jsonl"), "w", encoding="utf-8") as out:
        for code, file, error, label in rows:
            if error > args.max_error:
                dropped[code] += 1
                continue
            if args.check_audio and not os.path.exists(audio_path(code, file, folders)):
                missing_audio += 1
                continue
            out.write(json.dumps(record(code, file, label, folders), ensure_ascii=False) + "\n")
            kept[code] += 1

    for code in SOURCES:
        if kept[code] or dropped[code]:
            print(f"{code}: {kept[code]} kept, {dropped[code]} over {args.max_error}")
    print(f"replay: {sum(kept.values())}")
    print(f"missing audio: {missing_audio}")


if __name__ == "__main__":
    main()
