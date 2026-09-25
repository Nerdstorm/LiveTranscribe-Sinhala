#!/usr/bin/env python3
r"""Picks the LibriSpeech utterances for the replay set, and lays them out for pseudo-label.

    pick_librispeech.py --librispeech <folder> --out <folder> [--count 16000] [--seed 52] \
        [--shards 1]

Reads the utterances of <folder>/LibriSpeech/train-clean-100 (as fetch_librispeech.py --extract
leaves it), picks --count of them with --seed, and links each picked recording into --out as
<utterance>.flac, so tools/pseudo-label can label the folder. With --shards N the picks are dealt
into subfolders 1 to N of --out, for N pseudo-label processes at once. The same corpus, count and
seed always pick the same utterances. Stdlib only.
"""
import argparse
import glob
import os
import random
import sys

from make_replay import audio_path, read_librispeech

DEFAULT_COUNT = 16000


def pick(files, count, seed):
    """count of the files, chosen with seed, in name order."""
    ordered = sorted(files)
    if not 0 < count <= len(ordered):
        raise ValueError(f"can't pick {count} of {len(ordered)} utterances")
    return sorted(random.Random(seed).sample(ordered, count))


def link(files, librispeech_dir, out, shards):
    """Links each file's recording into out, or dealt into out/1 … out/<shards>, and returns the
    folders. Stops if a folder holds any other recording, so pseudo-label labels only the picks."""
    folders = [out] if shards == 1 else [os.path.join(out, str(number)) for number in range(1, shards + 1)]
    corpus = {"librispeech": os.path.abspath(librispeech_dir)}
    expected = {}
    for index, file in enumerate(files):
        expected[os.path.join(folders[index % shards], file)] = audio_path("librispeech", file, corpus)
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        strays = [path for path in glob.glob(os.path.join(folder, "*.flac")) if path not in expected]
        if strays:
            raise ValueError(f"{folder} has recordings that weren't picked, such as {os.path.basename(strays[0])}; "
                             "give an empty --out")
    for path, recording in expected.items():
        if not os.path.isfile(recording):
            raise ValueError(f"{recording} is missing; run fetch_librispeech.py --extract")
        if os.path.islink(path) and os.readlink(path) == recording:
            continue
        if os.path.lexists(path):
            raise ValueError(f"{path} is in the way; give an empty --out")
        os.symlink(recording, path)
    return folders


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--librispeech", required=True, help="the folder fetch_librispeech.py wrote, extracted")
    parser.add_argument("--out", required=True, help="the folder to link the picked recordings into")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"utterances to pick (default {DEFAULT_COUNT})")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--shards", type=int, default=1, help="subfolders to deal the picks into (default 1: none)")
    args = parser.parse_args()

    try:
        if args.shards < 1:
            raise ValueError("--shards must be at least 1")
        utterances = read_librispeech(args.librispeech)
        picked = pick([file for file, _ in utterances], args.count, args.seed)
        folders = link(picked, args.librispeech, args.out, args.shards)
    except (OSError, ValueError) as error:
        sys.exit(f"pick_librispeech: {error}")
    speakers = {file.split("-", 1)[0] for file in picked}
    print(f"picked {len(picked)} of {len(utterances)} utterances, from {len(speakers)} speakers, into "
          f"{', '.join(folders)}")


if __name__ == "__main__":
    main()
