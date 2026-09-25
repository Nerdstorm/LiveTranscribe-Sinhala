#!/usr/bin/env python3
"""Splits OpenSLR 52's speakers into train, dev and test.

    make_split.py <utt_spk_text.tsv> <speaker-split.tsv> [--seed 52] [--test 24] [--dev 12]

The split is by speaker, so no voice in dev or test is heard in training. The speakers are
sorted, shuffled with --seed, and the first --test go to test and the next --dev to dev; the rest
train. The same transcripts and seed always give the same file (data/speaker-split.tsv was made
this way). Writes speaker<TAB>train|dev|test, sorted by speaker. Stdlib only.
"""
import argparse
import collections
import random


def read_speakers(path):
    """{speaker: utterance count}."""
    speakers = collections.Counter()
    with open(path, encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                speakers[line.rstrip("\n").split("\t")[1]] += 1
    return speakers


def split_speakers(speakers, seed, test, dev):
    """{speaker: "train" | "dev" | "test"}."""
    if test + dev >= len(speakers):
        raise ValueError(f"{test} test and {dev} dev speakers leave none of {len(speakers)} for training")
    order = sorted(speakers)
    random.Random(seed).shuffle(order)
    held_out = {speaker: "test" for speaker in order[:test]}
    held_out.update({speaker: "dev" for speaker in order[test:test + dev]})
    return {speaker: held_out.get(speaker, "train") for speaker in order}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcripts", help="OpenSLR 52's utt_spk_text.tsv")
    parser.add_argument("out", help="the speaker-split.tsv to write")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--test", type=int, default=24, help="speakers held out for testing")
    parser.add_argument("--dev", type=int, default=12, help="speakers held out for choosing checkpoints")
    args = parser.parse_args()

    speakers = read_speakers(args.transcripts)
    split = split_speakers(speakers, args.seed, args.test, args.dev)
    with open(args.out, "w", encoding="utf-8") as out:
        for speaker in sorted(split):
            out.write(f"{speaker}\t{split[speaker]}\n")

    utterances = collections.Counter()
    for speaker, count in speakers.items():
        utterances[split[speaker]] += count
    for name in ("train", "dev", "test"):
        people = sum(1 for value in split.values() if value == name)
        print(f"{name}: {people} speakers, {utterances[name]} utterances")


if __name__ == "__main__":
    main()
