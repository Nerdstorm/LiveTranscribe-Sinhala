#!/usr/bin/env python3
"""Counts the Sinhala words in OpenSLR 52's transcripts, for the loanword classification.

    make_vocabulary.py <utt_spk_text.tsv> <out dir> [--min-count 3] [--chunk-size 3000]

Writes, in <out dir>:
- vocabulary.tsv: every Sinhala word and how often it occurs, most frequent first;
- frequent.tsv: the words that occur at least --min-count times;
- chunks/words-NN.tsv: frequent.tsv in pieces of --chunk-size lines, one for each classifier
  (docs/loanwords.md).

A word is a run of Sinhala characters (U+0D80 to U+0DFF) and zero-width joiners, so Latin text
and punctuation are left out. Words with the same count keep the order they first appear in.
Stdlib only.
"""
import argparse
import collections
import os
import re

WORD = re.compile("[" + chr(0x0D80) + "-" + chr(0x0DFF) + chr(0x200D) + "]+")


def count_words(path):
    """[(word, count)], most frequent first."""
    counts = collections.Counter()
    with open(path, encoding="utf-8") as lines:
        for line in lines:
            fields = line.rstrip("\n").split("\t")
            if len(fields) > 2:
                counts.update(WORD.findall(fields[2]))
    return counts.most_common()


def write(path, rows):
    with open(path, "w", encoding="utf-8") as out:
        for word, count in rows:
            out.write(f"{word}\t{count}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcripts", help="OpenSLR 52's utt_spk_text.tsv")
    parser.add_argument("out", help="folder for vocabulary.tsv, frequent.tsv and chunks/")
    parser.add_argument("--min-count", type=int, default=3, help="the fewest occurrences a word needs to be classified")
    parser.add_argument("--chunk-size", type=int, default=3000, help="words per classifier chunk")
    args = parser.parse_args()

    ranked = count_words(args.transcripts)
    frequent = [(word, count) for word, count in ranked if count >= args.min_count]
    chunks = os.path.join(args.out, "chunks")
    os.makedirs(chunks, exist_ok=True)
    write(os.path.join(args.out, "vocabulary.tsv"), ranked)
    write(os.path.join(args.out, "frequent.tsv"), frequent)
    for index, start in enumerate(range(0, len(frequent), args.chunk_size)):
        write(os.path.join(chunks, f"words-{index:02d}.tsv"), frequent[start:start + args.chunk_size])

    total = sum(count for _, count in ranked)
    covered = sum(count for _, count in frequent)
    print(f"{total} words, {len(ranked)} distinct. {len(frequent)} occur at least {args.min_count} times "
          f"and cover {100 * covered / max(1, total):.1f}% of the words.")


if __name__ == "__main__":
    main()
