#!/usr/bin/env python3
"""Merges the classifiers' loanword files into one table.

    merge_loanwords.py <vocabulary.tsv> <out dir of words-NN.tsv> <loanwords.tsv>

Each classifier wrote sinhala_word<TAB>category<TAB>english<TAB>suffix for its chunk
(docs/loanwords.md). Checks every line (four columns, a category of E, N or P, an English
spelling, and a word that is in the vocabulary) and writes the table most frequent word first.
A word listed twice keeps its first line. Any line that fails a check is printed and the run
stops without writing, so a classifier's mistake is fixed at its source. Stdlib only.
"""
import argparse
import glob
import os
import sys

CATEGORIES = {"E", "N", "P"}


def read_vocabulary(path):
    counts = {}
    with open(path, encoding="utf-8") as lines:
        for line in lines:
            word, count = line.rstrip("\n").split("\t")
            counts[word] = int(count)
    return counts


def merge(paths, counts):
    """(rows most frequent first, problems as "path:line: reason", duplicates skipped)."""
    rows, problems, duplicates = {}, [], 0
    for path in paths:
        with open(path, encoding="utf-8") as lines:
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) == 3:
                    fields.append("")  # an empty suffix whose tab an editor dropped
                where = f"{path}:{number}"
                if len(fields) != 4:
                    problems.append(f"{where}: expected 4 columns, got {len(fields)}")
                elif fields[1] not in CATEGORIES:
                    problems.append(f"{where}: unknown category {fields[1]!r}")
                elif not fields[2]:
                    problems.append(f"{where}: no English spelling")
                elif fields[0] not in counts:
                    problems.append(f"{where}: {fields[0]!r} is not in the vocabulary")
                elif fields[0] in rows:
                    duplicates += 1
                else:
                    rows[fields[0]] = fields
    ordered = sorted(rows.values(), key=lambda fields: -counts[fields[0]])
    return ordered, problems, duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vocabulary", help="vocabulary.tsv from make_vocabulary.py")
    parser.add_argument("classified", help="the folder of the classifiers' words-NN.tsv files")
    parser.add_argument("out", help="the loanwords.tsv to write")
    args = parser.parse_args()

    counts = read_vocabulary(args.vocabulary)
    paths = sorted(glob.glob(os.path.join(args.classified, "words-*.tsv")))
    if not paths:
        sys.exit(f"merge_loanwords: no words-*.tsv in {args.classified}")
    rows, problems, duplicates = merge(paths, counts)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(f"merge_loanwords: {len(problems)} lines need fixing; nothing written")

    with open(args.out, "w", encoding="utf-8") as out:
        for fields in rows:
            out.write("\t".join(fields) + "\n")
    total = sum(counts.values())
    for category in sorted(CATEGORIES):
        words = [fields[0] for fields in rows if fields[1] == category]
        share = 100 * sum(counts[word] for word in words) / max(1, total)
        print(f"{category}: {len(words)} words, {share:.2f}% of all words")
    print(f"{len(rows)} words from {len(paths)} files, {duplicates} duplicates skipped")


if __name__ == "__main__":
    main()
