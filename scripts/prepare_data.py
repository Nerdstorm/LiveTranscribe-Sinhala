#!/usr/bin/env python3
r"""Writes Qwen3-ASR fine-tuning files for Sinhala from OpenSLR 52.

    prepare_data.py --data-dir <extracted asr_sinhala> --split data/speaker-split.tsv \
        --loanwords data/loanwords.tsv --out <dir> [--categories E] [--check-audio]

Reads <data-dir>/utt_spk_text.tsv (utterance, speaker, text) and writes train.jsonl, dev.jsonl
and test.jsonl, one line per utterance in the format of Qwen's qwen3_asr_sft.py:

    {"audio": "<data-dir>/data/<first 2 characters>/<utterance>.flac",
     "text": "language Sinhala<asr_text><transcript>"}

The split is by speaker (speaker-split.tsv: speaker<TAB>train|dev|test), so no voice in dev or
test is heard in training. Every speaker must have a split: a missing one stops the run.

Transcripts are cleaned (NFC; zero-width joiners at the edges of a word dropped, the ones inside
conjuncts such as ශ්‍රී kept) and English loanwords are written in English letters, as the app
should write them. loanwords.tsv holds sinhala_word<TAB>category<TAB>english<TAB>suffix, where the
category is E (an English word), N (a loan Sinhala has made its own, such as පොලිසිය) or P (a name).
Only words whose category is in --categories are rewritten. A word without a suffix becomes the
English word; one whose suffix is a particle that stands on its own (එක, වල, ටික, …) becomes the
English word, a space and the particle; any other inflected form stays as it is.

Also writes rewrites.tsv (each rewrite and how often it was made) and prints counts, stdlib only.
"""
import argparse
import collections
import json
import os
import sys
import unicodedata

PREFIX = "language Sinhala<asr_text>"
SPLITS = ("train", "dev", "test")
JOINERS = "\N{ZERO WIDTH NON-JOINER}\N{ZERO WIDTH JOINER}"
# Particles that follow an English word as words of their own: "phone එක", "files වල".
PARTICLES = frozenset({
    "එක", "එකේ", "එකට", "එකෙන්", "එකක්", "එක්ක",
    "වල", "වලට", "වලින්",
    "ටික", "ටිකක්",
})


def clean(text):
    """NFC, with joiners at the edges of each word dropped and spaces collapsed."""
    words = (word.strip(JOINERS) for word in unicodedata.normalize("NFC", text).split())
    return " ".join(word for word in words if word)


def load_split(path):
    split = {}
    with open(path, encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            speaker, name = line.rstrip("\n").split("\t")
            if name not in SPLITS:
                raise ValueError(f"{path}:{number}: unknown split {name!r}")
            split[speaker] = name
    return split


def load_loanwords(path, categories):
    """{cleaned Sinhala word: (English, suffix)} for the chosen categories."""
    table = {}
    with open(path, encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"{path}:{number}: expected 4 columns, got {len(fields)}")
            word, category, english, suffix = fields
            if category not in {"E", "N", "P"}:
                raise ValueError(f"{path}:{number}: unknown category {category!r}")
            if category in categories:
                table[clean(word)] = (english, suffix.strip(JOINERS))
    return table


def rewrite_word(word, table, particles=PARTICLES):
    entry = table.get(word)
    if entry is None:
        return word
    english, suffix = entry
    if not suffix:
        return english
    if suffix in particles:
        return f"{english} {suffix}"
    return word


def rewrite(text, table, particles=PARTICLES):
    return " ".join(rewrite_word(word, table, particles) for word in text.split())


def audio_path(data_dir, utterance):
    return os.path.join(data_dir, "data", utterance[:2], f"{utterance}.flac")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="the extracted asr_sinhala folder")
    parser.add_argument("--split", required=True, help="speaker-split.tsv")
    parser.add_argument("--loanwords", required=True, help="the approved loanword table")
    parser.add_argument("--out", required=True, help="folder for train.jsonl, dev.jsonl and test.jsonl")
    parser.add_argument("--categories", default="E", help="categories to rewrite, such as E or EP (default E)")
    parser.add_argument("--check-audio", action="store_true", help="skip utterances whose FLAC is missing")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    split = load_split(args.split)
    table = load_loanwords(args.loanwords, set(args.categories))
    os.makedirs(args.out, exist_ok=True)

    outputs = {name: open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") for name in SPLITS}
    counts = collections.Counter()
    rewrites = collections.Counter()
    missing_speakers = set()
    try:
        with open(os.path.join(data_dir, "utt_spk_text.tsv"), encoding="utf-8") as lines:
            for line in lines:
                if not line.strip():
                    continue
                utterance, speaker, text = line.rstrip("\n").split("\t", 2)
                name = split.get(speaker)
                if name is None:
                    missing_speakers.add(speaker)
                    continue
                path = audio_path(data_dir, utterance)
                if args.check_audio and not os.path.exists(path):
                    counts["missing audio"] += 1
                    continue
                words = clean(text).split()
                if not words:
                    counts["empty"] += 1
                    continue
                written_words = [rewrite_word(word, table) for word in words]
                for before, after in zip(words, written_words):
                    if before != after:
                        rewrites[(before, after)] += 1
                if written_words != words:
                    counts[f"{name} rewritten"] += 1
                written = " ".join(written_words)
                record = {"audio": path, "text": PREFIX + written}
                outputs[name].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[name] += 1
    finally:
        for output in outputs.values():
            output.close()

    if missing_speakers:
        sys.exit(f"prepare_data: {len(missing_speakers)} speakers have no split, such as "
                 f"{sorted(missing_speakers)[:5]}; add them to {args.split}")

    with open(os.path.join(args.out, "rewrites.tsv"), "w", encoding="utf-8") as report:
        for (before, after), count in rewrites.most_common():
            report.write(f"{before}\t{after}\t{count}\n")
    for key in (*SPLITS, *(f"{name} rewritten" for name in SPLITS), "missing audio", "empty"):
        print(f"{key}: {counts[key]}")
    print(f"distinct rewrites: {len(rewrites)}")


if __name__ == "__main__":
    main()
