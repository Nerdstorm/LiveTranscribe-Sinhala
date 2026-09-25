#!/usr/bin/env python3
r"""Scores the base model's labels for the replay set, and writes data/replay.tsv.

    make_replay.py --fleurs <FLEURS folder> --librispeech <LibriSpeech folder> \
        --labels <labels folder> --out data/replay.tsv [--sources en_us,…,librispeech]

The replay set is speech in languages Qwen3-ASR already knows, labelled by the base model itself,
so training on it alongside Sinhala keeps what the model does in those languages. It has two
sources:

- FLEURS's train splits in English, Chinese, Spanish, French and German (en_us, cmn_hans_cn,
  es_419, fr_fr, de_de), as fetch_fleurs.py lays them out: every utterance.
- LibriSpeech's train-clean-100 (librispeech), as fetch_librispeech.py lays it out: the
  utterances pick_librispeech.py picked.

tools/pseudo-label writes the labels, one <source>.tsv per source in the labels folder, or several
.tsv files in a <source>/ folder when the source was labelled in shards.

Each label is scored against the corpus's own transcript: the word error rate, or for Chinese the
character error rate, with both texts normalised (NFKC, case folded, punctuation dropped, "1,000"
read as "1000"). FLEURS's Chinese transcripts gloss names in brackets,
"加西亚（Christopher Garcia）", and the speakers don't read the glosses, so a Chinese label is
scored against the transcript with and without them, and the closer one counts. Writes one line
per utterance, in source and then file order:

    <source><TAB><file><TAB><error><TAB><label>

prepare_replay.py keeps the lines whose error is at most its --max-error (default 0.3), so the
base model's mistakes aren't taught back to it. Every FLEURS utterance needs a label, and every
label needs a transcript: anything missing stops the run. Stdlib only.
"""
import argparse
import collections
import glob
import math
import os
import re
import statistics
import sys
import unicodedata

Source = collections.namedtuple("Source", "language corpus by_character glossed")

# Source code: the language as Qwen3-ASR names it (support_languages in the model's config.json),
# the corpus, whether errors are counted in characters (a language written without spaces), and
# whether the transcripts carry bracketed glosses that aren't read aloud.
SOURCES = {
    "en_us": Source("English", "fleurs", False, False),
    "cmn_hans_cn": Source("Chinese", "fleurs", True, True),
    "es_419": Source("Spanish", "fleurs", False, False),
    "fr_fr": Source("French", "fleurs", False, False),
    "de_de": Source("German", "fleurs", False, False),
    "librispeech": Source("English", "librispeech", False, False),
}
FLEURS_CODES = [code for code, source in SOURCES.items() if source.corpus == "fleurs"]
LIBRISPEECH_SUBSET = "train-clean-100"
DEFAULT_MAX_ERROR = 0.3
LABELS_HEADER = "file\tseconds\tms\ttext"

APOSTROPHES = str.maketrans(
    "\N{RIGHT SINGLE QUOTATION MARK}\N{LEFT SINGLE QUOTATION MARK}\N{MODIFIER LETTER APOSTROPHE}", "'''"
)
# "1,000", "1.000" and "1 000" are one number, and "3,5" matches "3.5".
NUMBER_SEPARATOR = re.compile(r"(?<=\d)(?:[.,]|\s(?=\d{3}(?!\d)))(?=\d)")
BRACKETED = re.compile(r"\s*[（(]([^（）()]*)[）)]")


def normalise(text):
    """NFKC, case folded, one kind of apostrophe, and no separators inside numbers."""
    text = unicodedata.normalize("NFKC", text).casefold().translate(APOSTROPHES)
    return NUMBER_SEPARATOR.sub("", text)


def counts(character):
    """Whether the error rates count a character: letters, marks and digits do."""
    return unicodedata.category(character)[0] in "LMN"


def words(text):
    """The words of a transcript, for the word error rate. An apostrophe inside a word stays part
    of it ("bank's", "l'homme"); other punctuation separates words."""
    spaced = "".join(c if counts(c) or c == "'" else " " for c in normalise(text))
    return [word for word in (w.strip("'") for w in spaced.split()) if word]


def characters(text):
    """The characters of a transcript, for the character error rate: no spaces or punctuation."""
    return [c for c in normalise(text) if counts(c)]


def without_glosses(text):
    """The transcript without bracketed text that has no Chinese characters in it: the original
    spelling after a transliterated name, "加西亚（Christopher Garcia）"."""
    def keep_if_chinese(match):
        chinese = any(unicodedata.name(c, "").startswith("CJK") for c in match.group(1))
        return match.group(0) if chinese else " "
    return " ".join(BRACKETED.sub(keep_if_chinese, text).split())


def edit_distance(reference, hypothesis):
    """The fewest substitutions, deletions and insertions that turn reference into hypothesis."""
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def error_rate(reference, label, by_character):
    """The label's errors per word (or character) of the reference; None if the reference is empty."""
    split = characters if by_character else words
    expected = split(reference)
    if not expected:
        return None
    return edit_distance(expected, split(label)) / len(expected)


def label_error(transcript, label, source):
    """The label's error rate against the corpus's transcript, and for a glossed source against the
    transcript without its glosses too, whichever is lower; None if there's nothing to score."""
    references = [transcript]
    if source.glossed:
        references.append(without_glosses(transcript))
    rates = [rate for rate in (error_rate(reference, label, source.by_character) for reference in references)
             if rate is not None]
    return min(rates) if rates else None


def parse_sources(text, allowed=tuple(SOURCES)):
    codes = [code.strip() for code in text.split(",") if code.strip()]
    unknown = [code for code in codes if code not in allowed]
    if unknown or not codes:
        raise ValueError(f"unknown sources {unknown}: use some of {', '.join(allowed)}")
    return codes


def audio_path(code, file, folders):
    """Where a replay utterance's recording is, given {corpus: the folder its fetch script wrote}."""
    if SOURCES[code].corpus == "fleurs":
        return os.path.join(folders["fleurs"], "data", code, "audio", "train", file)
    speaker, chapter, _ = file.split("-", 2)
    return os.path.join(folders["librispeech"], "LibriSpeech", LIBRISPEECH_SUBSET, speaker, chapter, file)


def read_fleurs(path):
    """[(file, transcript)] from a FLEURS <split>.tsv, whose columns are id, file, transcript,
    normalised transcript, characters, samples and gender."""
    utterances = []
    with open(path, encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 7:
                raise ValueError(f"{path}:{number}: expected 7 columns, got {len(fields)}")
            utterances.append((fields[1], fields[2]))
    return utterances


def read_librispeech(folder):
    """[(file, transcript)] for LibriSpeech's train-clean-100 in folder, in file order, from its
    <speaker>-<chapter>.trans.txt files ("<utterance> TRANSCRIPT")."""
    root = os.path.join(folder, "LibriSpeech", LIBRISPEECH_SUBSET)
    paths = sorted(glob.glob(os.path.join(root, "*", "*", "*.trans.txt")))
    if not paths:
        raise ValueError(f"no LibriSpeech transcripts in {root}; run fetch_librispeech.py --extract")
    utterances = []
    for path in paths:
        with open(path, encoding="utf-8") as lines:
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                utterance, _, transcript = line.rstrip("\n").partition(" ")
                if not transcript:
                    raise ValueError(f"{path}:{number}: expected an utterance and its transcript")
                utterances.append((f"{utterance}.flac", transcript))
    return sorted(utterances)


def read_label_file(path, labels):
    """Adds {file: (seconds, label)} from a labels file tools/pseudo-label finished."""
    with open(path, encoding="utf-8") as lines:
        header = next(lines, "")
        if header.rstrip("\n") != LABELS_HEADER:
            raise ValueError(f"{path}: expected the header {LABELS_HEADER!r}, got {header.rstrip()!r}")
        for number, line in enumerate(lines, 2):
            if not line.endswith("\n"):
                raise ValueError(f"{path}:{number}: the line is cut off; run pseudo-label again to finish it")
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"{path}:{number}: expected 4 columns, got {len(fields)}")
            file, seconds, _, label = fields
            if file in labels:
                raise ValueError(f"{path}:{number}: {file} is labelled twice")
            try:
                labels[file] = (float(seconds), label)
            except ValueError:
                raise ValueError(f"{path}:{number}: the seconds {seconds!r} aren't a number") from None
    return labels


def read_labels(labels_dir, code):
    """{file: (seconds, label)} for a source: <code>.tsv, or every .tsv in a <code>/ folder."""
    folder = os.path.join(labels_dir, code)
    paths = sorted(glob.glob(os.path.join(folder, "*.tsv"))) if os.path.isdir(folder) else [folder + ".tsv"]
    if not paths:
        raise ValueError(f"{folder} has no labels files")
    labels = {}
    for path in paths:
        read_label_file(path, labels)
    return labels


def score(folders, labels_dir, codes):
    """[(code, file, error, label, seconds)] for the sources, in source and then file order: every
    FLEURS train utterance, and the LibriSpeech utterances that were labelled."""
    rows = []
    for code in codes:
        source = SOURCES[code]
        if source.corpus == "fleurs":
            references = dict(read_fleurs(os.path.join(folders["fleurs"], "data", code, "train.tsv")))
        else:
            references = dict(read_librispeech(folders["librispeech"]))
        labels = read_labels(labels_dir, code)
        unknown = sorted(labels.keys() - references.keys())
        if unknown:
            raise ValueError(f"{code}: {len(unknown)} labels are for recordings without a transcript, "
                             f"such as {unknown[:3]}")
        missing = sorted(references.keys() - labels.keys())
        if source.corpus == "fleurs" and missing:
            raise ValueError(f"{code}: {len(missing)} utterances have no label, such as {missing[:3]}; "
                             "run pseudo-label again to finish them")
        for file in sorted(labels):
            seconds, label = labels[file]
            error = label_error(references[file], label, source)
            if error is None:
                raise ValueError(f"{code}: {file} has an empty transcript to score its label against")
            rows.append((code, file, error, label, seconds))
    return rows


def write_replay(path, rows):
    with open(path, "w", encoding="utf-8") as out:
        for code, file, error, label, _ in rows:
            out.write(f"{code}\t{file}\t{error:.3f}\t{label}\n")


def read_replay(path):
    """[(code, file, error, label)] from data/replay.tsv."""
    rows = []
    with open(path, encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"{path}:{number}: expected 4 columns, got {len(fields)}")
            code, file, error, label = fields
            if code not in SOURCES:
                raise ValueError(f"{path}:{number}: unknown source {code!r}")
            try:
                value = float(error)
            except ValueError:
                value = math.nan
            # NaN would pass every --max-error cut, so only a real rate is accepted.
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{path}:{number}: the error {error!r} isn't a rate of 0 or more")
            rows.append((code, file, value, label))
    return rows


def summary(rows, max_error=DEFAULT_MAX_ERROR):
    """One line per source and a total: utterances, hours, how many are kept at max_error, and the
    median error."""
    lines = [f"{'source':12} {'utterances':>10} {'hours':>6} {'kept':>6} {'kept %':>7} {'kept h':>7} {'median':>7}"]
    groups = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row[0], []).append(row)
    groups["total"] = rows
    for name, group in groups.items():
        kept = [row for row in group if row[2] <= max_error]
        hours = sum(row[4] for row in group) / 3600
        kept_hours = sum(row[4] for row in kept) / 3600
        median = statistics.median(row[2] for row in group) if group else 0
        share = 100 * len(kept) / len(group) if group else 0
        lines.append(f"{name:12} {len(group):>10} {hours:>6.1f} {len(kept):>6} {share:>6.1f}% {kept_hours:>7.1f} {median:>7.3f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fleurs", help="the folder fetch_fleurs.py wrote")
    parser.add_argument("--librispeech", help="the folder fetch_librispeech.py wrote")
    parser.add_argument("--labels", required=True, help="the folder of labels files tools/pseudo-label wrote")
    parser.add_argument("--out", required=True, help="the table to write, data/replay.tsv")
    parser.add_argument("--sources", default=",".join(SOURCES), help="sources, separated by commas (default: all six)")
    args = parser.parse_args()

    try:
        codes = parse_sources(args.sources)
        folders = {"fleurs": args.fleurs, "librispeech": args.librispeech}
        needed = sorted({SOURCES[code].corpus for code in codes if not folders[SOURCES[code].corpus]})
        if needed:
            raise ValueError(f"give --{' and --'.join(needed)} for the sources {', '.join(codes)}")
        rows = score(folders, args.labels, codes)
    except (OSError, ValueError) as error:
        sys.exit(f"make_replay: {error}")
    write_replay(args.out, rows)
    print(summary(rows))


if __name__ == "__main__":
    main()
