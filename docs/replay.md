# The replay set

`data/replay.tsv` holds speech in languages Qwen3-ASR already knows, each recording labelled by the
base model itself. Mixed into the Sinhala training data, it keeps the model doing what it does now
in those languages.

## Why

Fine-tuning on Sinhala alone would wear down the model's other languages, English above all,
which LiveTranscribe is mostly used for. Speech in those languages, trained on alongside Sinhala,
holds them in place. The labels are the base model's own transcripts rather than the corpora's, so
the training targets are its current behaviour, casing, punctuation and way of writing numbers
included, rather than another corpus's conventions.

## What's in it

| Source | Utterances | Hours | Kept | Kept hours | Median error |
|---|---:|---:|---:|---:|---:|
| FLEURS English (`en_us`) | 2,602 | 7.5 | 2,567 (98.7%) | 7.4 | 0.000 |
| FLEURS Chinese (`cmn_hans_cn`) | 3,246 | 9.7 | 3,203 (98.7%) | 9.6 | 0.000 |
| FLEURS Spanish (`es_419`) | 2,796 | 8.8 | 2,258 (80.8%) | 7.0 | 0.061 |
| FLEURS French (`fr_fr`) | 3,193 | 10.3 | 3,118 (97.7%) | 10.1 | 0.038 |
| FLEURS German (`de_de`) | 2,987 | 9.0 | 2,913 (97.5%) | 8.8 | 0.056 |
| LibriSpeech (`librispeech`) | 16,000 | 56.5 | 15,952 (99.7%) | 56.5 | 0.000 |
| **Total** | **30,824** | **101.9** | **30,011 (97.4%)** | **99.3** | **0.018** |

"Kept" is what `prepare_replay.py` keeps: labels at most 0.3 from the corpus's transcript. English
is 18,519 of the 30,011 kept utterances (62%), and 63.8 of the 99.3 hours. The median error is
the label's word error rate against the corpus's transcript (character error rate for Chinese).

- **FLEURS** ([google/fleurs](https://huggingface.co/datasets/google/fleurs), revision `70bb2e8`):
  every train utterance in English (`en_us`), Chinese (`cmn_hans_cn`), Spanish (`es_419`), French
  (`fr_fr`) and German (`de_de`). They are sentences from Wikinews, Wikijunior and Wikivoyage (by
  way of the FLORES benchmark), read aloud. Each language has about 1,500 sentences, each read by
  one to three speakers, most often two. None of the 350 sentences in FLEURS's English test split,
  which step 5 of the [runbook](training.md) checks English on, is in the set.
- **LibriSpeech** ([OpenSLR 12](https://www.openslr.org/12/)), `librispeech`: 16,000 of the 28,539
  utterances of train-clean-100, picked with seed 52 by `pick_librispeech.py`. They are
  audiobooks, read from public domain books. LibriSpeech's test sets, which Qwen3-ASR's English
  results are usually measured on, aren't used.

## How it was made

1. `fetch_fleurs.py --extract` and `fetch_librispeech.py --extract` fetched the corpora (7.9 and
   6.4 GB), checked each file against its published hash (the Hub's SHA-256 or git id, OpenSLR's
   MD5) and extracted them.
2. `pick_librispeech.py --shards 2` picked the LibriSpeech utterances and linked them into two
   folders, to label them in two processes at once.
3. `tools/pseudo-label` labelled every recording with `mlx-community/Qwen3-ASR-0.6B-8bit`
   (revision `89e96d9`), the model LiveTranscribe ships, loaded through mlx-audio-swift at the
   app's revision (`d302a5c`). It tells the model the language, so the prompt ends
   `language English<asr_text>` just as each training line's text begins, and it decodes greedily
   with the model's defaults. On an M4 Pro it took 4.1 hours of the model's time for 101.9 hours
   of audio, 25 times real time for each process; with two processes at once, about two hours
   in all. The labels are repeatable: a sample labelled twice, alone and alongside another
   language, came out the same.
4. `make_replay.py` scored each label against the corpus's transcript and wrote
   `data/replay.tsv`.

To make it again, on a Mac with Apple silicon and Xcode:

```bash
python3 scripts/fetch_fleurs.py --out fleurs --extract
python3 scripts/fetch_librispeech.py --out librispeech --extract
python3 scripts/pick_librispeech.py --librispeech librispeech --out picked --shards 2
(cd tools/pseudo-label && xcodebuild build -scheme PseudoLabel -configuration Release \
  -destination platform=macOS,arch=arm64 -derivedDataPath .build/xcode \
  -skipPackagePluginValidation CLANG_COVERAGE_MAPPING=NO)
label=tools/pseudo-label/.build/xcode/Build/Products/Release/pseudo-label
model=mlx-community/Qwen3-ASR-0.6B-8bit
mkdir -p labels/librispeech
for pair in en_us:English cmn_hans_cn:Chinese es_419:Spanish fr_fr:French de_de:German; do
  "$label" $model "${pair#*:}" "fleurs/data/${pair%%:*}/audio/train" "labels/${pair%%:*}.tsv"
done
for shard in 1 2; do
  "$label" $model English "picked/$shard" "labels/librispeech/$shard.tsv"
done
python3 scripts/make_replay.py --fleurs fleurs --librispeech librispeech --labels labels \
  --out data/replay.tsv
```

`pseudo-label` runs one folder at a time, so the loops can be split across terminals: two
processes at once label faster than one or three. If a run stops, the same command carries on
from where it stopped. `-skipPackagePluginValidation` keeps the build from stopping to ask about
mlx-swift's CUDA build plug-in, which a Mac build doesn't run, and `CLANG_COVERAGE_MAPPING=NO`
keeps Xcode from building the tool with code coverage, which would leave a `default.profraw` in
the folder of every run.

## Scoring

Each label is compared with the corpus's own transcript after both are normalised: Unicode NFKC,
case folded, one kind of apostrophe, punctuation dropped, and no separators inside numbers
("1,000", "1.000" and "1 000" all read as "1000"). English, Spanish, French and German count
words (the word error rate). Chinese, written without spaces, counts characters (the character
error rate). LibriSpeech's transcripts are in capitals without punctuation, which the
normalisation evens out.

FLEURS's Chinese transcripts follow a transliterated name with its original spelling in brackets,
"克里斯托弗·加西亚（Christopher Garcia）", and the speakers don't read it. So a Chinese label is
scored against the transcript both with and without bracketed text that has no Chinese characters
in it, and the closer of the two counts. That decides 138 of the 3,246 Chinese labels: 98.7% of
them are within the cut with the rule, 94.4% without it.

The error is kept in the table, so a stricter cut (`prepare_replay.py --max-error`) needs no new
labels.

## What the cut drops, and what it keeps

The cut drops 813 of the 30,824 labels (2.6%):

- **490 FLEURS Spanish recordings are silent.** 490 of the 2,796 `es_419` train recordings
  (17.5%) hold nothing but zeros, and the other four languages have none. Told the language is
  Spanish, the model writes one word and stops ("El." 439 times, "No." 37, "Puedes." 10); left
  to choose, as the app leaves it, it answers `language None`, no speech, with an empty
  transcript. All 490 are over the cut, so none reaches training.
- **Numbers written another way**: 123 of the other 323 drops. The model sometimes spells out a
  number the transcript writes in digits ("the two thousand and nine season" for "the 2009
  season", "zweitausendzweihundertfünfzig" for "2.250"), and in a short sentence that's enough
  to pass 0.3.
- **Misheard names and rare words**, mostly in short sentences: "Maruko Hideo" for
  Maroochydore, "Guk-Inseln" for the Cook Islands.
- **LibriSpeech's 48** (0.3%) are mostly transcripts in dialect spelling, which the model writes
  in standard English ("And that minds me of an owl" for "AN DAT MINES ME A OWL"), and names or
  Latin phrases.
- Odd ones: 3 Chinese labels in Traditional characters, which count as wrong against a Simplified
  transcript; 1 empty English label; and 3 English labels longer than their transcript and
  mostly wrong (error over 1).

What the cut keeps is the model's ordinary behaviour: in a long sentence, a number spelled out
or a name misheard stays under 0.3.

A label within the cut can still be wrong in up to 30% of its words, and those mistakes are
taught back to the model: they're part of what it does now. A lower `--max-error` keeps fewer of
them, and fewer utterances.

## Limits

- **The labels come from the 8-bit MLX model the app runs, not the bf16 weights that are
  fine-tuned.** Quantisation changes some outputs slightly, so the targets are what the shipped
  model writes, which is what the fine-tuned model must keep doing, but not exactly the trained
  weights' own. Labelling on the training machine with `Qwen/Qwen3-ASR-0.6B` in bf16 would remove
  the difference, with a script for Qwen's own inference package.
- It's all read speech: encyclopedic and news sentences, and audiobooks. Nothing conversational,
  and nothing like a dictated message.

## Columns

`source<TAB>file<TAB>error<TAB>label`

- `source`: `en_us`, `cmn_hans_cn`, `es_419`, `fr_fr` or `de_de` (FLEURS's language codes), or
  `librispeech`.
- `file`: the recording. For FLEURS it's in `fleurs/data/<source>/audio/train/`, and for
  LibriSpeech (`<speaker>-<chapter>-<utterance>.flac`) in
  `librispeech/LibriSpeech/train-clean-100/<speaker>/<chapter>/`, once the fetch scripts have
  run with `--extract`.
- `error`: the label's word error rate against the corpus's transcript (the character error rate
  for Chinese), to three decimals. It passes 1 when a label is longer than the transcript and
  mostly wrong.
- `label`: the base model's transcript, on one line.
