# LiveTranscribe Sinhala

Data and scripts for teaching Sinhala to [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B),
the speech-to-text model in [LiveTranscribe](https://github.com/Nerdstorm/LiveTranscribe), a
dictation app for macOS.

The aim is one model for English, the other languages Qwen3-ASR already knows, and Sinhala.
Sinhala comes out in Sinhala script, with English words in English letters, as people type it:
"meeting එක cancel කරන්න".

**Status: the training data is prepared and tested, but the model hasn't been trained yet.
Training needs a rented NVIDIA GPU ([docs/training.md](docs/training.md)).**

## Why fine-tune

- Qwen3-ASR covers 30 languages, not Sinhala. It takes the language as a prompt
  (`language Sinhala<asr_text>`), punctuates, and copes with mixed languages, so fine-tuning
  can add Sinhala to it.
- Meta's Omnilingual ASR covers Sinhala but has no language input, so it writes most Sinhala
  speech in Bengali or Latin script. Held to Sinhala script, it still had a 30% character error
  rate (CER) on OpenSLR 52 clips it was trained on, and it can't write the English words in
  English letters.

## The pipeline

1. **Loanwords.** `scripts/make_vocabulary.py` counts the words in OpenSLR 52's transcripts and
   splits the frequent ones into chunks. Claude agents classified each chunk, and
   `scripts/merge_loanwords.py` checks their output and writes `data/loanwords.tsv`: 1,339
   English words, loans and foreign names written in Sinhala letters.
   [docs/loanwords.md](docs/loanwords.md) has the method, the prompt and the numbers.
2. **Speaker split.** `scripts/make_split.py` writes `data/speaker-split.tsv`: 442 speakers for
   training, 12 for choosing checkpoints and 24 for testing, so no test voice is heard in
   training.
3. **Replay set.** Speech in languages the model already knows, so fine-tuning doesn't wear them
   down. `scripts/fetch_fleurs.py` fetches FLEURS in English, Chinese, Spanish, French and
   German, `scripts/fetch_librispeech.py` fetches LibriSpeech English, and
   `scripts/pick_librispeech.py` picks 16,000 of its utterances. `tools/pseudo-label`, a Swift
   tool for Macs with Apple silicon, labels them with the base model, and `scripts/make_replay.py`
   scores each label against the corpus's transcript and writes `data/replay.tsv`.
   [docs/replay.md](docs/replay.md) has the method and the numbers.
4. **Training files.** `scripts/prepare_data.py` writes `train.jsonl`, `dev.jsonl` and
   `test.jsonl` for Qwen's fine-tuning script, with the English words rewritten in English
   letters, and `scripts/prepare_replay.py` writes `replay.jsonl`.
5. **Fine-tuning, conversion to MLX and evaluation**: [docs/training.md](docs/training.md).

Steps 1 to 3 have been run and their output is committed, so a training run starts at step 4.
Every script shows its usage with `--help`, and needs only Python 3.9 or later. `pseudo-label`
needs Xcode, and runs only to remake the replay labels.

## Tests

```bash
python3 -m unittest discover -s tests
```

The tests cover every script, and check that the committed files in `data/` are well formed and
that the speaker split is the one `make_split.py` makes. `pseudo-label`'s own tests, for its
labels file, run on a Mac:

```bash
cd tools/pseudo-label
xcodebuild test -scheme PseudoLabel -destination platform=macOS,arch=arm64 -derivedDataPath .build/xcode \
  -skipPackagePluginValidation CLANG_COVERAGE_MAPPING=NO
```

## Using the model in LiveTranscribe

LiveTranscribe runs Qwen3-ASR with MLX (`mlx-community/Qwen3-ASR-0.6B-8bit`). It lets the model
detect the language and removes the `language …<asr_text>` prefix whatever the language, so a
fine-tuned model converted to MLX needs no change to the app's speech-to-text. The app's
`make eval` and `make bench` measure another model with `ARGS="--stt-model <repository>"`.
**How the app's cleanup and spoken commands treat Sinhala text hasn't been tested.**

## Licence

The code is under the MIT licence ([LICENSE](LICENSE)). The files in `data/` are under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/): `speaker-split.tsv` and
`loanwords.tsv` are derived from the [Large Sinhala ASR training data set](https://www.openslr.org/52/)
(OpenSLR 52), Copyright 2016, 2017, 2018 Google, Inc., under CC BY-SA 4.0, and `replay.tsv` from
[FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0), whose speakers read sentences
from FLORES (CC BY-SA 4.0), and the [LibriSpeech ASR corpus](https://www.openslr.org/12/)
(CC BY 4.0). [data/README.md](data/README.md) has the attribution. None of the corpora is
included.
