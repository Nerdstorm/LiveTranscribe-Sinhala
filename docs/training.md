# Training

How the fine-tuning run goes, from the OpenSLR zips to an MLX model LiveTranscribe can load.
The first model was trained this way on an M4 Pro Mac, and its test results are in
[the run](#the-run), step 10.

## What goes in

| Data | Size | Use |
|---|---|---|
| [OpenSLR 52](https://www.openslr.org/52/) (16 zips, about 14.7 GB, CC BY-SA 4.0) | 185,293 utterances, 478 speakers, about 224 h | Sinhala: 172,134 train, 4,411 dev and 8,748 test utterances |
| `data/speaker-split.tsv` (seed 52) | 442 / 12 / 24 speakers | no test voice is heard in training (sentences can be: [the run](#the-run), step 1) |
| `data/loanwords.tsv` | 1,339 words, 415 of them English (E) | English words written in English letters |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) train splits in English, Chinese, Spanish, French and German (about 7.9 GB, CC BY 4.0), [LibriSpeech](https://www.openslr.org/12/) train-clean-100 (6.4 GB, CC BY 4.0) and `data/replay.tsv` | 30,128 of 30,824 utterances kept, 99.6 h, 14.9% of the mix | replay: keeps the languages the model already knows ([below](#replay-keeping-english)) |
| Your own dictations | 10–20 clips | the real test: Sinhala as the app will hear it |

`speaker-split.tsv` and `loanwords.tsv` were made from the `utt_spk_text.tsv` that every zip
holds, SHA-256 `471a851decc677d2a70481ef5de75ce73a6b75009a2e3c5175f42131b52fa1de`.

## Replay: keeping English

Training on Sinhala alone would wear down what the model already does, and LiveTranscribe's
English eval (1.7% WER against what was meant, at the Medium cleanup level) must not get worse.
So each epoch mixes in speech in English, Chinese, Spanish, French and German (FLEURS, with
LibriSpeech for more English), labelled **by the base model itself**: the targets are then
exactly its current behaviour, casing and punctuation included, rather than another corpus's
style. A label more than 30% from the corpus's own transcript is left out, so the base model's
mistakes aren't taught back to it. [replay.md](replay.md) has how the labels were made, the
numbers and the limits.

- 30,128 utterances pass: 14,166 from FLEURS and 15,962 of the 16,000 picked from LibriSpeech,
  14.9% of the training mix once. English slipped at that share, so the run passes
  `replay.jsonl` twice, 26% of the mix ([the run](#the-run), step 6). If English slips again,
  give the replay a bigger share still before anything else.
- The labels come from the app's 8-bit MLX model, which is also where training starts
  ([replay.md](replay.md#limits)).
- FLEURS has no Sinhala, so the Sinhala tests are OpenSLR 52's 24 test speakers and your own
  dictations. Common Voice has a little Sinhala (CC0), but downloading it needs an account.

## The run

The fine-tune runs on a Mac with `tools/trainer`, written in Swift on mlx-audio-swift, the library
LiveTranscribe runs Qwen3-ASR with. So training sees the prompt, audio features and weights the app
uses, and nothing has to be converted afterwards. It was built and run on an M4 Pro with 48 GB.

**The trainer pins mlx-audio-swift `01dec7c`, past the app's v0.1.3, for its corrected audio
features ([below](#the-audio-features)). The app has to move to the same revision to run the model
as it was trained.**

Before a long run: plug the Mac in, and don't run LiveTranscribe's `make eval` or `make bench` at
the same time, as they'd compete for the GPU. Training takes about 35 GB of memory, so on a 48 GB
Mac close what else holds a lot (a Linux VM, several editors and browsers): while macOS swaps,
steps take about half as long again. Leave about 50 GB of disk free: two saved states of
9.4 GB each (fp32 weights and the optimizer's two moments), a third while one is written, and a
1.6 GB snapshot at each evaluation.

1. **Data**: unzip the 16 OpenSLR zips into one `asr_sinhala/`, fetch the replay corpora and
   FLEURS's English test and dev splits, and write `jsonl/`:

   ```bash
   for zip in asr_sinhala_*.zip; do unzip -q -n "$zip"; done
   python3 scripts/prepare_data.py --data-dir asr_sinhala --split data/speaker-split.tsv \
     --loanwords data/loanwords.tsv --out jsonl --check-audio
   python3 scripts/fetch_fleurs.py --out fleurs --extract
   python3 scripts/fetch_librispeech.py --out librispeech --extract
   python3 scripts/prepare_replay.py --fleurs fleurs --librispeech librispeech \
     --replay data/replay.tsv --out jsonl --check-audio
   python3 scripts/fetch_fleurs.py --out fleurs --languages en_us --splits test,dev --extract
   python3 scripts/prepare_english_test.py --fleurs fleurs --out jsonl --check-audio
   python3 scripts/prepare_english_test.py --fleurs fleurs --out jsonl --split dev --check-audio
   ```

   Every zip holds the same `LICENSE` and `utt_spk_text.tsv`, and `-n` keeps the first copy
   instead of asking. The scripts should print train 172134, dev 4411 and test 8748, dev_new
   1242 and test_new 2547, replay 30128, fleurs_en_test 647, fleurs_en_dev 394, and missing
   audio 0 each time. Training watches English on the dev split; the test split is kept for the
   end.

   `dev_new.jsonl` and `test_new.jsonl` are the dev and test recordings whose sentence isn't in
   `train.jsonl`. The split keeps voices out of training, not sentences: OpenSLR's speakers read
   from a shared pool, and 71% of test recordings read a sentence someone in train also reads, so
   the full sets partly measure sentences the model has learnt to write. The first model was
   trained on these files as they are. **For the next one, add `--hold-out-sentences`**: it leaves
   out of train every recording of a dev or test sentence (15,850, so train is 156284), and dev and
   test are then new in voice and sentence alike. Read `jsonl/rewrites.tsv`: every
   rewrite should be one a Sinhala speaker would type. FLEURS is about 7.9 GB and LibriSpeech
   6.4 GB; each file is checked against its published hash, and a download that breaks off carries
   on where it stopped when run again. The JSONL files point at the audio where it is, so leave it
   there. The trainer reads OpenSLR's and LibriSpeech's FLAC and FLEURS's 32-bit float WAV itself.

2. **Build**:

   ```bash
   (cd tools/trainer && xcodebuild build -scheme Trainer -configuration Release \
     -destination platform=macOS,arch=arm64 -derivedDataPath .build/xcode \
     -skipPackagePluginValidation CLANG_COVERAGE_MAPPING=NO)
   T=tools/trainer/.build/xcode/Build/Products/Release/trainer
   ```

   Run it from the repository root, where `jsonl/` is. It downloads
   `mlx-community/Qwen3-ASR-0.6B-8bit` into mlx-audio's cache the first time, as the app does.

3. **Check** that training computes what the app computes:
   `$T check --english jsonl/fleurs_en_test.jsonl`. Every line must say PASS:
   - the prompt and label tokenise as the whole text does;
   - each clip gets the app's count of audio placeholders and features;
   - the trainer's batched encoder gives each clip what the app's encoder gives it alone;
   - a batch's loss is the sum of its examples';
   - teacher forcing on replay labels picks the base model's own tokens (all 561 of them here);
   - gradients reach every tensor of the encoder and the decoder;
   - the fp32 copy being trained transcribes like the app's 8-bit model;
   - computing in bfloat16 gives float32's gradients, to bfloat16's precision.

4. **Overfit**: `$T overfit` trains on 32 utterances for 100 steps. The loss must fall under 0.05,
   and the 8 it then transcribes from their audio alone must come out right (CER under 5%).

5. **Bench**: `$T bench` times optimizer steps of 128 utterances on the real data mix. On the
   M4 Pro with 48 GB:

   | Compute | Token budget | Encoder | Seconds a step | Epoch | Peak memory |
   |---|---:|---|---:|---:|---:|
   | float32 | 8192 | trained | swaps | | 56.7 GB |
   | float32 | 4096 | trained | 21.4 | 9.4 h | 40.2 GB |
   | bfloat16 | 4096 | trained | 18.0 | 7.9 h | 35.2 GB |
   | bfloat16 | 8192 | trained | 69.6, swapping | | 52.0 GB |
   | bfloat16 | 8192 | frozen | 13.4 | 5.9 h | 32.2 GB |

   The defaults are the third row. In bfloat16 the forward and backward passes run in bfloat16,
   and the optimizer still updates float32 weights; `check` compares its gradients with
   float32's. A larger token budget means fewer passes a step, but 8192 doesn't fit in 48 GB. A
   frozen encoder (`--freeze-encoder`) would save a quarter of the time, but only the decoder
   would learn Sinhala's sounds. Peak memory counts the weights, the optimizer's moments and a
   pass's activations. The bench ran with the GPU to itself. The first real run shared the Mac
   with a Linux VM, LiveTranscribe with its model loaded and a browser: for its first hour, while
   macOS swapped them out to make room (and builds ran alongside), steps took about 26 s, and
   then 18–19 s, the bench's speed. An epoch is about 8.5 hours of steps, plus the evaluations.

6. **Learning rate**: two runs with the real run's settings, stopped early:

   ```bash
   caffeinate -i $T train --run out/lr-2e-5 --rate 2e-5 --english jsonl/fleurs_en_dev.jsonl \
     --stop-after 200 >> out/lr-2e-5.log 2>&1
   caffeinate -i $T train --run out/lr-1e-4 --rate 1e-4 --english jsonl/fleurs_en_dev.jsonl \
     --stop-after 200 >> out/lr-1e-4.log 2>&1
   ```

   Both see the same batches in the same order, and in the first epoch every batch is new, so
   their step losses in `metrics.jsonl` compare directly as held-out losses. Each step logs the
   Sinhala and replay losses apart. Keep the rate whose Sinhala loss falls faster without its
   replay loss rising. If the data stays the same, carry that run on: the same command without
   `--stop-after`.

   On the M4 Pro, 2e-5 won both ways. Over the 108 steps both ran (1e-4 was stopped there), its
   Sinhala loss was lower on 74 steps and its replay loss on 105, and by step 108 1e-4's replay
   loss had reached 0.72 against 0.10. But 2e-5 forgets too: exported after 200 steps, it had
   12% of Sinhala letters wrong and English WER of 6.36% on FLEURS's test split, against 4.96%
   for the base model and 4.91% for the untrained weights exported the same way. (That English
   check used the test split; English is now watched on the dev split instead.) So the run
   doubles the replay's share, to 26% of the mix, and starts afresh.

7. **Train**, with the replay passed twice:

   ```bash
   caffeinate -i $T train --run out/sinhala-2e-5-replay2 --rate 2e-5 \
     --train jsonl/train.jsonl --train jsonl/replay.jsonl --train jsonl/replay.jsonl \
     --english jsonl/fleurs_en_dev.jsonl >> out/sinhala-2e-5-replay2.log 2>&1
   ```

   That's 232,390 utterances an epoch, 1,816 steps. Each evaluation logs English WER on FLEURS's
   dev split beside step 0's. It rises even so: a quarter of the way through, 7.19% against
   5.40%. That's real forgetting, not a change of style: names and rare words spelled by sound
   ("Uthappa" as "Utapala"), and small words slipping ("safaris are" as "safari is a"). More
   replay only slows it in proportion, so the run carries on and step 9 blends it back.

   Ctrl-C (or `kill`) stops it after the current step and saves. Run the same command to carry
   on; a crash or a reboot loses at most the last 30 minutes. The run folder holds:
   - `run.json`, the settings. To carry on, the model, data, optimizer, schedule and precision
     must match; the token budget, `--cache-limit-mb` and the saving and evaluation options may
     change;
   - `state-<step>/`, the last two saved states;
   - `snapshots/step-<N>/`, bf16 weights with `dev.tsv` and `eval.json`, four times an epoch,
     and with `--english` the English dev transcripts, `english.tsv` and `english.json`;
   - `english-step-0/`, the untrained model's English dev transcripts, measured before the
     first step;
   - `metrics.jsonl`, every step and every evaluation, with English WER and CER beside dev
     CER.

8. **Pick the snapshot and the blend.** `export --blend <share>` keeps that share of each
   fine-tuned weight and takes the rest from the base model (weight-space ensembling, as in
   WiSE-FT). Moving back towards the base model gives English back faster than it takes Sinhala
   away, until Sinhala falls off a cliff. The step-454 snapshot, exported at 8-bit, on 500
   Sinhala dev recordings and FLEURS English dev (the base model scores 5.41%):

   | Fine-tuned share | Sinhala CER | English WER |
   |---|---:|---:|
   | 1.0 | 9.32% | 7.22% |
   | 0.75 | 9.93% | 6.12% |
   | 0.65 | 11.40% | 5.88% |
   | 0.5 | 56.62% | 5.70% |

   Export the last snapshots at a few shares, transcribe both dev sets (step 10's commands, with
   `out/eval/dev-sample500.jsonl` and `jsonl/fleurs_en_dev.jsonl`; next time, `jsonl/dev_new.jsonl`
   in place of the sample, so the Sinhala is new sentences too), and take the lowest Sinhala
   CER whose English WER is within half a point of the base model's, with room to spare: the
   English dev set is 394 recordings, so a few tenths of a point is noise, and the test sets get
   one look. At the end of the first epoch, 0.9 gave 5.74% and 6.03%, 0.85 gave 6.01% and 5.88%,
   and 0.8 gave 6.10% and 5.69%; 0.8 was chosen. Blending also stopped a repetition loop the
   unblended model fell into on one English recording. `--blend-audio` gives the audio encoder its
   own share, but splitting it never beat an even blend.

9. **Export** it as a model folder the app loads, quantised like the base model (text model
   8-bit, audio encoder bf16) and with `Sinhala` in `support_languages`:

   ```bash
   $T export --weights out/<name>/snapshots/step-<N> --blend <share> \
     --out out/export/Qwen3-ASR-0.6B-Sinhala-8bit --check jsonl/test.jsonl
   ```

10. **Evaluate** the export as the app runs it, with no language given:

    ```bash
    $T transcribe --model out/export/Qwen3-ASR-0.6B-Sinhala-8bit --records jsonl/test.jsonl --out out/eval/test.tsv
    $T transcribe --model out/export/Qwen3-ASR-0.6B-Sinhala-8bit --records jsonl/test_new.jsonl --out out/eval/test-new.tsv
    $T transcribe --model out/export/Qwen3-ASR-0.6B-Sinhala-8bit --records jsonl/fleurs_en_test.jsonl --out out/eval/english.tsv
    $T transcribe --model mlx-community/Qwen3-ASR-0.6B-8bit --records jsonl/fleurs_en_test.jsonl --out out/eval/english-base.tsv
    ```

    Each writes a TSV of transcripts and a JSON report beside it: CER, WER, how many English words
    came out in English letters, truncations and speed.

    The first model (the step-1816 snapshot at blend 0.8) scored:
    - Sinhala: CER 6.36% and WER 27.47% on all 8,748 test recordings.
    - **On the 2,547 recordings whose sentence isn't in the training data (`test_new.jsonl`):
      7.08% and 30.10%.**
      The split keeps the test speakers' voices out of training but not their sentences. OpenSLR's
      speakers read from a shared pool, and 71% of the test recordings (72% of dev) read a sentence
      that someone in train also reads. Those score 6.06%. Dictation is all new sentences, so the
      7.08% is the number to expect.
    - Every recording was recognised as Sinhala, and 730 of 889 English words (82%) came out in
      English letters.
    - English: FLEURS WER 5.24%, against the base model's 4.96%.
    - No transcript was cut short or ran away.

    Then your own dictations, and
    LiveTranscribe's own English eval on mlx-audio-swift `01dec7c`. Its Bench, like the app, loads
    models by Hugging Face repo id, and uses one it finds in mlx-audio's cache without asking the
    Hub. So to try an export before publishing it, clone it there under the id it will have
    (`cp -c` takes no extra space on APFS):

    ```bash
    D=~/.cache/huggingface/hub/mlx-audio/Nerdstorm_Qwen3-ASR-0.6B-Sinhala-8bit
    mkdir -p $D && cp -c out/export/Qwen3-ASR-0.6B-Sinhala-8bit/* $D/
    ```

    Nothing checks that copy against the Hub again, so move the folder to the Trash before trying
    the published model.

## The audio features

The app's mlx-audio-swift (v0.1.3) computes Qwen3-ASR's log-mel spectrogram with an HTK mel scale
and a symmetric Hann window. Qwen trained the model on `WhisperFeatureExtractor`'s features, which
use the Slaney scale and a periodic window. Upstream fixed this in
[#247](https://github.com/Blaizzy/mlx-audio-swift/pull/247), after its last tag. The base model on
FLEURS's English test split (647 recordings), as the app runs it:

| Word error rate | v0.1.3 | `01dec7c` |
|---|---:|---:|
| Clean | 5.46% | 4.96% |
| White noise at 10 dB | 16.38% | 9.78% |

A model learns whatever features it's trained on. So the trainer and `pseudo-label` pin `01dec7c`
(upstream main on 2026-09-18), and the replay set is labelled with it. **The app must move to
`01dec7c`, or a later revision with the same Qwen3-ASR front end, before it ships the fine-tuned
model.** The move is worth making for English alone.

The trainer takes its features from the model's own `preprocessAudio`, so it follows whatever
revision it's built against. The one thing it copies rather than calls is the audio placeholder
count. mlx-audio-swift works out `(frames / 100) × 13` in float32, where Qwen uses whole seconds,
so some clips get more `<|audio_pad|>` placeholders than the encoder fills. The trainer
reproduces that count (`AudioFeatures.audioTokens`). If a new revision counts differently,
preparing any example stops with an error that says so.

## What decides success

- Sinhala CER on the test speakers well under the 30% Omnilingual ASR had when held to Sinhala
  script, and every word in Sinhala script or English letters, never Bengali or Telugu.
- The English eval unchanged, within run-to-run noise.
- Speed. Qwen's tokenizer needs 8.2 tokens per second of Sinhala speech, against 3.6 for English,
  and the model writes one token at a time, so the same length of speech takes about 2.3 times
  as many steps. Measure it on your own clips against dictation's 1.2 s target.

If accuracy or speed falls short, add Sinhala tokens to the tokenizer before training (new merges,
with the embeddings and output layer resized). That's a standard step, but the trainer doesn't do
it yet, and the app would need the new tokenizer too.

## What the data can't show

English words are rare in OpenSLR 52: 2% of training sentences have one, mostly blog words such
as "blog", "film", "link" and "comment". Dictation mixes in far more ("meeting එක cancel කරන්න").
Only your own recordings can show whether the model carries the pattern over. If it doesn't, the
fix is code-switched training recordings, not more OpenSLR.
