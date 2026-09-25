# Training

How the fine-tuning run goes, from the OpenSLR zips to an MLX model LiveTranscribe can load.
**It hasn't been run yet.** Steps marked **(confirm)** are untested: check them on the machine
before relying on them.

## What goes in

| Data | Size | Use |
|---|---|---|
| [OpenSLR 52](https://www.openslr.org/52/) (16 zips, about 14.7 GB, CC BY-SA 4.0) | 185,293 utterances, 478 speakers, about 224 h | Sinhala: 172,134 train, 4,411 dev and 8,748 test utterances |
| `data/speaker-split.tsv` (seed 52) | 442 / 12 / 24 speakers | no test voice is heard in training |
| `data/loanwords.tsv` | 1,339 words, 415 of them English (E) | English words written in English letters |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) train splits in English, Chinese, Spanish, French and German (about 7.9 GB, CC BY 4.0), [LibriSpeech](https://www.openslr.org/12/) train-clean-100 (6.4 GB, CC BY 4.0) and `data/replay.tsv` | 30,011 of 30,824 utterances kept, 99.3 h, 14.8% of the mix | replay: keeps the languages the model already knows ([below](#replay-keeping-english)) |
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

- 30,011 utterances pass: 14,059 from FLEURS and 15,952 of the 16,000 picked from LibriSpeech,
  14.8% of the training mix. If English slips in step 5, give the replay a bigger share (repeat
  `replay.jsonl` in the mix) before anything else.
- **The labels come from the app's 8-bit MLX model, not the bf16 weights being trained**
  ([replay.md](replay.md#limits)).
- FLEURS has no Sinhala, so the Sinhala tests are OpenSLR 52's 24 test speakers and your own
  dictations. Common Voice has a little Sinhala (CC0), but downloading it needs an account.

**There's no script yet for scoring a checkpoint (steps 5 and 7).**

## The run

1. **Machine**: one NVIDIA A100 or H100 with 80 GB, Linux, and about 100 GB of disk. Qwen's
   fine-tuning setup uses FlashAttention 2, which needs an NVIDIA GPU, and the script trains in
   bf16 on Ampere (A100) and newer GPUs.
2. **Environment (confirm)**: follow Qwen's
   [fine-tuning README](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning):
   `pip install -U qwen-asr datasets`, then `pip install -U flash-attn --no-build-isolation`.
   Record the exact versions with `pip freeze`.
3. **Data**: copy the 16 zips over and unzip them into one `asr_sinhala/`:

   ```bash
   for zip in asr_sinhala_*.zip; do unzip -q -n "$zip"; done
   python3 scripts/prepare_data.py --data-dir asr_sinhala --split data/speaker-split.tsv \
     --loanwords data/loanwords.tsv --out jsonl --check-audio
   ```

   Every zip holds the same `LICENSE` and `utt_spk_text.tsv`, and `-n` keeps the first copy
   instead of asking. `prepare_data.py` should print train 172134, dev 4411, test 8748 and
   missing audio 0. Read `jsonl/rewrites.tsv`: every rewrite should be one a Sinhala speaker
   would type. The audio is FLAC, which the script reads with librosa **(confirm)**.

   Then the replay set: FLEURS from the Hub (about 7.9 GB), and LibriSpeech (6.4 GB) from a copy
   on the Hub or else OpenSLR's servers. Each file is checked against its published hash, and a
   download that breaks off carries on where it stopped when run again:

   ```bash
   python3 scripts/fetch_fleurs.py --out fleurs --extract
   python3 scripts/fetch_librispeech.py --out librispeech --extract
   python3 scripts/prepare_replay.py --fleurs fleurs --librispeech librispeech \
     --replay data/replay.tsv --out jsonl --check-audio
   ```

   `prepare_replay.py` should print replay 30011 and missing audio 0. FLEURS's audio is
   WAV, 32-bit float at 16 kHz, and LibriSpeech's is FLAC at 16 kHz.
4. **Train (confirm the numbers on a short run first)**:

   ```bash
   cat jsonl/train.jsonl jsonl/replay.jsonl > jsonl/train+replay.jsonl
   python qwen3_asr_sft.py --model_path Qwen/Qwen3-ASR-0.6B \
     --train_file jsonl/train+replay.jsonl --eval_file jsonl/dev.jsonl \
     --output_dir out --batch_size 32 --grad_acc 4 --lr 2e-5 --epochs 3 --save_steps 500
   ```

   The trainer shuffles, so joining the files is enough. 32 × 4 = 128 utterances a step, about
   1,580 steps an epoch with replay. It measures dev loss and saves a checkpoint every
   `--save_steps`, and keeps only the last 5 (`--save_total_limit`); raise that if the disk
   allows, so step 5 can choose from all of them. A new language may want a higher learning rate
   than Qwen's default of 2e-5: try 1e-4 on a 1,000-step run, and keep the one with the lower
   dev loss.
5. **Pick the checkpoint** by dev character error rate, not loss, and check its English on the
   FLEURS English test split against the base model
   (`fetch_fleurs.py --out fleurs --languages en_us --splits test --extract`). None of its 350
   sentences is in the replay set.
6. **Convert (confirm)**: to MLX 8-bit with mlx-audio's converter, and add `"Sinhala"` to
   `support_languages` in `config.json`.
7. **Evaluate on a Mac**:
   - the OpenSLR test speakers: CER and WER, and how many English words came out in English
     letters;
   - your own dictations;
   - in LiveTranscribe, `make eval ARGS="--multiline --stt-model <the new model>"`: English must
     stay at the base model's numbers.

## What decides success

- Sinhala CER on the test speakers well under the 30% Omnilingual ASR had when held to Sinhala
  script, and every word in Sinhala script or English letters, never Bengali or Telugu.
- The English eval unchanged, within run-to-run noise.
- Speed. Qwen's tokenizer needs 8.2 tokens per second of Sinhala speech, against 3.6 for English,
  and the model writes one token at a time, so the same length of speech takes about 2.3 times
  as many steps. Measure it on your own clips against dictation's 1.2 s target.

If accuracy or speed falls short, add Sinhala tokens to the tokenizer before training (new merges,
with the embeddings and output layer resized). That's a standard step, but it complicates the MLX
conversion.

## What the data can't show

English words are rare in OpenSLR 52: 2% of training sentences have one, mostly blog words such
as "blog", "film", "link" and "comment". Dictation mixes in far more ("meeting එක cancel කරන්න").
Only your own recordings can show whether the model carries the pattern over. If it doesn't, the
fix is code-switched training recordings, not more OpenSLR.
