# Data

| File | Contents |
|---|---|
| `speaker-split.tsv` | `speaker<TAB>train`, `dev` or `test` for OpenSLR 52's 478 speakers: 442 train (172,134 utterances), 12 dev (4,411) and 24 test (8,748). Written by `scripts/make_split.py` with seed 52. |
| `loanwords.tsv` | `sinhala_word<TAB>category<TAB>english<TAB>suffix` for 1,339 words in OpenSLR 52's transcripts, most frequent first: English words (E), English loans Sinhala has made its own (N) and foreign names (P). See [docs/loanwords.md](../docs/loanwords.md). |
| `replay.tsv` | `source<TAB>file<TAB>error<TAB>label` for FLEURS's 14,824 train utterances in English (`en_us`), Chinese (`cmn_hans_cn`), Spanish (`es_419`), French (`fr_fr`) and German (`de_de`), and 16,000 utterances of LibriSpeech's train-clean-100 (`librispeech`): the base Qwen3-ASR model's transcript of each recording, and how far it is from the corpus's own (the word error rate, or the character error rate for Chinese). Written by `scripts/make_replay.py`. See [docs/replay.md](../docs/replay.md). |

## Licence and attribution

### speaker-split.tsv and loanwords.tsv

Both files are derived from the **Large Sinhala ASR training data set** (OpenSLR 52,
<https://www.openslr.org/52/>), Copyright 2016, 2017, 2018 Google, Inc., licensed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0).

Changes from the original:

- `speaker-split.tsv` assigns the corpus's anonymised speaker IDs to train, dev and test.
- `loanwords.tsv` lists words from the corpus's transcripts, each with a category, an English
  spelling and its Sinhala ending added.

The corpus is described in:

> Oddur Kjartansson, Supheakmungkol Sarin, Knot Pipatsrisawat, Martin Jansche and Linne Ha.
> 2018. Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi
> Bengali. In *Proc. The 6th Intl. Workshop on Spoken Language Technologies for
> Under-Resourced Languages (SLTU)*, pages 52–55, Gurugram, India.
> <https://doi.org/10.21437/SLTU.2018-11>

```bibtex
@inproceedings{kjartansson-etal-sltu2018,
  title = {{Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali}},
  author = {Oddur Kjartansson and Supheakmungkol Sarin and Knot Pipatsrisawat and Martin Jansche and Linne Ha},
  booktitle = {Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
  year = {2018},
  address = {Gurugram, India},
  month = aug,
  pages = {52--55},
  URL = {http://dx.doi.org/10.21437/SLTU.2018-11}
}
```

### replay.tsv

`replay.tsv` is derived from two corpora.

**FLEURS** (<https://huggingface.co/datasets/google/fleurs>, revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`) is licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0). Its speakers read sentences from the **FLORES** machine translation benchmark
(<https://github.com/facebookresearch/flores>), licensed under CC BY-SA 4.0.

The **LibriSpeech ASR corpus** (OpenSLR 12, <https://www.openslr.org/12/>), train-clean-100, is
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its recordings are
LibriVox audiobooks of public domain texts from Project Gutenberg.

Changes from the originals: `replay.tsv` holds none of the corpora's recordings or transcripts.
It lists their file names, each with a transcript of the recording made by Qwen3-ASR 0.6B
(Apache-2.0) and that transcript's error rate against the corpus's.

FLEURS is described in:

> Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa,
> Clara Rivera and Ankur Bapna. 2022. FLEURS: Few-shot Learning Evaluation of Universal
> Representations of Speech. arXiv:2205.12446. <https://arxiv.org/abs/2205.12446>

```bibtex
@article{fleurs2022arxiv,
  title = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal = {arXiv preprint arXiv:2205.12446},
  url = {https://arxiv.org/abs/2205.12446},
  year = {2022},
}
```

LibriSpeech is described in:

> Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur. 2015. Librispeech: an ASR
> corpus based on public domain audio books. In *Proc. IEEE International Conference on
> Acoustics, Speech and Signal Processing (ICASSP)*, pages 5206–5210.
> <https://doi.org/10.1109/ICASSP.2015.7178964>

```bibtex
@inproceedings{panayotov2015librispeech,
  title = {Librispeech: an ASR corpus based on public domain audio books},
  author = {Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle = {2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages = {5206--5210},
  year = {2015},
  doi = {10.1109/ICASSP.2015.7178964}
}
```

### All three files

These files are licensed under CC BY-SA 4.0, and are provided as is, without warranties of any
kind (section 5 of the licence). The code in this repository is under the MIT licence
([LICENSE](../LICENSE)).
