# Data

| File | Contents |
|---|---|
| `speaker-split.tsv` | `speaker<TAB>train`, `dev` or `test` for OpenSLR 52's 478 speakers: 442 train (172,134 utterances), 12 dev (4,411) and 24 test (8,748). Written by `scripts/make_split.py` with seed 52. |
| `loanwords.tsv` | `sinhala_word<TAB>category<TAB>english<TAB>suffix` for 1,339 words in OpenSLR 52's transcripts, most frequent first: English words (E), English loans Sinhala has made its own (N) and foreign names (P). See [docs/loanwords.md](../docs/loanwords.md). |

## Licence and attribution

Both files are derived from the **Large Sinhala ASR training data set** (OpenSLR 52,
<https://www.openslr.org/52/>), Copyright 2016, 2017, 2018 Google, Inc., licensed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0).

Changes from the original:

- `speaker-split.tsv` assigns the corpus's anonymised speaker IDs to train, dev and test.
- `loanwords.tsv` lists words from the corpus's transcripts, each with a category, an English
  spelling and its Sinhala ending added.

These files are licensed under CC BY-SA 4.0 too, and are provided as is, without warranties of
any kind (section 5 of the licence). The code in this repository is under the MIT licence
([LICENSE](../LICENSE)).

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
