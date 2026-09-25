# The loanword table

`data/loanwords.tsv` lists the English words, English loans and foreign names that OpenSLR 52's
transcripts write in Sinhala letters. `scripts/prepare_data.py` uses it to write the English words
in English letters in the training transcripts.

## Why

People who dictate in Sinhala use English words, and write them in English letters: "meeting එක
cancel කරන්න". OpenSLR 52's transcripts spell every word in Sinhala script, English ones too
(බ්ලොග් for "blog"), so a model trained on them as they are would learn to spell English words in
Sinhala letters.

## How it was made

1. `make_vocabulary.py` found 785,351 words in the transcripts, 64,055 of them distinct. The 29,887
   that occur at least 3 times make up 93.9% of the text. They went into ten chunks of up to 3,000
   words, most frequent first. The rest, each seen once or twice, weren't classified.
2. On 2026-09-25, ten Claude agents (Claude Opus 5.5), one per chunk, classified the words with
   the prompt at the end of this page.
3. `merge_loanwords.py` checked every line the agents wrote (four columns, a known category, an
   English spelling, a word from the vocabulary). All 1,339 passed, with no duplicates, and it
   wrote the table, most frequent word first.

To make the table again, from the extracted corpus:

```bash
python3 scripts/make_vocabulary.py asr_sinhala/utt_spk_text.tsv work/vocabulary
# classify each work/vocabulary/chunks/words-NN.tsv into work/classified/words-NN.tsv
python3 scripts/merge_loanwords.py work/vocabulary/vocabulary.tsv work/classified data/loanwords.tsv
```

Merging is deterministic, classifying isn't, so a new classification gives a slightly different
table.

## Columns and categories

`sinhala_word<TAB>category<TAB>english<TAB>suffix`

- `english`: the English spelling of the English part, in lower case except for names and
  abbreviations (TV, PhD).
- `suffix`: the Sinhala ending attached to the English part, exactly as written in the word:
  ආමිඑක is army + එක, පොලිසිය is police + ිය. It's empty when the whole word is the English
  word.

| Category | What it is | Examples | Words | Share of all words |
|---|---|---|---:|---:|
| E | an English word, as used in code-switched Sinhala | බ්ලොග් blog, ෆිල්ම් film, කමෙන්ට් comment, ලින්ක් link | 415 | 0.54% |
| N | an English loan Sinhala has made its own, normally written in Sinhala letters | බස් bus, පොලිසිය police, ක්‍රිකට් cricket | 354 | 0.49% |
| P | a foreign name: a person, brand, organisation or place outside Sri Lanka | ඉන්දියාව India, අයින්ස්ටයින් Einstein, ෆේස්බුක් Facebook | 570 | 0.50% |

## How prepare_data.py uses it

Only English words (E) are rewritten, the default `--categories E`:

- A bare English word becomes the English word: බ්ලොග් → blog.
- An English word whose ending is a particle that stands on its own (එක, එකේ, එකට, එකෙන්, එකක්,
  එක්ක, වල, වලට, වලින්, ටික, ටිකක්) becomes the English word, a space and the particle:
  ආමිඑක → army එක.
- Any other inflected form stays as it is (කමෙන්ටුවක්, "a comment"): its ending can't follow
  English letters, and rewording it would change what was said.

Loans (N) and names (P) stay in Sinhala letters, as Sinhala writers write them: බස්, පොලිසිය,
ඉන්දියාව.

With the speaker split, this changes 3,364 of the 172,134 training utterances (2.0%), 57 of 4,411
in dev and 148 of 8,748 in test, with 357 distinct rewrites. The most frequent are සබ් → sub (335
times), බ්ලොග් → blog (305), ෆිල්ම් → film (251), කමෙන්ට් → comment (154) and සෙට් → set (104).
`prepare_data.py` writes each rewrite and its count to `rewrites.tsv`: read it before a training
run.

## Limits

- **OpenSLR 52 is read blog and news text, so English words are rare: 2% of training sentences
  have one, mostly blog words. Dictation mixes in far more.** Only recordings of real dictation
  can show whether the model carries the pattern over to heavy code-switching. If it doesn't, the
  fix is code-switched training recordings, not more of this corpus.
- The agents were told to prefer precision over recall, so some English words are missing, and
  nobody has checked every row by hand.
- Some words in the transcripts start with a stray zero-width joiner, so six words are listed
  twice, once with it. Their rows agree, and `prepare_data.py` strips the joiner.

## The classifier prompt

Each agent got this prompt, with its chunk's number in place of NN. The last chunk, which is
shorter, said "about 2,900 lines".

```text
You are classifying Sinhala words from the transcripts of the OpenSLR 52 Sinhala speech corpus, to find English words written in Sinhala script. This is text work only: don't search code, run builds or use the network.

Read the whole file <vocabulary>/chunks/words-NN.tsv (TSV: sinhala_word<TAB>count, about 3,000 lines; read it in parts with offset/limit so you see every line).

For every word that is, or contains, an English word or a foreign (non-Sri-Lankan) proper name written in Sinhala script, output one line:

sinhala_word<TAB>category<TAB>english<TAB>suffix

- category:
  - E: an English word as used in code-switched Sinhala, e.g. මියුසික් music, කමෙන්ට් comment, ටීම් team, ට්‍රයි try, මීටින් meeting, ෆෝන් phone, තෑන්කූ thank you.
  - N: an English loanword long nativised in Sinhala that people normally write in Sinhala script, e.g. පොලිසිය police, බස් bus, කාර් car, හෝටලය hotel, ක්‍රිකට් cricket, මහෝගනී mahogany, ඉස්කෝලය school.
  - P: a foreign proper name (person, brand, organisation or place outside Sri Lanka), e.g. පිටර් Peter, ෆේස්බුක් Facebook.
- english: the English spelling of the English part: lowercase, except proper names (e.g. comment, thank you, wrestling, Peter).
- suffix: any Sinhala ending attached to the English part, exactly as written in the word, e.g. හෝටලේක → hotel + ේක, ටීම්එකට → team + එකට, පොලිසිය → police + ිය. Leave it empty when the whole word is the English word.

Do not list:
- native Sinhala words;
- Pali or Sanskrit-derived words (ධර්මය, බුද්ධ, ප්‍රශ්න, විද්‍යාව);
- Sri Lankan names and places (සනත්, දඹුල්ල, කොළඹ);
- Portuguese, Dutch or Tamil-era loans that are fully Sinhala (කමිසය, මේසය, ජනේලය, පාන්, කෝප්පය);
- words you're unsure about. Prefer precision over recall.

Write the lines, with no header, to <classified>/words-NN.tsv using the Write tool (create it even if empty). Then reply with only: the number of lines per category (E, N, P), and 5 example lines.
```
