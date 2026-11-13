# P1a Results

How good the classifier is, in one page. Metric definitions: design doc §6.2. Numbers come
from `python -m modeling.eval.evaluate`, with 95% intervals in brackets. Full report:
[results/p1a/bert_v1/report_val.md](../results/p1a/bert_v1/report_val.md).

**Status:** model (`bert_v1`) and regex baseline measured on **val**. Test is run once, after
the model and threshold are frozen (design doc §6.4).

## Summary

| Requirement | Target | Model (val) | Met? |
|---|---|---|---|
| Q1 recall at FPR ≤ 0.1% | ≥ 0.90 | 0.994 | yes (val) |
| Q2 hard-positive recall | ≥ 0.75 | 0.995 | yes (val) |
| Q3 look-alike FPR | ≤ 0.5% | 0.19% | yes (val) |
| Q4 recall gain over regex, at the regex's FPR | ≥ +0.15 | +0.549 | yes (val) |

**Read this with the first insight below:** val is written by the same generator as train, so
these numbers show the model learned *our data*. They don't yet show it works on real chat.

## Model report (deployed threshold, set on val at FPR ≤ 0.1%)

| Metric | Target | Val | Test |
|---|---|---|---|
| Recall, all positives (primary) | ≥ 0.90 | 0.994 [0.991, 0.996] (4,529 / 4,556) | — |
| FPR, all clean messages | ≤ 0.1% | 0.10% [0.07, 0.14] (36 / 36,905) | — |
| Recall, hard positives (obfuscated + cross-message) | ≥ 0.75 | 0.995 [0.991, 0.997] (1,989 / 1,999) | — |
| FPR, look-alike negatives | ≤ 0.5% | 0.19% [0.09, 0.39] (7 / 3,727) | — |

## Model vs. regex baseline (model moved to the regex's FPR)

| | FPR | Recall, all | Recall, hard |
|---|---|---|---|
| Regex baseline (val) | 1.12% [1.02, 1.23] | 0.450 [0.435, 0.464] | 0.199 [0.182, 0.217] |
| Model at 1.12% FPR (val) | 1.11% [1.01, 1.23] | 0.998 [0.996, 0.999] | 1.000 [0.998, 1.000] |
| Gap | | **+0.549** | **+0.801** |

## Recall by kind of positive (val)

A positive can carry more than one hard tag.

| Positives | Count | Regex | Model |
|---|---|---|---|
| plain | 2,055 | 0.624 | 0.996 |
| obfuscated | 996 | 0.322 | 0.998 |
| cross_message | 1,122 | 0.078 | 0.992 |
| implicit | 509 | 0.721 | 0.984 |

## Insights

**Model**
- **Val is in-distribution, so it's close to a ceiling.** One LLM wrote train and val; splits
  share no conversations but share its style and stock phrasings. Test is built the same way
  and will likely look the same. The real question (does it work on chat the generator didn't
  write?) needs a small hand-written check set.
- **Real game chat is easy; LLM-written chat is where errors are.** FPR is 0.01% on CONDA
  negatives (1 / 8,040) vs. 0.12% on LLM-written ones (29 / 24,523). Positives spliced into
  real chat are caught slightly less often than fully synthetic ones (0.982 vs. 0.995).
- **False blocks come from a few learned shortcuts:**
  - the word *school*: "ngl i gotta bounce soon, school tmrw" scores 0.995 (location asks
    like "what school do u go to" taught it the word);
  - context bleed: a harmless reply after someone *else* shared contact info ("me 😂 hehe"
    right after an email), and refusals ("i dont give out my snap");
  - leetspeak alone: "u d0nt need t0 g0 t0 far in base" is flagged.
- **Misses are mostly polite or plain asks without slang** ("You have discord? Easier to talk
  strategy there.", "how do i reach u outside the game", "anyone want to swap emails") and
  **short answers** ("HyperStorm" right after "whats ur snap", "ok its Bell High").
- **Some errors are label questions, not model errors.** "762728" sent right after "we should
  talk somewhere more private" is labeled negative; "i post mine on insta" is arguably
  self-promotion. The 3 fills of one conversation tend to fail together, so the errors are
  fewer independent cases than the counts suggest.

**Regex baseline**
- **It can't use context.** It reads only the message being sent, so it misses almost every
  cross-message positive (8%): a bare `jake_22` after "whats ur snap" looks like nothing.
- **Disguises mostly beat it.** It catches a third of obfuscated positives, mostly through
  the plain words around the disguised value ("text me …", "add me on snap …").
- **Numbers are its false blocks.** 379 of its 413 false blocks on val are game numbers (match
  IDs, scores) hitting the phone pattern: a 10% FPR on look-alike negatives.
- **Its `implicit` score is inflated.** The generator reused a few phrasings ("gimme ur
  digits", "somewhere more private"), so its slang lists catch 72% of them.

## Setup

| | |
|---|---|
| Dataset | `data/dataset_v1` (val: 41,461 rows, 4,556 positives) — `docs/data_generation.md` |
| Regex baseline | `src/modeling/baselines/regex.py`, target message only |
| Model | `bert-base-uncased`, full fine-tune, best of 3 epochs by val PR AUC (epoch 1: 0.9990; epoch 2: 0.9976), T4, ~16 min/epoch — `src/modeling/notebooks/train_bert.ipynb` |
| Threshold | 0.0817 (val, FPR ≤ 0.1%). Scores are very confident, so the threshold sits low. |
