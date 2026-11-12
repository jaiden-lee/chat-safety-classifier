# P1a Results

How good the classifier is, in one page. Metric definitions: design doc §6.2. Numbers come
from `python -m modeling.eval.evaluate`, with 95% intervals in brackets.

**Status:** regex baseline measured on **val**. Model not trained yet. Test is run once, after
the model and threshold are frozen (design doc §6.4).

## Summary

| Requirement | Target | Model | Met? |
|---|---|---|---|
| Q1 recall at FPR ≤ 0.1% | ≥ 0.90 | — | — |
| Q2 hard-positive recall | ≥ 0.75 | — | — |
| Q3 look-alike FPR | ≤ 0.5% | — | — |
| Q4 recall gain over regex, at the regex's FPR | ≥ +0.15 | — | — |

## Model report (deployed threshold, set on val at FPR ≤ 0.1%)

| Metric | Target | Val | Test |
|---|---|---|---|
| Recall, all positives (primary) | ≥ 0.90 | — | — |
| FPR, all clean messages | ≤ 0.1% | — | — |
| Recall, hard positives (obfuscated + cross-message) | ≥ 0.75 | — | — |
| FPR, look-alike negatives | ≤ 0.5% | — | — |

## Model vs. regex baseline (model moved to the regex's FPR)

| | FPR | Recall, all | Recall, hard |
|---|---|---|---|
| Regex baseline (val) | 1.12% [1.02, 1.23] | 0.450 [0.435, 0.464] | 0.199 [0.182, 0.217] |
| Model at 1.12% FPR (val) | — | — | — |
| Gap | | — | — |

## Recall by kind of positive (val)

Where each system catches and misses. A positive can carry more than one hard tag.

| Positives | Count | Regex | Model |
|---|---|---|---|
| plain | 2,055 | 0.624 | — |
| obfuscated | 996 | 0.322 | — |
| cross_message | 1,122 | 0.078 | — |
| implicit | 509 | 0.721 | — |

## Insights

- **The regex can't use context.** It reads only the message being sent, so it misses
  almost every cross-message positive (8%): a bare `jake_22` after "whats ur snap" looks
  like nothing. This is where the model should gain the most.
- **Disguises mostly beat the regex.** It catches a third of obfuscated positives, but
  mostly through the plain words *around* the disguised value ("text me …", "my email is
  …", "add me on snap …"), plus some luck: Python's regex treats full-width digits (`３`)
  as digits and Cyrillic letters as letters. When the surrounding words are disguised too
  ("add me on $nap", "c//o//r//d"), it misses.
- **Numbers are the regex's false blocks.** 379 of its 413 false blocks on val are game
  numbers (match IDs, scores) hitting the phone pattern: a 10% FPR on look-alike negatives,
  20× the Q3 limit. The rest are warnings ("dont share ur number") and city mentions.
- **The `implicit` slice is easier than it sounds.** The generator reused a few phrasings
  ("gimme ur digits", "somewhere more private", "the other app, same name as here"), so
  the regex's slang lists catch 72% of them. Expect the model to score high here too;
  it reflects limited variety in the data, not real-world ease.

## Setup

| | |
|---|---|
| Dataset | `data/dataset_v1` (val: 41,461 rows, 4,556 positives) — `docs/data_generation.md` |
| Regex baseline | `src/modeling/baselines/regex.py`, target message only |
| Model | `bert-base-uncased`, full fine-tune, 3 epochs — `src/modeling/notebooks/train_bert.ipynb` |
| Threshold | — |
