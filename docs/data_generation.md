# Synthetic Data Generation

How labeled training conversations are produced. Code lives in `src/modeling/data/`.

## The idea

An LLM writes realistic chat conversations, but **never the contact details themselves**.
Where a phone number or handle would go, it writes a typed blank (a *slot*), and labels every
message. Code later fills the slots with fake values, several times per conversation, and
optionally disguises them. This gives realistic chat from the LLM, plus variety and exact
control over values, obfuscation, and tags from code.

## Pipeline

```
 1. spec           2. LLM call            3. validate           4. fill (x3)            5. rows
 ───────────       ────────────           ───────────           ──────────────          ──────────
 "group chat,  ──> claude -p writes  ──>  parse JSON, check ──> replace slots with ──>  one model
  4 speakers,      20 conversations       labels, slots, no     fake values; some       input per
  phone/split"     with slots + labels    leaked real values    disguised               message
 generate.py       llm_backend.py         generate.py           fill.py                 assembly/
```

1. **Spec.** For each conversation, `generate.py` samples what it should look like: DM or
   group, speaker count, length, game genre, chat style, and a *focus*. A focus is either a
   positive to include (e.g. "a phone number split across messages") or a look-alike negative
   (e.g. "players share lobby codes"). Positive conversations get 2-3 separate exchanges,
   so positives aren't too sparse. Specs are sampled by code from fixed lists, seeded per
   batch: different batches differ, and rerunning a batch reproduces it. One batch = 20 specs.
2. **LLM call.** The specs, slot rules, and the full labeling guidelines are inserted into
   `prompts/skeleton_v1.txt`. `llm_backend.py` sends it to `claude -p` (Sonnet, low effort).
   The response is 20 conversations, each message with text and a label, e.g.
   `{"speaker": "B", "text": "text me {PHONE}", "label": 1, "categories": ["phone"], ...}`.
3. **Validate.** Each conversation is checked: valid JSON, valid labels and tags, known slot
   kinds and platforms, and no real-looking phone numbers, emails, or links written outside a
   slot. Failing conversations are dropped (the reason is logged). Some tag mistakes are
   fixed by rule instead, e.g. a message with a complete phone number can't be
   `cross_message`. Survivors are **skeletons**.
4. **Fill.** `fill.py` turns each skeleton into 3 concrete conversations by sampling slot
   values (`primitives/values.py`). Fill 0 keeps values plain except slots the LLM marked `!`;
   fills 1-2 also disguise each contact value with 20% probability (`primitives/obfuscate.py`) and update the
   tags to `obfuscated`. All fills share the skeleton's id, so they always land in the same
   train/val/test split.
5. **Assembly** (`assembly/build.py`, below). Fills, real CONDA chat, and spliced
   conversations are split into train/val/test, and every message becomes one model input via
   the context builder: earlier messages as context, the message as target.

**Where labels come from:** the LLM labels each message in step 2, because it knows what it
meant each message to be. Code corrects tags only by fixed rules (step 3) and sets the
`obfuscated` tag when it disguises a value (step 4). A sample is reviewed by hand; there is no
second LLM labeling pass.

## Files

| File | Role | Run it? |
|---|---|---|
| `skeletons/generate.py` | Steps 1-3: specs, prompt, LLM call, validation, saving | **Entry point** |
| `skeletons/llm_backend.py` | Runs `claude -p` with tools, MCP, and sessions off. Knows nothing about skeletons | Library |
| `skeletons/prompts/skeleton_v1.txt` | The prompt template | — |
| `skeletons/fill.py` | Step 4: slot parsing, filling, disguising, tag updates | Library |
| `skeletons/preview.py` | Runs step 4 on generated skeletons; writes the mock dataset and a readable preview | **Entry point** (review) |
| `primitives/values.py` | Fake values for each slot kind | Library |
| `primitives/obfuscate.py` | Disguise transforms and splitting a value across messages | Library |

```
uv run python -m modeling.data.skeletons.generate --out data/pilot --batches 4 --workers 2  # steps 1-3
uv run python -m modeling.data.skeletons.preview  --out data/pilot --fills 3                # step 4 + preview
```

## Output files (under `--out`, gitignored)

| Path | Contents | Used for |
|---|---|---|
| `raw/<batch>.json` | Log of one LLM call: the specs sent, the LLM's response text exactly as returned, model, duration, token counts, rejected conversations and why, tags fixed by rule | Debugging and auditing generation |
| `skeletons/<batch>.jsonl` | The validated conversations from that response, one per line, slots unfilled | **The data**; input to filling |
| `conversations.jsonl` | Every fill of every skeleton (slots filled) | Preview only; the real dataset comes from assembly (below) |
| `preview.md` | Each skeleton, its fills, and the model's view of one fill | Human review |

`raw/` holds the full response too, not just the specs; `skeletons/` is the parsed, validated
subset of it. A batch whose skeleton file already exists is skipped, so an interrupted run
resumes where it stopped.

## Dataset assembly

`assembly/build.py` turns the skeletons into the train/val/test tables. Deterministic for a seed.

```
uv run python -m modeling.data.assembly.build --skeletons data/skeletons_v1/skeletons --out data/dataset_v1
```

1. **Split, then fill.** Each skeleton is assigned a split by a stable hash of its id
   (65/15/20 train/val/test), then filled 3 times. All fills of a skeleton share its split.
2. **CONDA** (`assembly/conda.py`). Real Dota 2 match chat ([CONDA](https://huggingface.co/datasets/Matrix430/CONDA),
   AFL-3.0; downloaded on first use), all labeled negative. One match = one group
   conversation. CONDA has no player ids, so speakers are assigned at random (adjacent
   utterances always differ). Matches with a URL, email, or 7+ digit run are dropped as
   possible real contact info. Matches are split 40/20/40, weighted to test, which needs
   ≥ 50k negatives.
3. **Splice** (`assembly/splice.py`). In 30% of each split's CONDA matches, the positive part
   of a skeleton from the same split (an extra fill) is inserted at a random point, so
   positives also appear inside real chat. The game topic can clash with the host match.
4. **Rows** (`assembly/rows.py`). One scoring row per message, with token counts from the real
   `bert-base-uncased` tokenizer. Rows with an identical (context, target) within a split are
   de-duplicated (fills repeat every message without a slot); identical inputs with
   conflicting labels would be dropped (there are none).
5. **Audit** (`assembly/audit.py`). Sizing targets (design doc §6.1), slice counts (§6.6),
   leakage (group overlap, exact inputs shared with train, near-duplicate skeletons by word
   5-gram Jaccard), rule checks, and `manual_review.csv`: a stratified sample of about 220 val
   rows with empty reviewer columns. Test is never sampled for review.

| Output (under `--out`) | Contents |
|---|---|
| `conversations/{split}.jsonl` | Every conversation, with per-message labels |
| `rows/{split}.parquet` | `ScoringRow`s: the model's tables |
| `audit.md`, `audit.json` | The audit report |
| `manual_review.csv` | Rows to label by hand; compare with `label` |

Train keeps its natural positive rate (~11%); class balancing happens in the training
sampler, not by duplicating rows.
