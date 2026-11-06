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
                                                                                        (Step 4)
 generate.py       llm_backend.py         generate.py           fill.py                 context.py
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
5. **Scoring rows** (Step 4, not built yet). Every message of every filled conversation becomes
   one model input via the context builder: earlier messages as context, the message as target.

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
| `conversations.jsonl` | Every fill of every skeleton (slots filled) | Mock dataset; becomes the dataset in Step 4 |
| `preview.md` | Each skeleton, its fills, and the model's view of one fill | Human review |

`raw/` holds the full response too, not just the specs; `skeletons/` is the parsed, validated
subset of it. A batch whose skeleton file already exists is skipped, so an interrupted run
resumes where it stopped.
