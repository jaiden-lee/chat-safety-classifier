"""Build the P1a dataset from skeletons and CONDA. Deterministic for a given seed and inputs.

    uv run python -m modeling.data.assembly.build \
        --skeletons data/skeletons_v1/skeletons --out data/dataset_v1

Steps:
1. Assign each skeleton a split (by skeleton id), then fill it `--fills` times.
2. Load CONDA matches (downloaded on first use) and assign each a split (by match).
3. In each split, splice an exchange from a positive skeleton of that split into
   `--splice-rate` of its CONDA matches (an extra fill, so values differ from the regular fills).
   The other matches stay pure real chat.
4. Expand every conversation into scoring rows (context builder, real BERT token counts) and
   de-duplicate identical model inputs within each split.
5. Audit and write a manual-review sample from val.

Outputs (in --out):
  conversations/{split}.jsonl   every conversation, with per-message labels
  rows/{split}.parquet          scoring rows: the model's train/val/test tables
  audit.md, audit.json          sizing targets, slices, leakage, rule checks
  manual_review.csv             stratified val sample with empty reviewer columns
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from modeling.context import DEFAULT_CONFIG
from modeling.data.assembly.audit import audit, manual_review_sample, to_markdown
from modeling.data.assembly.conda import load_matches
from modeling.data.assembly.rows import (
    DEFAULT_TOKENIZER,
    bert_token_counter,
    conversation_rows,
    dedupe,
)
from modeling.data.assembly.splice import splice
from modeling.data.assembly.split import SPLITS, assign_split, split_key
from modeling.data.skeletons.fill import fill_conversation, fill_skeleton
from modeling.schema import Conversation, read_conversations, write_conversations, write_rows

# Synthetic data is the bulk of every split. Test gets more than val, and CONDA is weighted
# toward test, because test needs >= 50k negatives and >= 2k positives (design doc §6.1 sizing)
# with margin, and benefits most from real chat.
SYNTHETIC_FRACTIONS = {"train": 0.65, "val": 0.15, "test": 0.20}
CONDA_FRACTIONS = {"train": 0.40, "val": 0.20, "test": 0.40}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skeletons", type=Path, default=Path("data/skeletons_v1/skeletons"))
    parser.add_argument("--conda", type=Path, default=Path("data/external/conda"))
    parser.add_argument("--out", type=Path, default=Path("data/dataset_v1"))
    parser.add_argument("--fills", type=int, default=3)
    parser.add_argument("--obfuscation-rate", type=float, default=0.2)
    parser.add_argument("--splice-rate", type=float, default=0.3,
                        help="fraction of each split's CONDA matches that get a spliced exchange")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args)


def build(args: argparse.Namespace) -> dict:
    t0 = time.time()
    stats: dict = {"args": {k: str(v) for k, v in vars(args).items()}}

    # 1. Skeletons -> split -> fills.
    skeletons: dict[str, list[Conversation]] = {s: [] for s in SPLITS}
    for path in sorted(args.skeletons.glob("*.jsonl")):
        for sk in read_conversations(path):
            skeletons[assign_split(split_key(sk), SYNTHETIC_FRACTIONS)].append(sk)
    stats["skeletons"] = {s: len(v) for s, v in skeletons.items()}
    conversations: dict[str, list[Conversation]] = {
        s: [c for sk in skeletons[s] for c in fill_skeleton(sk, args.fills, args.obfuscation_rate)]
        for s in SPLITS
    }

    # 2. CONDA -> split.
    conda, stats["conda"] = load_matches(args.conda, args.seed)
    conda_by_split: dict[str, list[Conversation]] = {s: [] for s in SPLITS}
    for conv in conda:
        conda_by_split[assign_split(split_key(conv), CONDA_FRACTIONS)].append(conv)

    # 3. Splice exchanges into some CONDA matches, within each split.
    stats["spliced"] = {}
    for split in SPLITS:
        rng = random.Random(f"{args.seed}:splice:{split}")
        hosts = conda_by_split[split]
        rng.shuffle(hosts)
        donors = [sk for sk in skeletons[split] if any(m.label == 1 for m in sk.messages)]
        rng.shuffle(donors)
        n_hosts = round(len(hosts) * args.splice_rate) if donors else 0
        for i, host in enumerate(hosts[:n_hosts]):
            donor = fill_conversation(donors[i % len(donors)], args.fills, args.obfuscation_rate)
            conversations[split].append(splice(host, donor, rng))
        conversations[split].extend(hosts[n_hosts:])
        stats["spliced"][split] = n_hosts

    for split in SPLITS:
        write_conversations(args.out / "conversations" / f"{split}.jsonl", conversations[split])
    print(f"conversations: { {s: len(v) for s, v in conversations.items()} } "
          f"({time.time() - t0:.0f}s)", flush=True)

    # 4. Scoring rows.
    count_tokens = bert_token_counter(args.tokenizer)
    stats["context"] = {"token_budget": DEFAULT_CONFIG.token_budget,
                        "lookback_messages": DEFAULT_CONFIG.lookback_messages}
    stats["dedupe"], rows = {}, {}
    for split in SPLITS:
        all_rows = (r for c in conversations[split] for r in conversation_rows(c, count_tokens))
        kept, stats["dedupe"][split] = dedupe(all_rows)
        path = args.out / "rows" / f"{split}.parquet"
        write_rows(path, kept)
        rows[split] = pd.read_parquet(path)
        print(f"{split}: {len(kept):,} rows ({time.time() - t0:.0f}s)", flush=True)

    # 5. Audit + manual-review sample.
    report = audit(rows, skeletons, stats)
    (args.out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "audit.md").write_text(to_markdown(report), encoding="utf-8")
    manual_review_sample(rows["val"], args.seed).to_csv(
        args.out / "manual_review.csv", index=False, encoding="utf-8-sig")
    print(f"done in {time.time() - t0:.0f}s -> {args.out}", flush=True)
    for check in report["targets"]:
        print(f"  {'ok ' if check['ok'] else 'NO '} {check['check']}: {check['value']} "
              f"(target {check['target']})", flush=True)
    return report


if __name__ == "__main__":
    main()
