"""Fill generated skeletons and write a human-readable preview, for reviewing generation quality.

Writes, under `--out`:
  conversations.jsonl   every fill of every skeleton (the mock dataset)
  preview.md            per skeleton: the skeleton with labels, each fill, and the scoring rows
                        of one obfuscated fill as the model would see them

    uv run python -m modeling.data.skeletons.preview --out data/pilot --fills 3
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from modeling.context import build_context_at, whitespace_token_count
from modeling.data.skeletons.fill import fill_skeleton
from modeling.schema import Conversation, Message, read_conversations, write_conversations


def _label(m: Message) -> str:
    if m.label == 1:
        return f"POS {'+'.join(m.categories)} / {'+'.join(m.hard_tags)}"
    return f"neg {m.neg_type}"


def _render_messages(conv: Conversation) -> list[str]:
    width = min(70, max(len(m.text) for m in conv.messages))
    return [f"{i:>2}  {m.speaker_id}  {m.text:<{width}}  {_label(m)}"
            for i, m in enumerate(conv.messages)]


def _render_rows(conv: Conversation) -> list[str]:
    lines = []
    for i, m in enumerate(conv.messages):
        ctx = build_context_at(conv, i, whitespace_token_count)
        lines += [f"row {i}  label={m.label}  ({_label(m)})",
                  f"  A: {ctx.context_text}", f"  B: {ctx.target_text}"]
    return lines


def render_preview(skeletons: list[Conversation], fills: dict[str, list[Conversation]]) -> str:
    out = ["# Generation pilot preview", ""]
    msgs = [m for s in skeletons for m in s.messages]
    filled = [m for fs in fills.values() for f in fs for m in f.messages]
    pos = [m for m in filled if m.label == 1]
    out += [
        (f"{len(skeletons)} skeletons, {sum(map(len, fills.values()))} filled conversations, "
        f"{len(filled)} messages ({len(pos)} positive)."), "",
        "Skeleton labels: " + ", ".join(f"{k}: {v}" for k, v in sorted(
            Counter(_label(m) for m in msgs).items())), "",
        "Filled positive hard tags: " + ", ".join(f"{k}: {v}" for k, v in sorted(
            Counter("+".join(m.hard_tags) for m in pos).items())), "",
    ]
    for sk in skeletons:
        p = sk.provenance
        out += [f"## {sk.conversation_id}", "",
                (f"{sk.channel_type}, {len(sk.speakers)} speakers, {len(sk.messages)} messages, "
                f"model `{p.generator_model}`"), "", "**Skeleton**", "```",
                *_render_messages(sk), "```", ""]
        for f in fills[sk.conversation_id]:
            kind = "plain, only `!` slots disguised" if f.conversation_id.endswith("-f0") \
                else "contact slots in positives disguised at random"
            out += [f"**Fill {f.conversation_id.rsplit('-f', 1)[1]}** ({kind})", "```",
                    *_render_messages(f), "```", ""]
        last = fills[sk.conversation_id][-1]
        out += [("<details><summary>Scoring rows for the last fill (what the model sees)"
                "</summary>"), "", "```", *_render_rows(last), "```", "</details>", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="directory with skeletons/")
    parser.add_argument("--fills", type=int, default=3)
    parser.add_argument("--obfuscation-rate", type=float, default=0.5)
    args = parser.parse_args()

    skeletons = [s for path in sorted((args.out / "skeletons").glob("*.jsonl"))
                 for s in read_conversations(path)]
    fills = {s.conversation_id: fill_skeleton(s, args.fills, args.obfuscation_rate)
             for s in skeletons}
    n = write_conversations(args.out / "conversations.jsonl",
                            [f for fs in fills.values() for f in fs])
    (args.out / "preview.md").write_text(render_preview(skeletons, fills), encoding="utf-8")
    print(f"{len(skeletons)} skeletons -> {n} conversations; wrote {args.out / 'preview.md'}")


if __name__ == "__main__":
    main()
