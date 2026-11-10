"""Expand conversations into ScoringRows (one per message) and de-duplicate them.

Token counts use the real BERT tokenizer (via the light `tokenizers` package, no torch), with the
context builder's tags added as special tokens, the same way training will see them.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from modeling.context import (
    DEFAULT_CONFIG,
    SPECIAL_TOKENS,
    ContextConfig,
    TokenCounter,
    build_context_at,
)
from modeling.schema import Conversation, ScoringRow

DEFAULT_TOKENIZER = "bert-base-uncased"


def bert_token_counter(name: str = DEFAULT_TOKENIZER) -> TokenCounter:
    """Token counter matching `AutoTokenizer.from_pretrained(name)` with SPECIAL_TOKENS added."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(name)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    @lru_cache(maxsize=1_000_000)
    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


def conversation_rows(
    conv: Conversation, count_tokens: TokenCounter, config: ContextConfig = DEFAULT_CONFIG
) -> list[ScoringRow]:
    """One ScoringRow per message of a labeled conversation."""
    rows = []
    for i, msg in enumerate(conv.messages):
        built = build_context_at(conv, i, count_tokens, config)
        rows.append(ScoringRow(
            row_id=f"{conv.conversation_id}:{i}",
            conversation_id=conv.conversation_id,
            target_index=i,
            channel_type=str(conv.channel_type),
            context_text=built.context_text,
            target_text=built.target_text,
            label=msg.label,
            categories=[str(c) for c in msg.categories],
            hard_tags=[str(t) for t in msg.hard_tags],
            neg_type=str(msg.neg_type) if msg.neg_type else None,
            total_tokens=built.total_tokens,
            n_context_messages=built.n_context_messages,
            n_speakers=built.n_speakers,
            source=str(conv.provenance.source),
            skeleton_id=conv.provenance.skeleton_id,
            template_id=conv.provenance.template_id,
            guideline_version=conv.provenance.guideline_version,
        ))
    return rows


def dedupe(rows: Iterable[ScoringRow]) -> tuple[list[ScoringRow], dict]:
    """Drop rows whose model input (context, target) repeats an earlier row.

    Fills of one skeleton share every message without a slot, so their rows often repeat
    exactly; keeping them would count one example several times. If repeats disagree on the
    label, every copy is dropped (the input can't have two right answers) and counted.
    """
    first: dict[tuple[str, str], ScoringRow] = {}
    labels: dict[tuple[str, str], set[int]] = {}
    copies: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row.context_text, row.target_text)
        first.setdefault(key, row)
        labels.setdefault(key, set()).add(row.label)
        copies[key] = copies.get(key, 0) + 1
    conflicts = [k for k, ls in labels.items() if len(ls) > 1]
    for k in conflicts:
        del first[k]
    kept = list(first.values())
    n_in, n_conflict_rows = sum(copies.values()), sum(copies[k] for k in conflicts)
    return kept, {"rows_in": n_in, "duplicates_dropped": n_in - len(kept) - n_conflict_rows,
                  "label_conflict_rows_dropped": n_conflict_rows,
                  "label_conflict_inputs": len(conflicts)}
