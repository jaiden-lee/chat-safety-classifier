"""Context builder (design doc §5.2): turns a target message and the chat before it into model input.

Training uses this module directly; serving may reimplement it in another language, so the
algorithm is kept deliberately simple and must stay byte-for-byte reproducible.

Output for one target message:
- `context_text` (BERT segment A): a channel token, then the selected earlier messages in
  chronological order, each prefixed by a speaker tag relative to the target's author.
- `target_text` (BERT segment B): the target message's text.

    [GROUP] [U1] anyone got discord [U2] me [SELF] same      <- context_text
    whats ur tag                                              <- target_text

Selection:
1. Take the last `lookback_messages` messages before the target.
2. Walk backward from the newest, adding each message while the total stays within
   `token_budget`; stop at the first message that doesn't fit. The target always counts
   toward the budget, so the kept context is one contiguous run of the most recent messages.
3. Assign tags over the kept run in chronological order and join with single spaces.

Token cost of a context message is 1 (its tag; every tag is a single token once
SPECIAL_TOKENS are added) plus the token count of its text.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from modeling.schema import ChannelType, Conversation, Message

SELF_TAG = "[SELF]"
MAX_OTHER_TAGS = 9  # [U1] .. [U9]; further speakers share [UOTHER]
OTHER_TAGS = [f"[U{i}]" for i in range(1, MAX_OTHER_TAGS + 1)]
UOTHER_TAG = "[UOTHER]"
CHANNEL_TAGS = {ChannelType.DIRECT: "[DM]", ChannelType.GROUP: "[GROUP]"}

# Tokens to add to the tokenizer vocabulary so each tag is a single token.
SPECIAL_TOKENS = [SELF_TAG, *OTHER_TAGS, UOTHER_TAG, *CHANNEL_TAGS.values()]

# [CLS] context [SEP] target [SEP]
SEQUENCE_OVERHEAD_TOKENS = 3

# BERT's maximum sequence length. The builder targets `ContextConfig.token_budget`; this cap is
# enforced by tokenizer truncation for the rare target message that is longer on its own.
MAX_SEQ_LEN = 512

TokenCounter = Callable[[str], int]


def whitespace_token_count(text: str) -> int:
    """Rough token count for tests and tokenizer-free use. Use a real tokenizer's count for data."""
    return len(text.split())


def hf_token_counter(tokenizer) -> TokenCounter:
    """Token counter backed by a Hugging Face tokenizer that already has SPECIAL_TOKENS added."""
    return lambda text: len(tokenizer.tokenize(text))


@dataclass(frozen=True)
class ContextConfig:
    """Context builder parameters (design doc §5.2)."""

    token_budget: int = 256  # total sequence length target, including target and overhead
    lookback_messages: int = 50


DEFAULT_CONFIG = ContextConfig()


@dataclass(frozen=True)
class BuiltContext:
    """Model input for one target message, plus metadata used for evaluation slices."""

    context_text: str  # segment A
    target_text: str  # segment B
    total_tokens: int  # full sequence length, including [CLS]/[SEP]
    n_context_messages: int  # the kept context is history[-n_context_messages:]
    n_speakers: int  # distinct speakers in the kept context plus the target's author


def build_context(
    history: Sequence[Message],
    target: Message,
    channel_type: ChannelType | str,
    count_tokens: TokenCounter,
    config: ContextConfig = DEFAULT_CONFIG,
) -> BuiltContext:
    """Build model input for `target`, given the earlier messages `history` (oldest first)."""
    channel_tag = CHANNEL_TAGS[ChannelType(channel_type)]
    author = target.speaker_id

    candidates = history[-config.lookback_messages:] if config.lookback_messages > 0 else []

    used = SEQUENCE_OVERHEAD_TOKENS + 1 + count_tokens(target.text)  # +1: channel tag
    n_kept = 0
    for msg in reversed(candidates):
        cost = 1 + count_tokens(msg.text)  # +1: speaker tag
        if used + cost > config.token_budget:
            break
        used += cost
        n_kept += 1

    kept = candidates[len(candidates) - n_kept:]
    tags = _speaker_tags([m.speaker_id for m in kept], author)
    parts = [channel_tag] + [f"{tags[m.speaker_id]} {m.text}" for m in kept]
    return BuiltContext(
        context_text=" ".join(parts),
        target_text=target.text,
        total_tokens=used,
        n_context_messages=n_kept,
        n_speakers=len({m.speaker_id for m in kept} | {author}),
    )


def build_context_at(
    conversation: Conversation,
    target_index: int,
    count_tokens: TokenCounter,
    config: ContextConfig = DEFAULT_CONFIG,
) -> BuiltContext:
    """Build model input for message `target_index` of a conversation."""
    msgs = conversation.messages
    return build_context(
        msgs[:target_index], msgs[target_index], conversation.channel_type, count_tokens, config
    )


def _speaker_tags(speakers_in_order: Sequence[str], author: str) -> dict[str, str]:
    """Map speaker ids to relative tags: the author is [SELF]; others [U1].. by first appearance."""
    tags = {author: SELF_TAG}
    n_others = 0
    for s in speakers_in_order:
        if s in tags:
            continue
        tags[s] = OTHER_TAGS[n_others] if n_others < MAX_OTHER_TAGS else UOTHER_TAG
        n_others += 1
    return tags
