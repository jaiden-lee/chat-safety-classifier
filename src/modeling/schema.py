"""Core data types shared by every stage of the modeling pipeline.

The dataset has two levels:

1. Conversations (source data). Generation produces `Conversation`s: ordered `Message`s where
   every message carries its own label, because the label is about one message (guidelines §1).
   Conversations are what we generate, audit, and split into train/val/test.

2. Scoring rows (model data). Each conversation of N messages is expanded into N `ScoringRow`s.
   Row i is "message i, judged with the messages before it": the context builder renders the
   earlier messages into `context_text` (segment A), message i becomes `target_text` (segment B),
   and the row's label is message i's label. This is what BERT trains on and what evaluation scores.

       Conversation: m1 m2 m3 m4   ->   rows: [m1] [m1|m2] [m1 m2|m3] [m1 m2 m3|m4]
                                                   (context | target)

Label rules and tag definitions live in docs/labeling_guidelines.md.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from modeling import GUIDELINE_VERSION


class ChannelType(StrEnum):
    """Whether a conversation is a one-to-one DM or a group chat."""

    DIRECT = "direct"
    GROUP = "group"


class Category(StrEnum):
    """Kind of contact info a positive message shares or asks for (guidelines §5.1)."""

    HANDLE = "handle"
    PHONE = "phone"
    EMAIL = "email"
    LOCATION = "location"
    PLATFORM_MOVE = "platform_move"


class HardTag(StrEnum):
    """Why a positive message is hard to catch (guidelines §5.2); used for the Q2 slices."""

    PLAIN = "plain"  # not hard; excludes the others
    OBFUSCATED = "obfuscated"
    CROSS_MESSAGE = "cross_message"
    IMPLICIT = "implicit"


class NegType(StrEnum):
    """What a negative message looks like (guidelines §5.3); the non-easy types are Q3's hard negatives."""

    EASY = "easy"
    NUMBER_LOOKALIKE = "number_lookalike"
    INGAME_SOCIAL = "ingame_social"
    APP_TOPIC = "app_topic"
    INGAME_NAME_REFERENCE = "ingame_name_reference"
    COARSE_LOCATION = "coarse_location"
    REFUSAL_WARNING = "refusal_warning"
    POST_EXCHANGE = "post_exchange"


class Source(StrEnum):
    """How a conversation was produced."""

    LLM_SKELETON = "llm_skeleton"  # LLM-written skeleton, slots filled by code
    TEMPLATE = "template"  # fully rule/template generated (e.g., hard negatives)
    CONDA = "conda"  # real Dota 2 chat (CONDA dataset), all negative
    SPLICE = "splice"  # synthetic exchange spliced into a CONDA conversation


@dataclass(frozen=True)
class Message:
    """One chat message inside a Conversation, with its label (generated data) or without (serving).

    `label` is 1 (positive) or 0 (negative). Positives carry `categories` and `hard_tags`;
    negatives carry `neg_type`. Strings are accepted and converted to the enums above.
    """

    speaker_id: str  # opaque id, never a real username
    text: str
    ts: float | None = None  # unix seconds
    label: int | None = None  # None only for unlabeled (serving) messages
    categories: tuple[Category, ...] = ()
    hard_tags: tuple[HardTag, ...] = ()
    neg_type: NegType | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__  # frozen dataclass
        set_(self, "categories", tuple(map(Category, self.categories)))
        set_(self, "hard_tags", tuple(map(HardTag, self.hard_tags)))
        set_(self, "neg_type", NegType(self.neg_type) if self.neg_type else None)
        self._validate_label()

    def _validate_label(self) -> None:
        if self.label is None:
            if self.categories or self.hard_tags or self.neg_type:
                raise ValueError("an unlabeled message can't have tags")
        elif self.label == 1:
            if not self.categories or not self.hard_tags:
                raise ValueError("a positive needs at least one category and one hard tag")
            if self.neg_type is not None:
                raise ValueError("a positive can't have a neg_type")
            if HardTag.PLAIN in self.hard_tags and len(self.hard_tags) > 1:
                raise ValueError("'plain' can't be combined with other hard tags")
        elif self.label == 0:
            if self.categories or self.hard_tags:
                raise ValueError("a negative can't have categories or hard tags")
            if self.neg_type is None:
                raise ValueError("a negative needs a neg_type")
        else:
            raise ValueError(f"label must be 0, 1, or None, got {self.label!r}")


@dataclass(frozen=True)
class Provenance:
    """Where a Conversation came from; used for the leakage-safe split and the unseen-template slice.

    Synthetic fills of one skeleton are near-duplicates, so the split keeps every conversation
    with the same `skeleton_id` in the same split, and some `template_id`s appear only in test
    (design doc §6.1).
    """

    source: Source
    skeleton_id: str | None = None  # split grouping key
    template_id: str | None = None  # prompt/template version
    generator_model: str | None = None
    guideline_version: str = GUIDELINE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Source(self.source))


@dataclass(frozen=True)
class Conversation:
    """One chat conversation: the unit we generate, audit, and split into train/val/test."""

    conversation_id: str
    channel_type: ChannelType
    messages: tuple[Message, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_type", ChannelType(self.channel_type))

    @property
    def speakers(self) -> list[str]:
        """Speaker ids in order of first appearance."""
        return list(dict.fromkeys(m.speaker_id for m in self.messages))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Conversation:
        return cls(
            conversation_id=d["conversation_id"],
            channel_type=d["channel_type"],
            messages=tuple(Message(**m) for m in d["messages"]),
            provenance=Provenance(**d["provenance"]),
        )


@dataclass
class ScoringRow:
    """One model input: a target message plus the rendered context before it, with the target's label.

    Built from message `target_index` of conversation `conversation_id` by the context builder.
    One row per message; rows make up the train/val/test tables.
    """

    row_id: str  # f"{conversation_id}:{target_index}"
    conversation_id: str
    target_index: int
    channel_type: str
    context_text: str  # segment A: speaker-tagged earlier messages, within the token budget
    target_text: str  # segment B: the message being judged
    label: int
    categories: list[str] = field(default_factory=list)
    hard_tags: list[str] = field(default_factory=list)
    neg_type: str | None = None
    # Slice metadata (design doc §6.6).
    total_tokens: int = 0  # full sequence length incl. [CLS]/[SEP] (design doc n)
    n_context_messages: int = 0
    n_speakers: int = 0
    # Copied from the conversation's provenance, for splitting and slicing.
    source: str = ""
    skeleton_id: str | None = None
    template_id: str | None = None
    guideline_version: str = GUIDELINE_VERSION


# --- IO ---------------------------------------------------------------------------------------


def write_conversations(path: str | Path, conversations: Iterable[Conversation]) -> int:
    """Write conversations as JSONL. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_conversations(path: str | Path) -> Iterator[Conversation]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Conversation.from_dict(json.loads(line))


def write_rows(path: str | Path, rows: Iterable[ScoringRow]) -> int:
    """Write scoring rows as a parquet table. Returns the number written."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_parquet(path, index=False)
    return len(df)


_OPTIONAL_STR_FIELDS = ("neg_type", "skeleton_id", "template_id")


def read_rows(path: str | Path) -> list[ScoringRow]:
    import pandas as pd

    records = pd.read_parquet(path).to_dict(orient="records")
    for r in records:
        r["categories"] = list(r["categories"])
        r["hard_tags"] = list(r["hard_tags"])
        # pandas reads missing strings back as NaN; restore None.
        for key in _OPTIONAL_STR_FIELDS:
            if not isinstance(r[key], str):
                r[key] = None
    return [ScoringRow(**r) for r in records]
