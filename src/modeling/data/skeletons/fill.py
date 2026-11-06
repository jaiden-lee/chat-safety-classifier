"""Fill skeleton slots with fake values, producing concrete labeled conversations.

A skeleton is a `Conversation` whose message texts contain slots (prompt: skeleton_v1.txt):

  {KIND}  {KIND:platform}  {KIND#2}  {KIND~1/3}  {KIND!}     (suffixes combine in this order)

- Same kind, platform, and instance number = same value everywhere in the conversation.
- `~i/k` inserts fragment i of k of that value (`split_value`), for split-across-messages cases.
- `!` disguises the value (`obfuscate`). Fills can also disguise unmarked contact slots in
  positive messages at `obfuscation_rate`, so one skeleton yields plain and obfuscated variants.
  All disguised slots in one message use the same transforms where they apply (one person,
  one style), e.g. "snap" and the handle both get look-alike characters.

Whenever a positive message gets a disguised value, its hard tags gain `obfuscated` (and lose
`plain`), so the tag always matches the text.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re
from dataclasses import dataclass

from modeling.data.primitives.obfuscate import applicable, obfuscate, split_value
from modeling.data.primitives.values import PLATFORMS, ValueSampler
from modeling.schema import Conversation, HardTag, Message

SLOT_RE = re.compile(
    r"\{(?P<kind>[A-Z_]+)(?::(?P<platform>[a-z]+))?(?:#(?P<instance>\d+))?"
    r"(?:~(?P<part>\d+)/(?P<parts>\d+))?(?P<obfuscate>!)?\}"
)
PLATFORM_KINDS = {"HANDLE", "LINK", "PLATFORM", "PLATFORM_HINT"}
OBFUSCATABLE_KINDS = {"PHONE", "EMAIL", "HANDLE", "LINK", "PLATFORM", "ZIP"}
SLOT_KINDS = set(ValueSampler(0).kinds)


@dataclass(frozen=True)
class Slot:
    """One parsed slot occurrence."""

    kind: str
    platform: str | None
    instance: int
    part: int | None  # fragment number (1-based), if this is a fragment
    parts: int | None
    obfuscate: bool

    @property
    def key(self) -> tuple[str, str | None, int]:
        """Identity of the underlying value within a conversation."""
        return (self.kind, self.platform, self.instance)

    @classmethod
    def from_match(cls, m: re.Match) -> Slot:
        return cls(
            kind=m["kind"],
            platform=m["platform"],
            instance=int(m["instance"] or 1),
            part=int(m["part"]) if m["part"] else None,
            parts=int(m["parts"]) if m["parts"] else None,
            obfuscate=bool(m["obfuscate"]),
        )


def slot_errors(text: str) -> list[str]:
    """Problems with the slots in `text`; empty if every slot is well-formed and known."""
    errors = []
    for m in SLOT_RE.finditer(text):
        slot = Slot.from_match(m)
        if slot.kind not in SLOT_KINDS:
            errors.append(f"unknown slot kind {slot.kind!r}")
        if slot.platform is not None:
            if slot.kind not in PLATFORM_KINDS:
                errors.append(f"{slot.kind} doesn't take a platform")
            elif slot.platform not in PLATFORMS:
                errors.append(f"unknown platform {slot.platform!r}")
        if slot.parts is not None and not (2 <= slot.parts and 1 <= slot.part <= slot.parts):
            errors.append(f"bad fragment {slot.part}/{slot.parts}")
    leftover = SLOT_RE.sub("", text)
    if re.search(r"\{[^}]*\}", leftover):
        errors.append(f"malformed slot in {text!r}")
    return errors


def fill_seed(skeleton_id: str, fill_id: int) -> int:
    """Stable seed per (skeleton, fill); Python's hash() is salted per process, so use sha256."""
    digest = hashlib.sha256(f"{skeleton_id}:{fill_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def fill_conversation(
    skeleton: Conversation, fill_id: int, obfuscation_rate: float = 0.0
) -> Conversation:
    """One concrete conversation from a skeleton. Deterministic for a given (skeleton, fill_id)."""
    values = ValueSampler(fill_seed(skeleton.conversation_id, fill_id))
    rng = values.rng
    cache: dict[tuple, str] = {}
    fragments: dict[tuple, list[str]] = {}

    def value_for(slot: Slot) -> str:
        if slot.parts is None:
            if slot.key not in cache:
                cache[slot.key] = values.sample(slot.kind, slot.platform)
            return cache[slot.key]
        frag_key = (slot.key, slot.parts)
        if frag_key not in fragments:
            fragments[frag_key] = _splittable_value(values, slot, cache, rng)
        return fragments[frag_key][slot.part - 1]

    messages = []
    for msg in skeleton.messages:
        disguised = False
        style: list[str] = []  # transforms of the message's first disguise, reused for the rest

        def replace(m: re.Match, msg: Message = msg) -> str:
            nonlocal disguised, style
            slot = Slot.from_match(m)
            value = value_for(slot)
            wanted = slot.obfuscate or (
                msg.label == 1 and slot.kind in OBFUSCATABLE_KINDS
                and rng.random() < obfuscation_rate
            )
            if wanted and slot.kind in OBFUSCATABLE_KINDS:
                out = _disguise(value, style, rng)
                if out is not None:
                    value, style, disguised = out.text, list(out.transforms), True
            return value

        text = SLOT_RE.sub(replace, msg.text)
        hard_tags = msg.hard_tags
        if disguised and msg.label == 1:
            hard_tags = tuple(t for t in hard_tags if t != HardTag.PLAIN)
            if HardTag.OBFUSCATED not in hard_tags:
                hard_tags += (HardTag.OBFUSCATED,)
        messages.append(dataclasses.replace(msg, text=text, hard_tags=hard_tags))

    return dataclasses.replace(
        skeleton, conversation_id=f"{skeleton.conversation_id}-f{fill_id}",
        messages=tuple(messages),
    )


def fill_skeleton(
    skeleton: Conversation, n_fills: int, obfuscation_rate: float = 0.2
) -> list[Conversation]:
    """`n_fills` conversations from one skeleton. Fill 0 only disguises `!` slots; the others
    also disguise unmarked contact slots in positive messages at `obfuscation_rate`."""
    return [
        fill_conversation(skeleton, k, obfuscation_rate if k > 0 else 0.0)
        for k in range(n_fills)
    ]


def _disguise(value: str, style: list[str], rng: random.Random):
    """Obfuscate with the message's existing style where it applies, else a random one.
    Returns None if nothing can disguise the value."""
    names = [t for t in style if t in applicable(value)]
    for attempt in (names, None):
        if attempt == []:
            continue
        try:
            return obfuscate(value, rng, attempt)
        except ValueError:
            continue
    return None


def _splittable_value(
    values: ValueSampler, slot: Slot, cache: dict, rng: random.Random
) -> list[str]:
    """Split the slot's value into fragments, resampling if the value is too short to split."""
    for _ in range(10):
        if slot.key not in cache:
            cache[slot.key] = values.sample(slot.kind, slot.platform)
        try:
            return split_value(cache[slot.key], slot.parts, rng)
        except ValueError:
            del cache[slot.key]
    raise ValueError(f"couldn't get a splittable value for {slot}")
