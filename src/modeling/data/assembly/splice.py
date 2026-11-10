"""Splice synthetic contact exchanges into real CONDA match chat.

The positive part of a filled synthetic conversation (from a little before its first positive
message to a little after its last) is inserted at a random point of a CONDA match, with its
speakers mapped onto the match's players. The result has real chat around a synthetic exchange,
so positives aren't only ever seen inside LLM-written chat.

Leakage: a spliced conversation inherits the donor's `skeleton_id`, and hosts and donors are
always drawn from the same split.
"""

from __future__ import annotations

import dataclasses
import random

from modeling.schema import Conversation, Message, Provenance, Source

SPAN_BEFORE = 2  # negative messages kept before the first positive (the lead-up)
SPAN_AFTER = 1  # and after the last positive


def exchange_span(conv: Conversation) -> tuple[Message, ...]:
    """The messages around the conversation's positives. The conversation must have a positive."""
    positives = [i for i, m in enumerate(conv.messages) if m.label == 1]
    if not positives:
        raise ValueError(f"{conv.conversation_id} has no positive messages")
    lo = max(0, positives[0] - SPAN_BEFORE)
    hi = min(len(conv.messages), positives[-1] + 1 + SPAN_AFTER)
    return conv.messages[lo:hi]


def splice(host: Conversation, donor: Conversation, rng: random.Random) -> Conversation:
    """`host` (real chat) with `donor`'s exchange span inserted at a random position."""
    span = exchange_span(donor)
    donor_speakers = list(dict.fromkeys(m.speaker_id for m in span))
    players = host.speakers
    rng.shuffle(players)
    newcomers = (f"S{i}" for i in range(1, len(donor_speakers) + 1))  # if the host has too few
    mapping = {s: players[i] if i < len(players) else next(newcomers)
               for i, s in enumerate(donor_speakers)}

    at = rng.randint(0, len(host.messages))
    ts = host.messages[at - 1].ts if at > 0 else host.messages[0].ts
    inserted = tuple(dataclasses.replace(m, speaker_id=mapping[m.speaker_id], ts=ts) for m in span)
    return Conversation(
        conversation_id=f"splice-{host.conversation_id}-{donor.conversation_id}",
        channel_type=host.channel_type,
        messages=host.messages[:at] + inserted + host.messages[at:],
        provenance=Provenance(
            source=Source.SPLICE,
            skeleton_id=donor.provenance.skeleton_id,
            template_id=donor.provenance.template_id,
            generator_model=donor.provenance.generator_model,
        ),
    )
