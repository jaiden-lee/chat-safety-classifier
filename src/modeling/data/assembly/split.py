"""Leakage-safe train/val/test assignment (design doc §6.1).

Conversations are grouped by `split_key`: the skeleton id for synthetic data (so all fills and
splices of one skeleton land together) and the match id for CONDA. A group's split is a stable
hash of its key, so adding data never moves existing groups between splits.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from modeling.schema import Conversation

SPLITS = ("train", "val", "test")


def split_key(conv: Conversation) -> str:
    """Grouping key: conversations sharing a key always share a split."""
    return conv.provenance.skeleton_id or conv.conversation_id


def assign_split(key: str, fractions: Mapping[str, float], salt: str = "split-v1") -> str:
    """Deterministic split for a group key, given fractions per split (summing to 1)."""
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError(f"fractions must sum to 1: {dict(fractions)}")
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name in SPLITS:
        cumulative += fractions.get(name, 0.0)
        if u < cumulative:
            return name
    return SPLITS[-1]
