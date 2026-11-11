"""CONDA (real Dota 2 match chat) as all-negative group conversations.

CONDA (Weld et al., 2021; https://huggingface.co/datasets/Matrix430/CONDA, license AFL-3.0) has
about 45k chat utterances from 1.9k Dota 2 matches, labeled for toxicity (which we ignore). It
gives the dataset real gamer chat, so "real-looking chat" isn't something the model can use as a
shortcut for "negative", and it hosts spliced-in synthetic exchanges (splice.py).

How it becomes Conversations:
- One match = one group conversation, utterances ordered by `chatTime` (its `ts`).
- An utterance joins one player's consecutive lines with " [SEPA] "; they're split back into
  separate messages from the same speaker.
- CONDA has no player ids. Because consecutive lines were merged, adjacent utterances come from
  different players, so each utterance gets a random speaker from a per-match pool, never the
  previous utterance's speaker. Speaker structure is therefore approximate.
- Every message is labeled negative (`easy`). Matches with anything that could be real contact
  info (a URL, an email, a run of 7+ digits) are dropped, since those would be unlabeled
  positives (e.g., "twitch.tv/<name>"). Keyword hits like "fb" (first blood) or "steam lag" are
  kept: they're realistic look-alike negatives.
- A keyword scan of the remaining messages (platform names, "add me", "phone", "email", ...)
  was read by hand; matches with a message that is positive under our guidelines are listed in
  REVIEWED_POSITIVE_MATCHES and dropped too. Positives without such keywords may remain; the
  rate looks very low (6 found in ~60k messages).
"""

from __future__ import annotations

import csv
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from modeling.schema import ChannelType, Conversation, Message, NegType, Provenance, Source

CONDA_FILES = ("train.csv", "valid.csv", "test.csv")
CONDA_URL = "https://huggingface.co/datasets/Matrix430/CONDA/resolve/main/{name}"
LINE_SEPARATOR = "[SEPA]"

# Matches with a message that's positive under our guidelines, found by hand-reading a keyword
# scan of the screened data (match id: the message).
REVIEWED_POSITIVE_MATCHES = {
    "358": "wat is ur facebook",
    "482": "sad life brah come add me on steam",  # Steam is off-platform (guidelines §6.2)
    "1228": "do u guys have a twitter??",
    "1438": "add me on tidner",
    "1778": "follow me on instagram ...",
    "2007": "send me your email so i can send you a video ...",
}

_CONTACT_PATTERNS = [
    (re.compile(r"https?://|www\.|\b[\w-]+\.(com|gg|me|tv|net|org|io|ru)\b", re.IGNORECASE), 0),
    (re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE), 0),
    (re.compile(r"\d[\d\s().-]{5,}\d"), 7),  # (pattern, minimum digits in the match)
]


def download(conda_dir: Path) -> None:
    """Fetch the CONDA CSVs into `conda_dir` (skips files already there)."""
    conda_dir.mkdir(parents=True, exist_ok=True)
    for name in CONDA_FILES:
        path = conda_dir / name
        if not path.exists():
            urllib.request.urlretrieve(CONDA_URL.format(name=name), path)


def might_be_contact_info(text: str) -> bool:
    """True if `text` contains a URL, an email, or a run of 7+ digits."""
    for pattern, min_digits in _CONTACT_PATTERNS:
        for m in pattern.finditer(text):
            if sum(c.isdigit() for c in m.group()) >= min_digits:
                return True
    return False


def load_matches(conda_dir: Path, seed: int = 0) -> tuple[list[Conversation], dict]:
    """All CONDA matches as Conversations (deterministic for a seed), plus screening stats."""
    download(conda_dir)
    utterances: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    for name in CONDA_FILES:
        with (conda_dir / name).open(encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                utterances[row["matchId"]].append(
                    (float(row["chatTime"]), int(row["Id"]), row["utterance"]))

    conversations, dropped, reviewed = [], [], []
    for match_id in sorted(utterances, key=int):
        rows = sorted(utterances[match_id])  # by chatTime, then Id
        if any(might_be_contact_info(text) for _, _, text in rows):
            dropped.append(match_id)
            continue
        if match_id in REVIEWED_POSITIVE_MATCHES:
            reviewed.append(match_id)
            continue
        conv = _to_conversation(match_id, rows, random.Random(f"{seed}:conda:{match_id}"))
        if conv is not None:
            conversations.append(conv)
    stats = {"matches": len(utterances), "dropped_contact_like": len(dropped),
             "dropped_reviewed_positive": len(reviewed), "kept": len(conversations)}
    return conversations, stats


def _to_conversation(match_id: str, rows: list, rng: random.Random) -> Conversation | None:
    pool = [f"P{i}" for i in range(1, rng.randint(3, 8) + 1)]
    messages, previous = [], None
    for ts, _, utterance in rows:
        lines = [s.strip() for s in utterance.split(LINE_SEPARATOR) if s.strip()]
        if not lines:
            continue
        speaker = rng.choice([p for p in pool if p != previous])
        previous = speaker
        messages.extend(Message(speaker_id=speaker, text=line, ts=ts, label=0,
                                neg_type=NegType.EASY) for line in lines)
    if not messages:
        return None
    return Conversation(
        conversation_id=f"conda-m{match_id}",
        channel_type=ChannelType.GROUP,
        messages=tuple(messages),
        provenance=Provenance(source=Source.CONDA),
    )
