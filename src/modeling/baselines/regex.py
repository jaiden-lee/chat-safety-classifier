"""Regex baseline (design doc §6.5): a plain keyword-and-pattern filter, the "day one" approach.

    uv run python -m modeling.baselines.regex --rows data/dataset_v1/rows --out outputs/regex

How it decides
- It looks at the **target message only** (no context), **lowercased**, nothing else (no
  Unicode cleanup, so look-alike characters defeat it, as they would a real keyword filter).
- A message is **flagged if any rule below matches**. There's no score: the output score is
  1.0 (flagged) or 0.0 (not), plus the name of the first rule that matched.

Everything it detects is listed in this file, in plain lists, in this order:

  1. phone          7+ digits, allowing spaces / dashes / dots / parentheses between them
  2. email          something@something.something
  3. link           LINK_DOMAINS followed by "/...", or any http(s):// URL
  4. platform_ask   a VERB_BEFORE_PLATFORM phrase, then a PLATFORM ("add me on" + "snap")
  5. platform_offer "my" + PLATFORM + is/=/: ("my insta is"), or PLATFORM + "me" ("snap me")
  6. contact_phrase CONTACT_PHRASES: number/email/text-me requests without a platform name
  7. move_phrase    MOVE_PHRASES: moving the chat elsewhere without naming a platform
  8. location       LOCATION_PHRASES: asking for or giving a specific place

Platform names alone ("snap", "discord") are NOT flagged: "oh snap", "cold snap", and
"discord is down" are ordinary chat. A platform only counts next to an ask/offer phrase.

The lists were written once from common sense, not tuned on our data (design doc §6.5),
and are independent of the data generator's lists.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# --- what it looks for --------------------------------------------------------------------------

PLATFORMS = [
    "snapchat", "snap", "instagram", "insta", "ig", "discord", "whatsapp", "whats app",
    "telegram", "kik", "tiktok", "twitter", "facebook", "messenger", "signal", "skype",
    "twitch", "youtube", "reddit", "wechat", "steam", "xbox", "psn", "epic games",
    "battle.net",
]

LINK_DOMAINS = [
    "discord.gg", "discord.com/invite", "t.me", "wa.me", "instagram.com", "snapchat.com",
    "tiktok.com", "twitch.tv", "youtube.com", "youtu.be", "twitter.com", "x.com",
    "facebook.com", "fb.com", "reddit.com", "steamcommunity.com",
]

# Phrases that come right before a platform name: "<phrase> <platform>".
VERB_BEFORE_PLATFORM = [
    # asking / offering to connect there
    "add me on", "add me at", "add on", "add my", "follow me on", "follow my", "follow on",
    "dm me on", "message me on", "msg me on", "pm me on", "hmu on", "hit me up on",
    "text me on", "find me on", "join my", "sub to my", "subscribe to my", "check out my",
    # asking for someone's account
    "whats ur", "whats your", "what's ur", "what's your", "what is ur", "what is your",
    "wats ur", "wuts ur", "u have", "you have", "do u have", "do you have", "u got",
    "you got", "drop ur", "drop your", "send ur", "send your", "give me ur", "give me your",
    # moving there
    "talk on", "chat on", "move to", "go to", "switch to", "hop on", "continue on",
]

# Contact requests and offers that don't name a platform.
CONTACT_PHRASES = [
    "text me", "call me at", "ur number", "your number", "my number", "phone number",
    "ur cell", "your cell", "my cell", "ur digits", "your digits",
    "ur email", "your email", "my email", "ur gmail", "your gmail", "my gmail",
]

# Moving the conversation elsewhere without naming where.
MOVE_PHRASES = [
    "somewhere else", "somewhere more private", "somewhere private", "off this app",
    "off this game", "off the game", "outside the game", "outside this game",
    "a different app", "another app", "other app",
]

# Asking for or giving a specific location. ("i live in" is left out on purpose: it is
# usually a city or country, which is allowed.)
LOCATION_PHRASES = [
    "i live on", "i live at", "i live near", "i live by", "i live off",
    "where do u live", "where do you live", "where u live", "my address", "ur address",
    "your address", "my addy", "ur addy", "what school", "which school", "my school is",
    "what street", "which street",
]

# --- the rules, built from the lists ------------------------------------------------------------


def _any_of(phrases: list[str]) -> str:
    """Regex alternation of literal phrases, longest first, as whole words."""
    alternation = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return rf"(?<!\w)(?:{alternation})(?!\w)"


_PLATFORM = _any_of(PLATFORMS)

RULES: list[tuple[str, re.Pattern]] = [
    ("phone", re.compile(r"\+?\(?\d(?:[\s().-]{0,2}\d){6,}")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("link", re.compile(rf"(?:{'|'.join(map(re.escape, LINK_DOMAINS))})/\S+|https?://\S+")),
    ("platform_ask", re.compile(rf"{_any_of(VERB_BEFORE_PLATFORM)}\s+{_PLATFORM}")),
    ("platform_offer",
     re.compile(rf"(?<!\w)my\s+{_PLATFORM}\s*(?:is|=|:|'s)|{_PLATFORM}\s+me(?!\w)")),
    ("contact_phrase", re.compile(_any_of(CONTACT_PHRASES))),
    ("move_phrase", re.compile(_any_of(MOVE_PHRASES))),
    ("location", re.compile(_any_of(LOCATION_PHRASES))),
]


def first_match(text: str) -> str | None:
    """Name of the first rule that matches the message, or None if none does."""
    lowered = text.lower()
    for name, pattern in RULES:
        if pattern.search(lowered):
            return name
    return None


# --- scoring a dataset --------------------------------------------------------------------------


def score_split(rows_path: Path) -> pd.DataFrame:
    """One row per message: row_id, score (1.0 flagged / 0.0 not), rule (which matched)."""
    rows = pd.read_parquet(rows_path, columns=["row_id", "target_text"])
    rule = rows.target_text.map(first_match)
    return pd.DataFrame({"row_id": rows.row_id, "score": rule.notna().astype(float),
                         "rule": rule})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=Path, default=Path("data/dataset_v1/rows"))
    parser.add_argument("--out", type=Path, default=Path("outputs/regex"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        scored = score_split(args.rows / f"{split}.parquet")
        scored.to_parquet(args.out / f"{split}.parquet", index=False)
        print(f"{split}: {int(scored.score.sum()):,} of {len(scored):,} flagged")


if __name__ == "__main__":
    main()
