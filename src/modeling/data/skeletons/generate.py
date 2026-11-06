"""Generate labeled conversation skeletons with an LLM (design doc §6.1, guidelines).

One batch = one LLM call that writes `per_batch` conversations, each from a sampled spec
(channel type, speakers, length, game, chat style, and what the conversation should contain).
Responses are validated message by message; conversations that break a rule are dropped and
the reason is recorded. Each batch writes:

  <out>/skeletons/<batch_id>.jsonl   valid skeletons, as `Conversation`s with slots in the text
  <out>/raw/<batch_id>.json          prompt specs, raw response, model, timing, rejections

Batches whose skeleton file already exists are skipped, so an interrupted run can be resumed.

    uv run python -m modeling.data.skeletons.generate --out data/pilot --batches 2 --models haiku,sonnet
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template

from modeling.data.primitives.values import PLATFORMS
from modeling.data.skeletons.fill import SLOT_RE, slot_errors
from modeling.data.skeletons.llm_backend import ClaudeCLIBackend, LLMBackend
from modeling.schema import (
    ChannelType,
    Conversation,
    Message,
    Provenance,
    Source,
    write_conversations,
)

PROMPT_DIR = Path(__file__).parent / "prompts"
GUIDELINES_PATH = Path(__file__).parents[4] / "docs" / "labeling_guidelines.md"  # repo root

SYSTEM_PROMPT = (
    "You generate synthetic chat data for training a child-safety classifier. Follow the "
    "instructions exactly and output only the requested JSON."
)

SLOT_KIND_DOCS = """\
  PHONE, EMAIL, HANDLE:<platform>, LINK:<platform> (profile, channel, or server invite),
  ADDRESS (street address), STREET (street name), SCHOOL, ZIP,
  PLATFORM:<platform> (the app's name as said in chat), PLATFORM_HINT:<platform> (a euphemism),
  IGN (in-game name), SCORE, MATCH_ID, COORDS (in-game coordinates), LOBBY_CODE,
  CITY, STATE, COUNTRY, FIRST_NAME"""

GAMES = (
    "a battle royale shooter", "a block-building sandbox", "a team-based hero shooter",
    "a mobile MOBA", "a fantasy MMO", "a soccer game", "a co-op horror game",
    "a parkour obstacle-course game", "a kart racing game", "a collectible card battler",
    "a social hangout game", "a tactical FPS", "a farming and life sim", "a survival crafting game",
)
STYLES = (
    "casual younger teens, lots of slang and typos",
    "competitive tryhards, short callouts, some trash talk",
    "friendly and chatty, lots of emotes and lol",
    "minimal punctuation, everything lowercase, very short messages",
    "mixed: one chatty player, others terse",
    "chill older players, full words but casual",
)

# What the conversation is about. Positive focuses list the category and hard tag the
# interesting messages should get; negative focuses are look-alike clean conversations.
POSITIVE_FOCUSES = (
    ("handle", "plain", "a player shares their handle on another app, naming the app"),
    ("handle", "cross_message", ("one player asks another for their handle on an app; the other "
                                "replies in a later message with just the handle")),
    ("handle", "implicit", ("a player points others to their account on another app without "
                           "naming it (a euphemism, or 'same name as here')")),
    ("handle", "obfuscated", ("a player shares a handle and the app name disguised to dodge the "
                             "filter (mark those slots with !)")),
    ("handle", "plain", ("a player posts a server invite or profile link, or promotes their own "
                        "stream or channel")),
    ("handle", "plain", "a player shares a friend's or sibling's handle on another app"),
    ("phone", "plain", "a player shares a phone number or asks to be texted"),
    ("phone", "cross_message", ("a player sends a phone number split across 2-3 messages, using "
                               "fragment slots")),
    ("phone", "obfuscated", ("a player shares a phone number disguised to dodge the filter "
                            "(mark the slot with !)")),
    ("phone", "implicit", "a player asks for a number using slang, without saying 'phone'"),
    ("email", "plain", "a player shares an email address to send something"),
    ("email", "obfuscated", "a player shares an email disguised to dodge the filter (mark with !)"),
    ("location", "plain", "a player mentions their school or the street they live on"),
    ("location", "cross_message", ("a player asks what school or street another lives on, and "
                                  "gets an answer later")),
    ("location", "plain", "a player shares a ZIP code or suggests meeting up in person"),
    ("platform_move", "plain", "a player suggests moving the chat to a named app"),
    ("platform_move", "implicit", ("a player suggests talking somewhere more private, without "
                                  "naming an app")),
)
NEGATIVE_FOCUSES = (
    ("number_lookalike", ("players talk about scores, elo, match ids, or coordinates "
                         "(use SCORE, MATCH_ID, COORDS slots); no contact info")),
    ("ingame_social", ("players add each other in game, invite to parties, or share lobby codes "
                      "(use IGN, LOBBY_CODE slots); everything stays in game")),
    ("app_topic", ("players mention other apps only as topics (an app being down, a video they "
                  "saw, the game's official account); nobody shares or asks for contact info")),
    ("ingame_name_reference", "players talk about other players by in-game name (IGN slots)"),
    ("coarse_location", ("players say what country, state, city, or timezone they're in; nothing "
                        "more specific")),
    ("refusal_warning", ("someone asks for contact info and is refused, or players warn each "
                        "other not to share it (the request itself is still positive)")),
    ("easy", "ordinary game chat with nothing resembling contact info"),
)
POSITIVE_SHARE = 0.6


@dataclass(frozen=True)
class Spec:
    """What one conversation in a batch should look like."""

    index: int  # 1-based position in the batch
    channel_type: str
    n_speakers: int
    n_messages: int
    game: str
    style: str
    focus: str  # "positive" or "negative"
    target: str  # category/hard_tag for positives, neg_type for negatives
    description: str

    def render(self) -> str:
        channel = "one-to-one DM" if self.channel_type == ChannelType.DIRECT else "group chat"
        return (f"Spec {self.index}: {channel}, {self.n_speakers} speakers, about "
                f"{self.n_messages} messages. Game: {self.game}. Style: {self.style}.\n"
                f"  Include: {self.description}. Target labels: {self.target}.")


def sample_specs(rng: random.Random, n: int) -> list[Spec]:
    specs = []
    for i in range(1, n + 1):
        direct = rng.random() < 0.35
        if rng.random() < POSITIVE_SHARE:
            category, hard_tag, description = rng.choice(POSITIVE_FOCUSES)
            focus, target = "positive", f"{category}/{hard_tag}"
        else:
            neg_type, description = rng.choice(NEGATIVE_FOCUSES)
            focus, target = "negative", neg_type
        specs.append(Spec(
            index=i,
            channel_type=ChannelType.DIRECT if direct else ChannelType.GROUP,
            n_speakers=2 if direct else rng.randint(3, 6),
            n_messages=rng.randint(6, 25),
            game=rng.choice(GAMES),
            style=rng.choice(STYLES),
            focus=focus,
            target=target,
            description=description,
        ))
    return specs


def render_prompt(template_id: str, specs: list[Spec]) -> str:
    template = Template((PROMPT_DIR / f"{template_id}.txt").read_text(encoding="utf-8"))
    return template.substitute(
        n_conversations=len(specs),
        slot_kinds=SLOT_KIND_DOCS,
        platforms=", ".join(PLATFORMS),
        specs="\n\n".join(s.render() for s in specs),
        guidelines=GUIDELINES_PATH.read_text(encoding="utf-8"),
    )


# Contact details the LLM was told never to write itself (outside slots).
_LEAK_PATTERNS = [
    (re.compile(r"\d[\d\s().-]{5,}\d"), "phone-like digits"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE), "email address"),
    (re.compile(r"(discord\.gg|t\.me|wa\.me|instagram\.com|snapchat\.com|twitch\.tv|"
                r"tiktok\.com|youtube\.com|x\.com|twitter\.com)/\S+", re.IGNORECASE), "link"),
]


def parse_response(
    text: str, specs: list[Spec], batch_id: str, template_id: str, model: str
) -> tuple[list[Conversation], list[str]]:
    """Valid skeletons from an LLM response, plus one error string per rejected conversation."""
    try:
        items = json.loads(_json_array(text))
    except (ValueError, json.JSONDecodeError) as e:
        return [], [f"response is not a JSON array: {e}"]
    by_index = {s.index: s for s in specs}
    skeletons, errors, seen = [], [], set()
    for item in items:
        spec = by_index.get(item.get("spec") if isinstance(item, dict) else None)
        if spec is None or spec.index in seen:
            errors.append(f"unknown or repeated spec in item: {str(item)[:100]}")
            continue
        seen.add(spec.index)
        try:
            skeletons.append(_to_skeleton(item, spec, batch_id, template_id, model))
        except ValueError as e:
            errors.append(f"spec {spec.index}: {e}")
    errors += [f"spec {i}: missing from response" for i in sorted(set(by_index) - seen)]
    return skeletons, errors


def _to_skeleton(item: dict, spec: Spec, batch_id: str, template_id: str,
                 model: str) -> Conversation:
    raw_messages = item.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) < 4:
        raise ValueError("fewer than 4 messages")
    messages = []
    for j, m in enumerate(raw_messages):
        text, speaker = m.get("text"), m.get("speaker")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"message {j}: empty text")
        if not isinstance(speaker, str) or not re.fullmatch(r"[A-Z]", speaker):
            raise ValueError(f"message {j}: bad speaker {speaker!r}")
        if errs := slot_errors(text):
            raise ValueError(f"message {j}: {'; '.join(errs)}")
        outside_slots = SLOT_RE.sub(" ", text)
        for pattern, what in _LEAK_PATTERNS:
            if pattern.search(outside_slots):
                raise ValueError(f"message {j}: {what} written outside a slot: {text!r}")
        messages.append(Message(
            speaker_id=speaker, text=text, label=m.get("label"),
            categories=m.get("categories") or (), hard_tags=m.get("hard_tags") or (),
            neg_type=m.get("neg_type"),
        ))  # Message validates the label and tags
    return Conversation(
        conversation_id=f"{batch_id}-c{spec.index:02d}",
        channel_type=spec.channel_type,
        messages=tuple(messages),
        provenance=Provenance(Source.LLM_SKELETON, skeleton_id=f"{batch_id}-c{spec.index:02d}",
                              template_id=template_id, generator_model=model),
    )


def _json_array(text: str) -> str:
    """The outermost JSON array in `text`, tolerating code fences or stray prose."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array found")
    return text[start:end + 1]


def run_batch(
    backend: LLMBackend, out_dir: Path, template_id: str, seed: int, batch_index: int,
    per_batch: int, model: str,
) -> Path | None:
    """Generate one batch unless its output already exists. Returns the skeleton file path."""
    batch_id = f"{template_id}-s{seed}-b{batch_index:04d}"
    skeleton_path = out_dir / "skeletons" / f"{batch_id}.jsonl"
    if skeleton_path.exists():
        print(f"{batch_id}: exists, skipping")
        return None
    specs = sample_specs(random.Random(f"{template_id}:{seed}:{batch_index}"), per_batch)
    prompt = render_prompt(template_id, specs)
    result = backend.complete(prompt, model=model, system=SYSTEM_PROMPT)
    skeletons, errors = parse_response(result.text, specs, batch_id, template_id, result.model)

    raw_path = out_dir / "raw" / f"{batch_id}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({
        "batch_id": batch_id, "template_id": template_id, "seed": seed,
        "requested_model": model, "model": result.model, "duration_s": result.duration_s,
        "specs": [asdict(s) for s in specs], "n_valid": len(skeletons), "errors": errors,
        "response": result.text,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_conversations(skeleton_path, skeletons)
    print(f"{batch_id}: {len(skeletons)}/{len(specs)} valid ({result.model}, "
          f"{result.duration_s:.0f}s)" + "".join(f"\n  - {e}" for e in errors))
    return skeleton_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--per-batch", type=int, default=10)
    parser.add_argument("--models", default="haiku", help="comma-separated, cycled per batch")
    parser.add_argument("--template", default="skeleton_v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    backend = ClaudeCLIBackend()
    models = args.models.split(",")
    for b in range(args.batches):
        run_batch(backend, args.out, args.template, args.seed, b, args.per_batch,
                  models[b % len(models)])

if __name__ == "__main__":
    main()
