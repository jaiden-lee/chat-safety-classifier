"""Filter-evasion transforms applied to slot values (guidelines §5.2, `obfuscated` tag).

`skeletons/fill.py` calls `obfuscate(value, rng)` on a value from `primitives/values.py` when a message is meant to
be an obfuscated positive. The result records which transforms were applied. `split_value`
breaks a value into fragments for information split across messages (`cross_message`).

Transforms, applied in this fixed order when combined:

  at_dot          kayla@gmail.com     -> kayla at gmail dot com     (emails, links, dotted handles)
  spelled_digits  415 201 8890        -> four one five two oh one   (not emails or links)
  leet            shadowfox           -> sh4d0wf0x                  (not emails or links)
  homoglyph       charlie             -> сhаrlіе      (Cyrillic / full-width look-alikes)
  keycap          415                 -> 4️⃣1️⃣5️⃣                     (not emails or links)
  separators      discord             -> d i  s    c o  rd  /  4.1..5.2...0
                                         (random gap per position; emails/links get spaces only)

Every function takes a seeded `random.Random`, so a seed reproduces the output.
"""

from __future__ import annotations

import itertools
import random
import re
from collections.abc import Callable
from dataclasses import dataclass

# Spoken forms of each digit. "o" (for zero) is only used when words are separated, since
# "nineoone" is unreadable.
_DIGIT_WORDS = {
    "0": ("zero", "oh", "o"), "1": ("one",), "2": ("two",), "3": ("three",), "4": ("four",),
    "5": ("five",), "6": ("six",), "7": ("seven",), "8": ("eight",), "9": ("nine",),
}

_LEET = {"a": "4@", "e": "3", "i": "1!", "o": "0", "s": "5$", "t": "7", "l": "1", "g": "9",
         "b": "8", "z": "2"}

_HOMOGLYPHS = {
    "a": "а", "c": "сⅽ", "e": "е", "i": "і", "j": "ј", "o": "о", "p": "р", "s": "ѕ", "x": "х",
    "y": "у", "l": "ⅼ", "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х",
    **{str(d): chr(0xFF10 + d) for d in range(10)},  # full-width digits
}

_AT_FORMS = (" at ", "[at]", "(at)", " @ ", " AT ")
_DOT_FORMS = (" dot ", "[dot]", "(dot)", " d0t ", " DOT ")
# Space is listed several times so most separator obfuscation uses spaces.
_SEPARATORS = (" ", " ", " ", " ", ".", "-", "_", "*", "|", "/", "~", ",")


@dataclass(frozen=True)
class Obfuscated:
    """An obfuscated value and the names of the transforms that produced it."""

    text: str
    transforms: tuple[str, ...]


@dataclass(frozen=True)
class Transform:
    """One evasion technique: when it applies, and how to apply it."""

    name: str
    applies: Callable[[str], bool]
    apply: Callable[[str, random.Random], str]


# --- individual transforms ---


def at_dot(value: str, rng: random.Random) -> str:
    """Spell out '@' and '.'; sometimes drop a trailing '.com' (kayla at gmail)."""
    if value.endswith(".com") and "@" in value and rng.random() < 0.3:
        value = value[: -len(".com")]
    at, dot = rng.choice(_AT_FORMS), rng.choice(_DOT_FORMS)
    return value.replace("@", at).replace(".", dot).strip()


def spelled_digits(value: str, rng: random.Random) -> str:
    """Write digits as words. Phone-like values drop their punctuation; others keep letters."""
    sep = rng.choice((" ", " ", " ", "-", ""))
    # One spoken form per digit for the whole value, so a number reads consistently.
    forms = {d: rng.choice([w for w in words if sep or w != "o"])
             for d, words in _DIGIT_WORDS.items()}
    mixed = rng.random() < 0.3  # spell only some digits: "4 one 5"

    def word(d: str) -> str:
        return forms[d]

    if _is_numeric(value):
        digits = [c for c in value if c.isdigit()]
        spell = _choose_mask(len(digits), 0.5 if mixed else 1.0, rng)
        return sep.join(word(d) if s else d for d, s in zip(digits, spell, strict=True))
    # Mixed letters and digits (e.g., a handle): replace each digit run with words.
    return re.sub(r"\d+", lambda m: " " + " ".join(word(d) for d in m.group()) + " ",
                  value).strip()


def leet(value: str, rng: random.Random) -> str:
    """Replace about half of the eligible letters with look-alike digits or symbols."""
    eligible = [i for i, c in enumerate(value) if c.lower() in _LEET]
    chars = list(value)
    for i in _choose_subset(eligible, 0.5, rng):
        chars[i] = rng.choice(_LEET[chars[i].lower()])
    return "".join(chars)


def homoglyph(value: str, rng: random.Random) -> str:
    """Replace about half of the eligible characters with visually identical Unicode ones."""
    eligible = [i for i, c in enumerate(value) if c in _HOMOGLYPHS]
    chars = list(value)
    for i in _choose_subset(eligible, 0.5, rng):
        chars[i] = rng.choice(_HOMOGLYPHS[chars[i]])
    return "".join(chars)


def keycap(value: str, rng: random.Random) -> str:
    """Write digits as keycap emoji (4️⃣). Phone-like values drop their punctuation."""
    if _is_numeric(value):
        value = "".join(c for c in value if c.isdigit())
    return re.sub(r"\d", lambda m: m.group() + "️⃣", value)


def separators(value: str, rng: random.Random) -> str:
    """Break the value into chunks with a separator between them (d i  s c  o rd, 4.1..5.2).

    One separator character per value (space most often), repeated a random number of times
    at each gap. Chunks are single characters, or in 30% of values 1-3 characters each.
    Phone-like values drop their own punctuation first; emails and links only get spaces,
    since symbols inside them would be confused with their own '.' and '/'.
    """
    if _is_numeric(value):
        chars = [c for c in value if c.isdigit()]
    else:
        chars = [c for c in value if not c.isspace()]
    sep = " " if _is_address(value) else rng.choice(_SEPARATORS)
    max_chunk = 3 if rng.random() < 0.3 else 1

    chunks, i = [], 0
    while i < len(chars):
        size = rng.randint(1, max_chunk)
        chunks.append("".join(chars[i:i + size]))
        i += size
    if len(chunks) == 1:  # never return the value as one unbroken chunk
        chunks = [chars[0], "".join(chars[1:])]

    out = chunks[0]
    for chunk in chunks[1:]:
        out += sep * _gap_width(sep, rng) + chunk
    return out


def _gap_width(sep: str, rng: random.Random) -> int:
    """How many separator characters go in one gap. Spaces vary more than symbols."""
    if sep == " ":
        return rng.choices((1, 2, 3, 4, 5), weights=(60, 20, 10, 6, 4))[0]
    return rng.choices((1, 2, 3), weights=(75, 20, 5))[0]


def _has_digits(v: str) -> bool:
    return any(c.isdigit() for c in v)


def _is_address(v: str) -> bool:
    """Emails and links: transforms that rewrite characters inside them would break them."""
    return "@" in v or "/" in v


TRANSFORMS: dict[str, Transform] = {t.name: t for t in [
    Transform("at_dot", lambda v: "@" in v or "." in v, at_dot),
    Transform("spelled_digits", lambda v: _has_digits(v) and not _is_address(v), spelled_digits),
    Transform("leet", lambda v: not _is_address(v) and any(c.lower() in _LEET for c in v), leet),
    Transform("homoglyph", lambda v: any(c in _HOMOGLYPHS for c in v), homoglyph),
    Transform("keycap", lambda v: _has_digits(v) and not _is_address(v), keycap),
    Transform("separators", lambda v: len(v.replace(" ", "")) >= 3, separators),
]}

# Pairs that combine badly: spelled-out and keycap digits replace the same characters,
# separators on top of spelled words or keycaps just scatter letters and emoji, and leet plus
# separators is unreadable ("$*n**a***p").
_EXCLUSIVE = {frozenset(p) for p in [
    ("spelled_digits", "keycap"), ("spelled_digits", "separators"), ("keycap", "separators"),
    ("homoglyph", "keycap"), ("leet", "separators"),
]}
TWO_TRANSFORM_RATE = 0.15


def applicable(value: str) -> list[str]:
    """Names of the transforms that can change `value`."""
    return [name for name, t in TRANSFORMS.items() if t.applies(value)]


def obfuscate(
    value: str, rng: random.Random, transforms: list[str] | None = None, max_transforms: int = 2
) -> Obfuscated:
    """Apply `transforms` in canonical order, or random applicable ones (usually one; two at
    TWO_TRANSFORM_RATE, never an _EXCLUSIVE pair).

    Explicit transforms must all apply. Random picks skip a second transform that no longer
    applies after the first. Raises ValueError if the result would equal the input.
    """
    if transforms is not None:
        result = _apply(value, transforms, rng, strict=True)
        if result.text == value:
            raise ValueError(f"obfuscation left {value!r} unchanged")
        return result
    for _ in range(5):
        result = _apply(value, _pick(value, rng, max_transforms), rng, strict=False)
        if result.text != value:
            return result
    raise ValueError(f"couldn't change {value!r}")


def split_value(value: str, n_parts: int, rng: random.Random) -> list[str]:
    """Split a value into `n_parts` fragments for sending across several messages.

    Cuts prefer natural break points (phone digit groups, '@', '.', '_', '-', spaces), and
    fragments drop the punctuation at their edges, the way people type them: '415-201-8890'
    -> ['415', '201', '8890']. The fragments' letters and digits, in order, are the value's.
    """
    if n_parts < 2:
        raise ValueError("n_parts must be at least 2")
    v = value.strip()
    if v.isdigit():
        natural = _digit_group_cuts(len(v))  # 4152018890 -> 415 | 201 | 8890
    else:
        natural = [i for i, c in enumerate(v) if c in _SPLIT_CHARS and 0 < i < len(v)]
    for cuts in (natural, list(range(2, len(v) - 1))):
        if len(cuts) < n_parts - 1:
            continue
        for _ in range(20):
            bounds = [0, *sorted(rng.sample(cuts, n_parts - 1)), len(v)]
            parts = [v[a:b].strip(_SPLIT_CHARS) for a, b in itertools.pairwise(bounds)]
            if all(len(p) >= 1 for p in parts):
                return parts
    raise ValueError(f"{value!r} can't be split into {n_parts} parts")


_SPLIT_CHARS = " ._-()"


# --- helpers ---


def _apply(value: str, names: list[str], rng: random.Random, strict: bool) -> Obfuscated:
    order = list(TRANSFORMS)
    text, applied = value, []
    for name in sorted(names, key=order.index):
        if not TRANSFORMS[name].applies(text):
            if strict:
                raise ValueError(f"transform {name!r} doesn't apply to {text!r}")
            continue
        text = TRANSFORMS[name].apply(text, rng)
        applied.append(name)
    return Obfuscated(text, tuple(applied))


def _pick(value: str, rng: random.Random, max_transforms: int) -> list[str]:
    candidates = applicable(value)
    if not candidates:
        raise ValueError(f"no transform applies to {value!r}")
    k = 2 if max_transforms >= 2 and rng.random() < TWO_TRANSFORM_RATE else 1
    first = rng.choice(candidates)
    picked = [first]
    if k == 2:
        rest = [c for c in candidates if c != first and frozenset((c, first)) not in _EXCLUSIVE]
        if rest:
            picked.append(rng.choice(rest))
    return picked


def _is_numeric(value: str) -> bool:
    """True if the value's letters/digits are all digits (phone numbers, codes)."""
    alnum = [c for c in value if c.isalnum()]
    return bool(alnum) and all(c.isdigit() for c in alnum)


def _digit_group_cuts(n: int) -> list[int]:
    if n == 10:
        return [3, 6]
    if n == 11:
        return [1, 4, 7]
    return list(range(3, n, 3))


def _choose_subset(items: list[int], p: float, rng: random.Random) -> list[int]:
    """Each item with probability p; never empty if items is non-empty."""
    chosen = [i for i in items if rng.random() < p]
    return chosen or ([rng.choice(items)] if items else [])


def _choose_mask(n: int, p: float, rng: random.Random) -> list[bool]:
    mask = [rng.random() < p for _ in range(n)]
    if n and not any(mask):
        mask[rng.randrange(n)] = True
    return mask
