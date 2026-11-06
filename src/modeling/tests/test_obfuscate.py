import random
import re

import pytest

from modeling.data.primitives.obfuscate import TRANSFORMS, applicable, obfuscate, split_value
from modeling.data.primitives.values import ValueSampler


def _alnum(s: str) -> str:
    return re.sub(r"[\W_]", "", s)


@pytest.fixture
def rng():
    return random.Random(0)


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_each_transform_changes_values_it_applies_to(name, rng):
    values = ValueSampler(1)
    pool = [values.phone() for _ in range(20)] + [values.email() for _ in range(20)] + [
        values.handle() for _ in range(40)]
    tried = 0
    for v in pool:
        if name in applicable(v):
            tried += 1
            out = obfuscate(v, rng, [name])
            assert out.text != v and out.transforms == (name,)
    assert tried > 0


@pytest.mark.parametrize("value", ["kayla.m09@gmail.com", "discord.gg/k7Hq2Lp"])
def test_emails_and_links_only_get_safe_transforms(value):
    assert set(applicable(value)) <= {"at_dot", "homoglyph", "separators"}


def test_separators_on_emails_and_links_are_spaces(rng):
    for _ in range(50):
        for value in ("kayla.m09@gmail.com", "discord.gg/k7Hq2Lp"):
            out = obfuscate(value, rng, ["separators"]).text
            assert out.replace(" ", "") == value, out


def test_separators_on_phone_drop_punctuation_and_keep_digits(rng):
    for _ in range(50):
        out = obfuscate("(415) 201-8890", rng, ["separators"]).text
        assert re.sub(r"\D", "", out) == "4152018890"
        assert "(" not in out and ")" not in out


def test_separator_gaps_vary_within_and_across_values():
    widths = set()
    for i in range(200):
        out = obfuscate("shadowfox", random.Random(i), ["separators"]).text
        widths.update(len(m) for m in re.findall(r" +", out))
    assert {1, 2, 3} <= widths


def test_one_spoken_form_for_zero_within_a_value():
    for i in range(200):
        out = obfuscate("4000100", random.Random(i), ["spelled_digits"]).text
        zero_forms = set(re.findall(r"zero|oh|\bo\b", out))
        assert len(zero_forms) <= 1, out
        assert _decode_spelled(out) == "4000100", out


def test_inapplicable_transform_rejected(rng):
    with pytest.raises(ValueError):
        obfuscate("shadowfox", rng, ["spelled_digits"])  # no digits
    with pytest.raises(ValueError):
        obfuscate("shadowfox", rng, ["at_dot"])


_WORD_TO_DIGIT = {"zero": "0", "three": "3", "seven": "7", "eight": "8", "four": "4", "five": "5",
                  "nine": "9", "one": "1", "two": "2", "six": "6", "oh": "0", "o": "0"}


def _decode_spelled(text: str) -> str:
    pattern = "|".join(_WORD_TO_DIGIT)  # longest words first, so "four" wins over "o"
    return re.sub(r"[ \-]", "", re.sub(pattern, lambda m: _WORD_TO_DIGIT[m.group()], text))


def test_spelled_digits_on_phone_decodes_back_to_the_digits(rng):
    for _ in range(200):
        out = obfuscate("(415) 201-8890", rng, ["spelled_digits"]).text
        assert re.search("[a-z]", out), out  # at least one digit spelled
        assert _decode_spelled(out) == "4152018890", out


def test_at_dot_removes_symbols_from_email(rng):
    for _ in range(50):
        out = obfuscate("kayla.m09@gmail.com", rng, ["at_dot"]).text
        assert "@" not in out.replace(" @ ", "") or " @ " in out
        assert "gmail" in out


def test_homoglyph_looks_the_same_but_differs(rng):
    out = obfuscate("charlie", rng, ["homoglyph"]).text
    assert out != "charlie" and len(out) == len("charlie")
    assert not out.isascii()


def test_random_obfuscation_is_reproducible_and_never_identity():
    values = ValueSampler(3)
    pool = [values.sample(k) for k in ("PHONE", "EMAIL", "HANDLE", "LINK") for _ in range(50)]
    a = [obfuscate(v, random.Random(i)) for i, v in enumerate(pool)]
    b = [obfuscate(v, random.Random(i)) for i, v in enumerate(pool)]
    assert a == b
    assert all(o.text != v and 1 <= len(o.transforms) <= 2 for o, v in zip(a, pool, strict=True))


@pytest.mark.parametrize("value,n,expected", [
    ("4152018890", 3, ["415", "201", "8890"]),
    ("415-201-8890", 3, ["415", "201", "8890"]),
    ("(415) 201-8890", 3, None),
])
def test_split_phone_on_digit_groups(value, n, expected, rng):
    parts = split_value(value, n, rng)
    assert len(parts) == n
    assert _alnum("".join(parts)) == _alnum(value)
    if expected:
        assert parts == expected


def test_split_preserves_content_and_count():
    values = ValueSampler(5)
    for i in range(200):
        v = values.sample(["PHONE", "EMAIL", "HANDLE"][i % 3])
        n = 2 + i % 2
        if len(_alnum(v)) < 2 * n:
            continue
        parts = split_value(v, n, random.Random(i))
        assert len(parts) == n and all(parts)
        assert _alnum("".join(parts)) == _alnum(v), (v, parts)


def test_split_rejects_bad_requests(rng):
    with pytest.raises(ValueError):
        split_value("abc", 1, rng)
    with pytest.raises(ValueError):
        split_value("ab", 3, rng)
