import re

import pytest

from modeling.data.primitives.values import PLATFORMS, ValueSampler

N = 300  # samples per property check


@pytest.fixture
def vs():
    return ValueSampler(seed=0)


def test_same_seed_same_values():
    a, b = ValueSampler(42), ValueSampler(42)
    kinds = a.kinds
    assert [a.sample(k) for k in kinds * 5] == [b.sample(k) for k in kinds * 5]


def test_different_seeds_differ():
    assert [ValueSampler(1).handle() for _ in range(5)] != [ValueSampler(2).handle()
                                                           for _ in range(5)]


def test_every_kind_samples_a_nonempty_string(vs):
    for kind in vs.kinds:
        for _ in range(20):
            value = vs.sample(kind)
            assert isinstance(value, str) and value.strip(), kind


def test_unknown_kind_and_platform_rejected(vs):
    with pytest.raises(ValueError):
        vs.sample("PASSWORD")
    with pytest.raises(ValueError):
        vs.sample("HANDLE", "myspace")
    with pytest.raises(ValueError):
        vs.sample("LINK", "kik")  # kik has no link format


def test_phone_digit_counts(vs):
    for _ in range(N):
        digits = re.sub(r"\D", "", vs.phone())
        assert 10 <= len(digits) <= 13


def test_us_phone_avoids_n11_codes(vs):
    for _ in range(N):
        phone = vs.phone()
        if phone.startswith(("+44", "+61", "+52")):
            continue
        digits = re.sub(r"\D", "", phone)[-10:]
        area, exchange = digits[:3], digits[3:6]
        for code in (area, exchange):
            assert code[0] in "23456789" and code[1:] != "11"


def test_email_shape(vs):
    pattern = re.compile(r"^[a-z0-9][a-z0-9._]*@[a-z]+\.[a-z]+$")
    for _ in range(N):
        assert pattern.match(vs.email()), vs.email()


@pytest.mark.parametrize("key", [k for k, p in PLATFORMS.items() if not p.phone_based])
def test_handles_respect_platform_length(vs, key):
    for _ in range(50):
        h = vs.handle(key)
        if key == "discord" and "#" in h:  # legacy tag: name#1234
            assert re.fullmatch(r".+#\d{4}", h)
            continue
        assert len(h) <= PLATFORMS[key].max_len


def test_modern_discord_handles_are_lowercase(vs):
    for _ in range(N):
        h = vs.handle("discord")
        if "#" not in h:
            assert re.fullmatch(r"[a-z0-9._]+", h), h


def test_phone_based_platforms_give_phone_numbers(vs):
    for key in ("whatsapp", "signal"):
        assert len(re.sub(r"\D", "", vs.handle(key))) >= 10


def test_links_match_their_platform(vs):
    for _ in range(50):
        assert vs.link("discord").split("//")[-1].startswith(("discord.gg/", "discord.com/invite/"))
        assert re.fullmatch(r"wa\.me/1\d{10}", vs.link("whatsapp"))


def test_platform_names_and_hints(vs):
    assert vs.sample("PLATFORM", "instagram") in PLATFORMS["instagram"].aliases
    assert vs.sample("PLATFORM_HINT", "snapchat") in PLATFORMS["snapchat"].hints


def test_lobby_codes_are_not_phone_shaped(vs):
    for _ in range(N):
        assert len(re.sub(r"\D", "", vs.lobby_code())) < 10
