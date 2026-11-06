"""Fake-but-plausible values for the slots in generated conversation skeletons.

LLM-written skeletons contain typed slots instead of real values, e.g.
`whats ur {PLATFORM:discord}` -> `{HANDLE:discord}`. `fill.py` replaces each slot with a value
from `ValueSampler.sample(kind, arg)`. Values are plain (not obfuscated); `obfuscate.py`
transforms them afterwards.

Slot kinds (the vocabulary the generation prompts may use):

  Contact values (positives)        Look-alike values (hard negatives)
  PHONE                             IGN            in-game name (same shapes as handles)
  EMAIL                             SCORE          scores, elo, phone-length numbers
  HANDLE[:platform]                 MATCH_ID       long numeric ids
  LINK[:platform]                   COORDS         in-game coordinates
  ADDRESS   street address          LOBBY_CODE     in-game lobby/room codes
  STREET    street name             CITY, STATE, COUNTRY   coarse location
  SCHOOL                            FIRST_NAME     out of scope (negative)
  ZIP
  PLATFORM[:platform]       the platform's name as said in chat ("insta", "dc")
  PLATFORM_HINT[:platform]  a euphemism for it ("the ghost app"); implicit positives

All randomness comes from one seeded `random.Random`, so a seed reproduces every value.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

from faker import Faker


@dataclass(frozen=True)
class Platform:
    """An off-platform service: how chat refers to it, and what its handles and links look like."""

    key: str
    aliases: tuple[str, ...]  # how people name it in chat; repeats weight common names
    hints: tuple[str, ...] = ()  # euphemisms, for implicit positives
    links: tuple[str, ...] = ()  # link templates; {h} = handle, {code} = invite code, {digits}
    phone_based: bool = False  # identified by a phone number rather than a username
    max_len: int = 30


PLATFORMS: dict[str, Platform] = {p.key: p for p in [
    Platform("discord", ("discord", "disc", "dc", "discord", "cord"),
             ("the gamer chat app", "the purple app", "the app with the controller logo"),
             ("discord.gg/{code}", "https://discord.gg/{code}", "discord.com/invite/{code}"),
             max_len=32),
    Platform("snapchat", ("snap", "snapchat", "sc", "snap"),
             ("the ghost app", "the yellow app", "👻"),
             ("snapchat.com/add/{h}",), max_len=15),
    Platform("instagram", ("insta", "ig", "instagram", "gram"),
             ("the camera app", "the photo app", "📸"),
             ("instagram.com/{h}", "https://www.instagram.com/{h}"), max_len=30),
    Platform("tiktok", ("tiktok", "tt", "tik tok"), ("the clock app", "the video app"),
             ("tiktok.com/@{h}",), max_len=24),
    Platform("telegram", ("telegram", "tg", "tele"), ("the paper plane app",),
             ("t.me/{h}",), max_len=32),
    Platform("whatsapp", ("whatsapp", "wa", "whats app"), ("the green app",),
             ("wa.me/{digits}",), phone_based=True),
    Platform("signal", ("signal",), ("the encrypted app",), phone_based=True),
    Platform("kik", ("kik",), ("the k app",), max_len=20),
    Platform("x", ("twitter", "x", "twt"), ("the bird app",), ("x.com/{h}", "twitter.com/{h}"),
             max_len=15),
    Platform("twitch", ("twitch", "ttv"), ("my stream",), ("twitch.tv/{h}", "ttv/{h}"),
             max_len=25),
    Platform("youtube", ("youtube", "yt"), ("my channel",), ("youtube.com/@{h}",), max_len=30),
    Platform("reddit", ("reddit",), (), ("reddit.com/u/{h}",), max_len=20),
    Platform("facebook", ("facebook", "fb", "messenger"), ("the blue app",),
             ("facebook.com/{h}",), max_len=30),
    Platform("steam", ("steam",), (), ("steamcommunity.com/id/{h}",), max_len=32),
    Platform("xbox", ("xbox", "xbl", "gamertag"), (), max_len=15),
    Platform("psn", ("psn", "playstation", "ps"), (), max_len=16),
    Platform("epic", ("epic", "epic games"), (), max_len=16),
]}

_ADJECTIVES = ("shadow", "frost", "pixel", "ghost", "nova", "toxic", "lunar", "blaze", "cyber",
               "storm", "silent", "crimson", "neon", "dark", "lil", "big", "sweaty", "salty",
               "cosmic", "turbo", "venom", "glitch", "rogue", "astro", "midnight", "hyper")
_NOUNS = ("fox", "wolf", "byte", "ninja", "dragon", "kid", "gamer", "sniper", "panda", "viper",
          "knight", "reaper", "bean", "slayer", "beast", "fury", "hawk", "boi", "queen", "cat",
          "phantom", "noob", "legend", "rex", "storm", "ace")
_EMAIL_DOMAINS = ("gmail.com", "gmail.com", "gmail.com", "yahoo.com", "outlook.com",
                  "hotmail.com", "icloud.com", "aol.com", "proton.me")
_SCHOOL_TYPES = ("Middle School", "High School", "Elementary", "Academy", "Junior High",
                 "Middle", "High", "Prep")
# Common real street names and suffixes; Faker's generated ones ("Dakota Ports") don't read like
# something a person would type in chat.
_STREET_NAMES = ("Main", "Oak", "Maple", "Pine", "Cedar", "Elm", "Washington", "Lake", "Hill",
                 "Park", "Church", "Walnut", "Lincoln", "Sunset", "Jackson", "Highland", "Willow",
                 "River", "Mill", "Spring", "Center", "Forest", "Meadow", "Cherry", "Birch",
                 "Franklin", "Madison", "Ridge", "Valley", "Lakeview", "1st", "2nd", "3rd", "4th",
                 "5th", "6th", "10th", "Broadway", "Market", "Grand")
_STREET_SUFFIXES = ("St", "Ave", "Rd", "Dr", "Ln", "Blvd", "Ct", "Way", "Pl", "Street", "Avenue",
                    "Road", "Drive", "Lane", "st", "ave", "rd", "dr")
_SCHOOL_NAMES = ("Lincoln", "Washington", "Jefferson", "Roosevelt", "Kennedy", "Franklin",
                 "Oak Ridge", "Lakeview", "Westview", "Northside", "Eastwood", "Hillcrest",
                 "Riverside", "Central", "Valley View", "Monroe", "Madison", "Jackson",
                 "Sunset Ridge", "Pine Grove", "St. Mary's", "St. Joseph", "Liberty", "Heritage")
_CITIES = ("houston", "dallas", "austin", "chicago", "phoenix", "atlanta", "miami", "orlando",
           "denver", "seattle", "portland", "boston", "detroit", "nashville", "la", "nyc",
           "san diego", "san jose", "philly", "vegas", "toronto", "vancouver", "london",
           "manchester", "sydney", "melbourne", "columbus", "charlotte", "tampa", "sacramento")
_COUNTRIES = ("us", "usa", "america", "canada", "uk", "england", "australia", "mexico", "germany",
              "philippines", "brazil", "ireland", "new zealand")
_US_PHONE_FORMATS = ("({a}) {e}-{n}", "{a}-{e}-{n}", "{a}.{e}.{n}", "{a}{e}{n}", "{a} {e} {n}",
                     "+1 {a} {e} {n}", "1-{a}-{e}-{n}", "+1{a}{e}{n}")
_INTL_PHONE_FORMATS = ("+44 7{d3} {d6}", "+61 4{d2} {d3} {d3}", "+52 1 {d2} {d4} {d4}")


class ValueSampler:
    """Seeded source of slot values. One instance per fill keeps a conversation reproducible."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.faker = Faker("en_US")
        self.faker.seed_instance(seed)
        self._kinds = {
            "PHONE": lambda arg: self.phone(),
            "EMAIL": lambda arg: self.email(),
            "HANDLE": self.handle,
            "LINK": self.link,
            "ADDRESS": lambda arg: self.address(),
            "STREET": lambda arg: self.street(),
            "SCHOOL": lambda arg: self.school(),
            "ZIP": lambda arg: self.faker.zipcode(),
            "PLATFORM": lambda arg: self.rng.choice(self.platform(arg).aliases),
            "PLATFORM_HINT": self.platform_hint,
            "IGN": lambda arg: self.username(),
            "SCORE": lambda arg: self.score(),
            "MATCH_ID": lambda arg: self._digits(self.rng.randint(9, 12)),
            "COORDS": lambda arg: self.coords(),
            "LOBBY_CODE": lambda arg: self.lobby_code(),
            "CITY": lambda arg: self.rng.choice(_CITIES),
            "STATE": lambda arg: self.rng.choice((self.faker.state(), self.faker.state_abbr())),
            "COUNTRY": lambda arg: self.rng.choice(_COUNTRIES),
            "FIRST_NAME": lambda arg: self.faker.first_name(),
        }

    @property
    def kinds(self) -> list[str]:
        return list(self._kinds)

    def sample(self, kind: str, arg: str | None = None) -> str:
        """Value for slot `{KIND}` or `{KIND:arg}`; `arg` is a platform key where applicable."""
        if kind not in self._kinds:
            raise ValueError(f"unknown slot kind {kind!r}")
        return self._kinds[kind](arg)

    # --- platforms ---

    def platform(self, key: str | None = None) -> Platform:
        if key is None:
            return PLATFORMS[self.rng.choice(list(PLATFORMS))]
        if key not in PLATFORMS:
            raise ValueError(f"unknown platform {key!r}")
        return PLATFORMS[key]

    def platform_hint(self, key: str | None = None) -> str:
        if key is None:
            key = self.rng.choice([k for k, p in PLATFORMS.items() if p.hints])
        p = self.platform(key)
        if not p.hints:
            raise ValueError(f"platform {key!r} has no hints")
        return self.rng.choice(p.hints)

    # --- contact values ---

    def phone(self) -> str:
        """A plausible phone number, mostly US formats. Area code and exchange avoid N11 codes."""
        if self.rng.random() < 0.1:
            fmt = self.rng.choice(_INTL_PHONE_FORMATS)
            return fmt.format(d2=self._digits(2), d3=self._digits(3), d4=self._digits(4),
                              d6=self._digits(6))
        area, exchange = self._nanp_code(), self._nanp_code()
        fmt = self.rng.choice(_US_PHONE_FORMATS)
        return fmt.format(a=area, e=exchange, n=self._digits(4))

    def email(self) -> str:
        r = self.rng.random()
        if r < 0.4:
            local = self.username(style="name")
        elif r < 0.7:
            local = self.username(style="words")
        else:
            first, last = self.faker.first_name().lower(), self.faker.last_name().lower()
            local = self.rng.choice((f"{first}.{last}", f"{first}{last}{self._digits(2)}",
                                     f"{first[0]}{last}{self._digits(self.rng.randint(1, 4))}"))
        local = "".join(c for c in local.lower() if c.isalnum() or c in "._")
        return f"{local.strip('._') or 'user'}@{self.rng.choice(_EMAIL_DOMAINS)}"

    def handle(self, key: str | None = None) -> str:
        """A username in the style of the given platform (random platform if None)."""
        p = self.platform(key)
        if p.phone_based:
            return self.phone()
        if p.key == "discord":
            if self.rng.random() < 0.15:  # legacy name#1234 tags still show up in chat
                return f"{self.username(style='words', camel=True)}#{self._digits(4)}"
            return self._clean(self.username().lower(), "._", p.max_len)
        if p.key == "steam" and self.rng.random() < 0.3:
            return self._digits(self.rng.randint(8, 10))  # numeric friend code
        if p.key == "xbox":
            return self.username(style="words", camel=True, spaces=self.rng.random() < 0.3)[
                : p.max_len]
        allowed = "._-" if p.key == "snapchat" else "._"
        handle = self._clean(self.username(), allowed, p.max_len)
        if p.key == "telegram" and len(handle) < 5:
            handle += self._digits(5 - len(handle))
        return handle

    def link(self, key: str | None = None) -> str:
        """A profile, channel, or invite link for the given platform (random if None)."""
        if key is None:
            key = self.rng.choice([k for k, p in PLATFORMS.items() if p.links])
        p = self.platform(key)
        if not p.links:
            raise ValueError(f"platform {key!r} has no link format")
        template = self.rng.choice(p.links)
        handle = self.handle(key) if "{h}" in template else ""
        code = "".join(self.rng.choices(string.ascii_letters + string.digits,
                                        k=self.rng.randint(7, 10)))
        digits = "1" + self._nanp_code() + self._nanp_code() + self._digits(4)
        return template.format(h=handle, code=code, digits=digits)

    def street(self) -> str:
        return f"{self.rng.choice(_STREET_NAMES)} {self.rng.choice(_STREET_SUFFIXES)}"

    def address(self) -> str:
        number = str(self.rng.choice((self.rng.randint(1, 999), self.rng.randint(1000, 19999))))
        address = f"{number} {self.street()}"
        if self.rng.random() < 0.15:
            address += self.rng.choice((" apt ", " Apt ", " #", " unit ")) + str(
                self.rng.randint(1, 40)) + self.rng.choice(("", "", "B", "C"))
        return address

    def school(self) -> str:
        base = self.rng.choice((self.faker.last_name(), self.rng.choice(_SCHOOL_NAMES),
                                self.rng.choice(_SCHOOL_NAMES)))
        return f"{base} {self.rng.choice(_SCHOOL_TYPES)}"

    # --- look-alike values ---

    def username(self, style: str | None = None, camel: bool = False, spaces: bool = False) -> str:
        """A gamer-style username. Used for both off-platform handles and in-game names."""
        style = style or self.rng.choice(("words", "words", "name", "wrapped"))
        if style == "name":
            first = self.faker.first_name()
            parts = [first, self.rng.choice(("", self.faker.last_name()[0], "xo", "x", "the"))]
        else:
            parts = [self.rng.choice(_ADJECTIVES), self.rng.choice(_NOUNS)]
        if camel or self.rng.random() < 0.3:
            parts = [w.capitalize() for w in parts]
        sep = " " if spaces else self.rng.choice(("", "", "_", "."))
        name = sep.join(w for w in parts if w)
        if self.rng.random() < 0.6:
            name += self.rng.choice(("", "_", "")) + self.rng.choice(
                (self._digits(2), self._digits(3), str(self.rng.randint(2005, 2015))[-2:], "69",
                 "420", "1", "7"))
        if style == "wrapped":
            name = f"xX{name}Xx"
        return name

    def score(self) -> str:
        r = self.rng.random()
        if r < 0.3:
            return self._digits(10)  # phone-length number in a score context
        if r < 0.6:
            return str(self.rng.randint(800, 3200))  # elo / mmr
        if r < 0.8:
            return f"{self.rng.randint(0, 40)}-{self.rng.randint(0, 40)}"  # kills-deaths
        return f"{self.rng.randint(10_000, 9_999_999):,}"  # points with commas

    def coords(self) -> str:
        x, y, z = (self.rng.randint(-3000, 3000) for _ in range(3))
        return self.rng.choice((f"{x} {z}", f"{x} {y} {z}", f"x: {x} y: {y} z: {z}",
                                f"{x}, {z}", f"({x}, {z})"))

    def lobby_code(self) -> str:
        alphabet = string.ascii_uppercase + string.digits
        a = "".join(self.rng.choices(alphabet, k=4))
        b = "".join(self.rng.choices(alphabet, k=4))
        return self.rng.choice((a, f"{a}-{b}", a + b, a.lower()))

    # --- helpers ---

    def _digits(self, n: int) -> str:
        return "".join(self.rng.choices(string.digits, k=n))

    def _nanp_code(self) -> str:
        """Three digits valid as a North American area code or exchange: [2-9]XX, not N11."""
        while True:
            code = str(self.rng.randint(2, 9)) + self._digits(2)
            if code[1:] != "11":
                return code

    @staticmethod
    def _clean(name: str, allowed: str, max_len: int) -> str:
        cleaned = "".join(c for c in name if c.isalnum() or c in allowed)
        return cleaned[:max_len].strip("._-") or "user"
