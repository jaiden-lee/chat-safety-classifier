import re

from modeling.data.skeletons.fill import SLOT_RE, fill_conversation, fill_skeleton, slot_errors
from modeling.schema import Conversation, Message, Provenance


def _skeleton(*messages: Message) -> Conversation:
    return Conversation("sk1", "direct", messages, Provenance("llm_skeleton", skeleton_id="sk1"))


def _pos(speaker, text, cats, tags):
    return Message(speaker, text, label=1, categories=cats, hard_tags=tags)


def _neg(speaker, text, neg_type="easy"):
    return Message(speaker, text, label=0, neg_type=neg_type)


def test_slot_errors():
    assert slot_errors("add me {HANDLE:discord} or {PHONE~2/3} or {IGN#2}") == []
    assert slot_errors("{PASSWORD}")
    assert slot_errors("{HANDLE:myspace}")
    assert slot_errors("{PHONE:discord}")  # PHONE takes no platform
    assert slot_errors("{PHONE~4/3}")
    assert slot_errors("{handle}")  # malformed: lowercase kind


def test_same_slot_same_value_and_instances_differ():
    sk = _skeleton(
        _pos("A", "its {HANDLE:discord}", ["handle"], ["plain"]),
        _pos("A", "again {HANDLE:discord} and {HANDLE:discord#2}", ["handle"], ["plain"]),
    )
    filled = fill_conversation(sk, 0)
    first = filled.messages[0].text.removeprefix("its ")
    second = filled.messages[1].text.removeprefix("again ").split(" and ")
    assert second[0] == first and second[1] != first
    assert not SLOT_RE.search(" ".join(m.text for m in filled.messages))


def test_fragments_reassemble_into_one_value():
    sk = _skeleton(
        _pos("A", "my number is", ["phone"], ["plain"]),
        _pos("A", "{PHONE~1/3}", ["phone"], ["cross_message"]),
        _pos("A", "{PHONE~2/3}", ["phone"], ["cross_message"]),
        _pos("A", "{PHONE~3/3} text me", ["phone"], ["cross_message"]),
    )
    for k in range(20):
        filled = fill_conversation(sk, k)
        frags = [filled.messages[i].text.removesuffix(" text me") for i in (1, 2, 3)]
        digits = re.sub(r"\D", "", "".join(frags))
        assert 10 <= len(digits) <= 13


def test_bang_slots_are_disguised_and_tagged():
    sk = _skeleton(_pos("A", "my snap {HANDLE:snapchat!}", ["handle"], ["plain"]))
    msg = fill_conversation(sk, 0).messages[0]
    assert "obfuscated" in msg.hard_tags and "plain" not in msg.hard_tags


def test_fill_zero_keeps_unmarked_slots_plain():
    sk = _skeleton(_pos("A", "add {HANDLE:discord}", ["handle"], ["plain"]))
    assert fill_conversation(sk, 0, obfuscation_rate=0.0).messages[0].hard_tags == ("plain",)


def test_obfuscation_rate_one_disguises_positive_contact_slots_only():
    sk = _skeleton(
        _neg("A", "gg {IGN} carried", "ingame_name_reference"),
        _pos("B", "add my {PLATFORM:discord} {HANDLE:discord}", ["handle"], ["plain"]),
    )
    filled = fill_conversation(sk, 1, obfuscation_rate=1.0)
    assert filled.messages[0].neg_type == "ingame_name_reference"
    assert filled.messages[1].hard_tags == ("obfuscated",)


def test_fills_are_deterministic_and_distinct():
    sk = _skeleton(_pos("A", "text me {PHONE}", ["phone"], ["plain"]))
    a, b = fill_skeleton(sk, 3), fill_skeleton(sk, 3)
    assert a == b
    assert len({f.messages[0].text for f in a}) == 3
    assert [f.conversation_id for f in a] == ["sk1-f0", "sk1-f1", "sk1-f2"]
