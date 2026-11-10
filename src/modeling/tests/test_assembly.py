import csv
import random

import pytest

from modeling.context import whitespace_token_count
from modeling.data.assembly import audit, conda, splice, split
from modeling.data.assembly.rows import conversation_rows, dedupe
from modeling.schema import Conversation, Message, Provenance, Source

FIELDS = ["Id", "utterance", "tokenized", "intentClass", "slotClasses", "matchId",
          "conversationId", "chatTime"]


def _write_conda(tmp_path, rows):
    for i, name in enumerate(conda.CONDA_FILES):
        with (tmp_path / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            for r in rows[i::3]:
                w.writerow({k: r.get(k, "") for k in FIELDS})


def _row(i, match, t, text):
    return {"Id": i, "utterance": text, "matchId": match, "chatTime": t}


def test_conda_matches_become_group_conversations(tmp_path):
    rows = [_row(1, 7, 30, "gg [SEPA] wp"), _row(2, 7, 10, "hi all"), _row(3, 7, 20, "mid"),
            _row(4, 7, 40, "ez"), _row(5, 8, 5, "join twitch.tv/somebody"), _row(6, 8, 6, "no")]
    _write_conda(tmp_path, rows)
    convs, stats = conda.load_matches(tmp_path, seed=1)
    assert stats == {"matches": 2, "dropped_contact_like": 1, "dropped_reviewed_positive": 0,
                     "kept": 1}
    (conv,) = convs
    assert conv.conversation_id == "conda-m7" and conv.channel_type == "group"
    assert [m.text for m in conv.messages] == ["hi all", "mid", "gg", "wp", "ez"]
    assert all(m.label == 0 and m.neg_type == "easy" for m in conv.messages)
    sp = [m.speaker_id for m in conv.messages]
    assert sp[2] == sp[3]  # one utterance's lines share a speaker
    assert sp[0] != sp[1] and sp[1] != sp[2] and sp[3] != sp[4]  # adjacent utterances differ
    assert conda.load_matches(tmp_path, seed=1)[0] == convs  # deterministic


@pytest.mark.parametrize("text,expected", [
    ("www.leagueoflegends.com/register", True), ("mail me a@b.com", True),
    ("call 555 123 4567", True), ("fb", False), ("4-0-4-0", False), ("10 8 200", False),
])
def test_might_be_contact_info(text, expected):
    assert conda.might_be_contact_info(text) is expected


def _msg(speaker, text, label=0):
    if label:
        return Message(speaker, text, label=1, categories=("handle",), hard_tags=("plain",))
    return Message(speaker, text, label=0, neg_type="easy")


def _conv(cid, messages, source=Source.LLM_SKELETON, skeleton_id=None):
    return Conversation(cid, "group", tuple(messages),
                        Provenance(source=source, skeleton_id=skeleton_id, template_id="t"))


def test_splice_inserts_the_exchange_with_mapped_speakers():
    donor = _conv("sk-f3", [_msg("A", f"chat {i}") for i in range(5)]
                  + [_msg("B", "add my snap", 1), _msg("A", "ok"), _msg("A", "later"),
                     _msg("A", "bye")], skeleton_id="sk")
    host = _conv("conda-m1", [_msg(f"P{i % 3 + 1}", f"real {i}") for i in range(6)],
                 source=Source.CONDA)
    out = splice.splice(host, donor, random.Random(0))
    inserted = [m for m in out.messages if not m.text.startswith("real")]
    assert [m.text for m in inserted] == ["chat 3", "chat 4", "add my snap", "ok"]
    assert [m.label for m in inserted] == [0, 0, 1, 0]
    assert {m.speaker_id for m in inserted} <= {"P1", "P2", "P3"}
    assert len({m.speaker_id for m in inserted}) == 2  # A and B stay distinct
    assert [m.text for m in out.messages if m.text.startswith("real")] == [
        m.text for m in host.messages]
    assert out.provenance.source == "splice" and out.provenance.skeleton_id == "sk"


def test_splice_needs_a_positive():
    with pytest.raises(ValueError):
        splice.exchange_span(_conv("c", [_msg("A", "hi")]))


def test_split_is_deterministic_and_respects_fractions():
    fr = {"train": 0.6, "val": 0.2, "test": 0.2}
    got = [split.assign_split(f"k{i}", fr) for i in range(5000)]
    assert got == [split.assign_split(f"k{i}", fr) for i in range(5000)]
    assert abs(got.count("train") / 5000 - 0.6) < 0.03
    with pytest.raises(ValueError):
        split.assign_split("k", {"train": 0.5})


def test_split_key_groups_fills_of_one_skeleton():
    a = _conv("sk1-f0", [_msg("A", "x")], skeleton_id="sk1")
    b = _conv("splice-conda-m3-sk1-f3", [_msg("A", "x")], Source.SPLICE, skeleton_id="sk1")
    c = _conv("conda-m3", [_msg("A", "x")], Source.CONDA)
    assert split.split_key(a) == split.split_key(b) == "sk1"
    assert split.split_key(c) == "conda-m3"


def test_rows_and_dedupe():
    f0 = _conv("sk-f0", [_msg("A", "hey"), _msg("B", "snap me", 1)], skeleton_id="sk")
    f1 = _conv("sk-f1", [_msg("A", "hey"), _msg("B", "snap me", 1)], skeleton_id="sk")
    rows = conversation_rows(f0, whitespace_token_count) + conversation_rows(
        f1, whitespace_token_count)
    assert [r.row_id for r in rows[:2]] == ["sk-f0:0", "sk-f0:1"]
    assert rows[1].context_text == "[GROUP] [U1] hey" and rows[1].label == 1
    kept, stats = dedupe(rows)
    assert [r.row_id for r in kept] == ["sk-f0:0", "sk-f0:1"]
    assert stats["duplicates_dropped"] == 2

    clash = conversation_rows(_conv("x", [_msg("A", "hey"), _msg("B", "snap me")]),
                              whitespace_token_count)
    kept, stats = dedupe(rows + clash)
    assert [r.row_id for r in kept] == ["sk-f0:0"]
    assert stats["label_conflict_inputs"] == 1 and stats["label_conflict_rows_dropped"] == 3


def test_near_duplicate_detection():
    words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    ref = [_conv("r", [_msg("A", " ".join(words))])]
    same = _conv("q1", [_msg("B", " ".join(words))])
    other = _conv("q2", [_msg("B", "completely different words in this chat line")])
    assert audit._max_jaccard(ref, [same, other]) == [1.0, 0.0]
