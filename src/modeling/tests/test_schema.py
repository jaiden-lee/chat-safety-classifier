import pytest

from modeling.schema import (
    Conversation,
    Message,
    Provenance,
    ScoringRow,
    read_conversations,
    read_rows,
    write_conversations,
    write_rows,
)


def _conversation() -> Conversation:
    return Conversation(
        conversation_id="c1",
        channel_type="direct",
        messages=(
            Message("a", "whats ur discord", ts=1.0, label=1,
                    categories=["platform_move"], hard_tags=["plain"]),
            Message("b", "shadowfox_22", ts=2.0, label=1,
                    categories=["handle"], hard_tags=["cross_message"]),
            Message("a", "ok", ts=3.0, label=0, neg_type="post_exchange"),
        ),
        provenance=Provenance("llm_skeleton", skeleton_id="s1", template_id="t1",
                              generator_model="haiku"),
    )


@pytest.mark.parametrize("kwargs", [
    {"label": 2, "neg_type": "easy"},
    {"label": 1},  # positive without category or tag
    {"label": 1, "categories": ["handle"], "hard_tags": ["plain", "obfuscated"]},
    {"label": 1, "categories": ["handle"], "hard_tags": ["plain"], "neg_type": "easy"},
    {"label": 0},  # negative without neg_type
    {"label": 0, "categories": ["handle"], "neg_type": "easy"},
    {"label": None, "neg_type": "easy"},  # unlabeled with tags
    {"label": 1, "categories": ["not_a_category"], "hard_tags": ["plain"]},
])
def test_invalid_labels_rejected(kwargs):
    with pytest.raises(ValueError):
        Message("a", "hi", **kwargs)


def test_unlabeled_message_allowed():
    assert Message("a", "hi").label is None


def test_conversation_jsonl_round_trip(tmp_path):
    conv = _conversation()
    path = tmp_path / "convs.jsonl"
    assert write_conversations(path, [conv]) == 1
    assert list(read_conversations(path)) == [conv]


def test_speakers_in_first_appearance_order():
    assert _conversation().speakers == ["a", "b"]


def test_rows_parquet_round_trip(tmp_path):
    rows = [
        ScoringRow("c1:1", "c1", 1, "direct", "[DM] [U1] whats ur discord", "shadowfox_22", 1,
                   categories=["handle"], hard_tags=["cross_message"], total_tokens=8,
                   n_context_messages=1, n_speakers=2, source="llm_skeleton", skeleton_id="s1"),
        ScoringRow("c1:2", "c1", 2, "direct", "[DM] ...", "ok", 0, neg_type="post_exchange"),
    ]
    path = tmp_path / "rows.parquet"
    assert write_rows(path, rows) == 2
    assert read_rows(path) == rows
