from modeling.context import (
    SPECIAL_TOKENS,
    ContextConfig,
    build_context,
    build_context_at,
    whitespace_token_count,
)
from modeling.schema import Conversation, Message, Provenance

count = whitespace_token_count


def _msgs(*pairs, ts_step=1.0):
    return [Message(s, t, ts=i * ts_step) for i, (s, t) in enumerate(pairs)]


def test_empty_history_gives_channel_token_only():
    ctx = build_context([], Message("a", "hi"), "direct", count)
    assert ctx.context_text == "[DM]"
    assert ctx.target_text == "hi"
    assert ctx.n_context_messages == 0
    assert ctx.n_speakers == 1


def test_tags_are_relative_to_target_author_and_chronological():
    history = _msgs(("b", "whats ur discord"), ("a", "why"), ("c", "just tell him"))
    ctx = build_context(history, Message("a", "shadowfox_22", ts=3.0), "group", count)
    assert ctx.context_text == "[GROUP] [U1] whats ur discord [SELF] why [U2] just tell him"
    assert ctx.n_speakers == 3


def test_same_history_gets_different_tags_for_a_different_author():
    history = _msgs(("b", "whats ur discord"), ("a", "why"))
    ctx = build_context(history, Message("b", "just curious", ts=2.0), "direct", count)
    assert ctx.context_text == "[DM] [SELF] whats ur discord [U1] why"


def test_speakers_beyond_nine_share_uother():
    history = _msgs(*[(f"s{i}", "hey") for i in range(11)])
    ctx = build_context(history, Message("self", "hi", ts=11.0), "group", count)
    assert "[U9] hey" in ctx.context_text
    assert ctx.context_text.count("[UOTHER]") == 2
    assert all(t in SPECIAL_TOKENS for t in ctx.context_text.split() if t.startswith("["))


def test_budget_keeps_most_recent_contiguous_window():
    history = _msgs(("b", "one two three"), ("b", "four five"), ("c", "six"))
    # overhead 3 + [DM] 1 + target 1 = 5; "[X] six" = 2, "[X] four five" = 3 -> 10; next is 4.
    cfg = ContextConfig(token_budget=12)
    ctx = build_context(history, Message("a", "hi", ts=3.0), "direct", count, cfg)
    assert ctx.n_context_messages == 2
    assert ctx.total_tokens == 10
    assert ctx.total_tokens <= cfg.token_budget


def test_stops_at_first_message_that_does_not_fit():
    # A long message blocks everything older than it, even messages that would fit.
    history = _msgs(("a", "my number is"), ("b", "word " * 30), ("b", "ok"))
    cfg = ContextConfig(token_budget=15)
    ctx = build_context(history, Message("a", "415"), "direct", count, cfg)
    assert ctx.n_context_messages == 1
    assert ctx.context_text == "[DM] [U1] ok"


def test_lookback_message_count_limit():
    history = _msgs(*[("b", f"m{i}") for i in range(10)])
    cfg = ContextConfig(lookback_messages=3)
    ctx = build_context(history, Message("a", "hi", ts=10.0), "direct", count, cfg)
    assert ctx.context_text == "[DM] [U1] m7 [U1] m8 [U1] m9"


def test_build_context_at_uses_messages_before_target():
    conv = Conversation(
        "c1", "direct",
        tuple(_msgs(("b", "whats ur discord"), ("a", "shadowfox_22"), ("b", "added"))),
        Provenance("llm_skeleton"),
    )
    ctx = build_context_at(conv, 1, count)
    assert ctx.context_text == "[DM] [U1] whats ur discord"
    assert ctx.target_text == "shadowfox_22"
