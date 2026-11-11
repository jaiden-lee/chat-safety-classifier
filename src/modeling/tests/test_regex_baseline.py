import pandas as pd
import pytest

from modeling.baselines.regex import first_match, score_split


@pytest.mark.parametrize("text,rule", [
    ("text me 555-123-4567", "phone"),
    ("(555) 123 4567", "phone"),
    ("+1 555.123.4567", "phone"),
    ("its jake22@gmail.com", "email"),
    ("join discord.gg/k7Hq2", "link"),
    ("https://example.com/me", "link"),
    ("add me on snap", "platform_ask"),
    ("whats ur insta", "platform_ask"),
    ("Do You Have Discord?", "platform_ask"),  # lowercased first
    ("follow my twitch pls", "platform_ask"),
    ("my snap is jaydenxo", "platform_offer"),
    ("my ig: cool.kid", "platform_offer"),
    ("snap me later", "platform_offer"),
    ("whats ur number", "contact_phrase"),
    ("text me when ur home", "contact_phrase"),
    ("lets talk somewhere else", "move_phrase"),
    ("what school do u go to", "location"),
    ("i live on pine street", "location"),
])
def test_rules_catch_their_examples(text, rule):
    assert first_match(text) == rule


@pytest.mark.parametrize("text", [
    "oh snap",  # platform word alone isn't flagged
    "cold snap kid",
    "discord is down again",
    "i saw it on youtube",
    "you have epic loot",
    "add me ingame",  # on-platform friend add
    "gg 4-2-3-1 wins",  # too few digits
    "scored 1,234,567",  # commas aren't phone separators
    "i live in canada",  # coarse location is allowed
    "w ѕnaр me",  # look-alike characters defeat a plain filter (by design)
])
def test_rules_leave_ordinary_chat_alone(text):
    assert first_match(text) is None


def test_score_split_writes_one_row_per_message(tmp_path):
    path = tmp_path / "val.parquet"
    pd.DataFrame({"row_id": ["a", "b"], "target_text": ["add me on snap", "gg"]}).to_parquet(path)
    out = score_split(path)
    assert out.row_id.tolist() == ["a", "b"] and out.score.tolist() == [1.0, 0.0]
    assert out.rule[0] == "platform_ask" and pd.isna(out.rule[1])
