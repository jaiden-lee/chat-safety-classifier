import json
import random

from modeling.data.skeletons import generate as gen
from modeling.data.skeletons.fill import SLOT_KINDS
from modeling.data.skeletons.llm_backend import LLMResult
from modeling.schema import read_conversations


def _specs(n=2):
    return gen.sample_specs(random.Random(0), n)


def _response(items) -> str:
    return "Here you go:\n```json\n" + json.dumps(items) + "\n```"


GOOD = {"spec": 1, "messages": [
    {"speaker": "A", "text": "gg", "label": 0, "neg_type": "easy"},
    {"speaker": "B", "text": "u got {PLATFORM:discord}?", "label": 1,
     "categories": ["platform_move"], "hard_tags": ["plain"]},
    {"speaker": "A", "text": "{HANDLE:discord}", "label": 1, "categories": ["handle"],
     "hard_tags": ["cross_message"]},
    {"speaker": "B", "text": "added", "label": 0, "neg_type": "post_exchange"},
]}


def _with_text(text: str) -> dict:
    item = json.loads(json.dumps(GOOD))
    item["messages"][1]["text"] = text
    return item


def test_prompt_documents_every_slot_kind_and_fills_all_placeholders():
    prompt = gen.render_prompt("skeleton_v1", _specs())
    assert "$" not in prompt.replace("$$", "")  # no unfilled Template placeholders
    for kind in SLOT_KINDS:
        assert kind in gen.SLOT_KIND_DOCS
    assert "Spec 1:" in prompt and "Spec 2:" in prompt
    assert "Labeling Guidelines" in prompt


def test_parse_valid_response_and_report_missing_spec():
    skeletons, errors, _ = gen.parse_response(_response([GOOD]), _specs(), "b0", "t1", "haiku")
    assert len(skeletons) == 1
    sk = skeletons[0]
    assert sk.conversation_id == "b0-c01" and sk.provenance.skeleton_id == "b0-c01"
    assert sk.messages[2].hard_tags == ("cross_message",)
    assert errors == ["spec 2: missing from response"]


def test_rejects_leaked_contact_info_bad_labels_and_bad_slots():
    for text, reason in [
        ("text me 415 201 8890", "phone-like"),
        ("its kayla@gmail.com", "email"),
        ("join discord.gg/abc123", "link"),
        ("add {HANDLE:myspace}", "unknown platform"),
    ]:
        skeletons, errors, _ = gen.parse_response(
            _response([_with_text(text)]), _specs(1), "b0", "t1", "haiku")
        assert skeletons == [] and reason in errors[0], (text, errors)

    bad_label = json.loads(json.dumps(GOOD))
    bad_label["messages"][0] = {"speaker": "A", "text": "gg", "label": 1}  # positive, no tags
    skeletons, errors, _ = gen.parse_response(_response([bad_label]), _specs(1), "b", "t", "m")
    assert skeletons == [] and errors


def test_non_json_response():
    skeletons, errors, _ = gen.parse_response("sorry, I can't", _specs(1), "b", "t", "m")
    assert skeletons == [] and "not a JSON array" in errors[0]


class FakeBackend:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def complete(self, prompt, *, model, system):
        self.calls += 1
        return LLMResult(self.text, "fake-model", 0.1)


def test_run_batch_writes_outputs_and_resumes(tmp_path):
    backend = FakeBackend(_response([GOOD]))
    path = gen.run_batch(backend, tmp_path, "skeleton_v1", 0, 0, 1, "haiku")
    assert path is not None and len(list(read_conversations(path))) == 1
    raw = json.loads((tmp_path / "raw" / "skeleton_v1-s0-b0000.json").read_text(encoding="utf-8"))
    assert raw["model"] == "fake-model" and raw["n_valid"] == 1
    assert gen.run_batch(backend, tmp_path, "skeleton_v1", 0, 0, 1, "haiku") is None
    assert backend.calls == 1


def test_main_model_ignores_claude_code_side_calls():
    from modeling.data.skeletons.llm_backend import _main_model
    response = {"modelUsage": {"claude-haiku-5-5": {"outputTokens": 12},
                               "claude-sonnet-5-5": {"outputTokens": 6000}}}
    assert _main_model(response, "sonnet") == "claude-sonnet-5-5"
    assert _main_model({}, "sonnet") == "sonnet"


def test_positive_specs_ask_for_several_exchanges():
    specs = gen.sample_specs(random.Random(1), 200)
    positives = [s for s in specs if s.focus == "positive"]
    assert positives and all(len(s.targets) in (2, 3) for s in positives)
    assert all(len(set(s.descriptions)) == len(s.descriptions) for s in positives)
    assert all(len(s.targets) == 1 for s in specs if s.focus == "negative")
    assert "separate exchanges" in positives[0].render()


def test_complete_phone_is_not_cross_message_but_fragment_is():
    item = json.loads(json.dumps(GOOD))
    item["messages"][1] = {"speaker": "B", "text": "sure {PHONE}", "label": 1,
                           "categories": ["phone"], "hard_tags": ["cross_message"]}
    item["messages"][2] = {"speaker": "A", "text": "{PHONE#2~1/2}", "label": 1,
                           "categories": ["phone"], "hard_tags": ["cross_message"]}
    item["messages"][3] = {"speaker": "A", "text": "{PHONE#2~2/2}", "label": 1,
                           "categories": ["phone"], "hard_tags": ["cross_message"]}
    skeletons, errors, fixes = gen.parse_response(_response([item]), _specs(1), "b", "t", "m")
    assert errors == [] and len(fixes) == 1
    msgs = skeletons[0].messages
    assert msgs[1].hard_tags == ("plain",)
    assert msgs[2].hard_tags == ("cross_message",) and msgs[3].hard_tags == ("cross_message",)


def test_failed_batches_are_skipped_not_fatal(tmp_path):
    from modeling.data.skeletons.llm_backend import LLMError

    class FlakyBackend(FakeBackend):
        def complete(self, prompt, *, model, system):
            self.calls += 1
            if self.calls == 2:
                raise LLMError("boom")
            return LLMResult(self.text, "fake-model", 0.1)

    backend = FlakyBackend(_response([GOOD]))
    failed = gen.run_batches(backend, tmp_path, "skeleton_v1", 0, 3, 1, ["haiku"])
    assert failed == [1]
    assert len(list((tmp_path / "skeletons").glob("*.jsonl"))) == 2
    assert gen.run_batches(backend, tmp_path, "skeleton_v1", 0, 3, 1, ["haiku"]) == []
    assert backend.calls == 4  # the rerun only retried batch 1


def test_usage_limit_detection():
    from modeling.data.skeletons.llm_backend import _is_usage_limit
    assert _is_usage_limit("exit 1: Claude usage limit reached. Your limit will reset at 5pm")
    assert _is_usage_limit("exit 1: rate limit exceeded")
    assert not _is_usage_limit("exit 1: non-JSON output")


def test_backend_waits_out_usage_limits_and_gives_up_on_other_errors(monkeypatch):
    import subprocess

    import pytest

    from modeling.data.skeletons import llm_backend
    ok = json.dumps({"result": "hi", "modelUsage": {"claude-sonnet-5-5": {"outputTokens": 3}}})
    limit = json.dumps({"is_error": True, "result": "Claude usage limit reached, resets at 5pm"})
    bad = json.dumps({"is_error": True, "result": "something broke"})
    outputs, sleeps = [], []

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs.pop(0), stderr="")

    monkeypatch.setattr(llm_backend.subprocess, "run", fake_run)
    monkeypatch.setattr(llm_backend.time, "sleep", sleeps.append)
    backend = llm_backend.ClaudeCLIBackend(executable="python", max_retries=2, backoff_s=1,
                                           limit_wait_s=100)

    outputs[:] = [limit, limit, ok]
    assert backend.complete("p", model="sonnet", system="s").text == "hi"
    assert sleeps == [100, 100]  # limit waits don't use up retries

    outputs[:], sleeps[:] = [bad, bad, bad], []
    with pytest.raises(llm_backend.LLMError):
        backend.complete("p", model="sonnet", system="s")
    assert sleeps == [1, 2]  # 1 try + 2 retries with exponential backoff


def _good(spec: int) -> dict:
    item = json.loads(json.dumps(GOOD))
    item["spec"] = spec
    return item


def test_salvages_conversations_around_malformed_json():
    specs = _specs(3)
    text = ("[" + json.dumps(_good(1)) + ", {\"spec\": 2, \"messages\": [ {broken ,"
            + json.dumps(_good(3)) + "]")
    skeletons, errors, _ = gen.parse_response(text, specs, "b", "t", "m")
    assert [s.conversation_id for s in skeletons] == ["b-c01", "b-c03"]
    assert any("malformed JSON" in e for e in errors)
    assert "spec 2: missing from response" in errors


def test_salvages_conversations_after_an_early_close_and_rambling():
    specs = _specs(2)
    text = "[" + json.dumps(_good(1)) + "] Wait, one more: " + json.dumps(_good(2)) + " ]"
    skeletons, errors, _ = gen.parse_response(text, specs, "b", "t", "m")
    assert len(skeletons) == 2 and errors == []


def test_lowercase_speakers_and_formations_are_accepted():
    item = _with_text("we run 4-2-3-1 this game")
    item["messages"][0]["speaker"] = "a"
    skeletons, errors, _ = gen.parse_response(_response([item]), _specs(1), "b", "t", "m")
    assert errors == [] and skeletons[0].messages[0].speaker_id == "A"


def test_batch_with_no_valid_conversations_is_retried(tmp_path):
    import pytest

    from modeling.data.skeletons.llm_backend import LLMError
    backend = FakeBackend("sorry, no JSON here")
    with pytest.raises(LLMError):
        gen.run_batch(backend, tmp_path, "skeleton_v1", 0, 0, 1, "haiku")
    assert not (tmp_path / "skeletons" / "skeleton_v1-s0-b0000.jsonl").exists()
    backend.text = _response([GOOD])
    assert gen.run_batch(backend, tmp_path, "skeleton_v1", 0, 0, 1, "haiku") is not None


def test_reparse_recovers_previously_rejected_conversations(tmp_path):
    item = _with_text("we run 4-2-3-1 this game")
    item["messages"][0]["speaker"] = "a"
    gen.run_batch(FakeBackend(_response([GOOD, _good(2) | {"messages": item["messages"]}])),
                  tmp_path, "skeleton_v1", 0, 0, 2, "haiku")
    raw_path = tmp_path / "raw" / "skeleton_v1-s0-b0000.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw.update(n_valid=1, errors=["spec 2: old stricter rule"])  # as if parsed by old code
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    gen.reparse(tmp_path)
    assert json.loads(raw_path.read_text(encoding="utf-8"))["n_valid"] == 2
    assert len(list(read_conversations(tmp_path / "skeletons" / "skeleton_v1-s0-b0000.jsonl"))) == 2
