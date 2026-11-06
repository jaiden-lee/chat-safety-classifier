"""Checks the context builder against the shared golden cases in fixtures/.

The same file is meant to be checked by any other implementation of the builder (e.g., serving).
"""

import json
from pathlib import Path

import pytest

from modeling.context import ContextConfig, build_context, whitespace_token_count
from modeling.schema import Message

FIXTURE = Path(__file__).parents[3] / "fixtures" / "context_builder_cases.json"
SPEC = json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_uses_whitespace_counter():
    assert SPEC["token_counter"] == "whitespace"


@pytest.mark.parametrize("case", SPEC["cases"], ids=lambda c: c["name"])
def test_golden_case(case):
    ctx = build_context(
        [Message(**m) for m in case["history"]],
        Message(**case["target"]),
        case["channel_type"],
        whitespace_token_count,
        ContextConfig(**case["config"]),
    )
    assert {
        "context_text": ctx.context_text,
        "target_text": ctx.target_text,
        "total_tokens": ctx.total_tokens,
        "n_context_messages": ctx.n_context_messages,
        "n_speakers": ctx.n_speakers,
    } == case["expected"]
