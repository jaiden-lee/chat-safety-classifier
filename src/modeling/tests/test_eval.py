"""Evaluation tests. Each expected number is worked out by hand in the comments."""

import math
import random

import pandas as pd
import pytest

from modeling.eval import evaluate as ev
from modeling.eval.metrics import (
    Rate,
    choose_threshold,
    compute_metrics,
    flagged,
    wilson_interval,
)

# --- Wilson interval --------------------------------------------------------------------------


def test_wilson_matches_published_values():
    # 5/10: the standard textbook example, 95% Wilson interval [0.2366, 0.7634].
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    # 0/100: lower bound 0; upper bound z²/(n+z²) = 3.8416/103.8416 = 0.0370.
    low, high = wilson_interval(0, 100)
    assert low == 0.0 and high == pytest.approx(0.0370, abs=1e-4)
    # 100/100 mirrors it.
    low, high = wilson_interval(100, 100)
    assert low == pytest.approx(1 - 0.0370, abs=1e-4) and high == pytest.approx(1.0)


def test_rate_with_no_messages_is_nan():
    r = Rate(0, 0)
    assert math.isnan(r.value) and math.isnan(r.low)


# --- threshold choice -------------------------------------------------------------------------


def test_threshold_example_from_the_docstring():
    # 5 negatives, max FPR 40% -> K = floor(0.4 * 5) = 2 false blocks allowed.
    # Negatives high to low: 0.99, 0.97, 0.95 (the 3rd = K+1-th), 0.60, 0.40.
    neg = [0.99, 0.97, 0.95, 0.60, 0.40]
    pos = [0.96, 0.95, 0.50]
    scores, labels = neg + pos, [0] * 5 + [1] * 3
    t = choose_threshold(scores, labels, max_fpr=0.4)
    assert 0.95 < t < 0.9500001  # just above 0.95
    # Flagged: negatives 0.99, 0.97 (2 of 5) and positive 0.96 only (1 of 3).
    assert flagged(scores, t).tolist() == [True, True, False, False, False, True, False, False]


def test_threshold_with_tied_negatives_flags_fewer_rather_than_too_many():
    # K = floor(0.5 * 4) = 2, but three negatives tie at 0.9: flagging any of them flags all
    # three (3 > 2), so none are flagged.
    scores, labels = [0.9, 0.9, 0.9, 0.1, 0.95], [0, 0, 0, 0, 1]
    t = choose_threshold(scores, labels, max_fpr=0.5)
    assert flagged(scores, t).tolist() == [False, False, False, False, True]


def test_threshold_edge_cases():
    # Budget covers every negative -> flag everything.
    assert choose_threshold([0.1, 0.2, 0.9], [0, 0, 1], max_fpr=1.0) == -math.inf
    # Zero budget -> just above the highest negative.
    t = choose_threshold([0.1, 0.7, 0.9], [0, 0, 1], max_fpr=0.0)
    assert 0.7 < t <= 0.9
    with pytest.raises(ValueError):
        choose_threshold([0.5], [1], max_fpr=0.01)


def test_threshold_is_optimal_against_brute_force():
    # For random data, no threshold has higher recall while staying within the FPR budget.
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(5, 60)
        labels = [rng.random() < 0.3 for _ in range(n)]
        labels[0], labels[1] = 0, 1  # at least one of each
        scores = [round(rng.random(), 2) for _ in range(n)]  # rounding creates ties
        max_fpr = rng.choice([0.0, 0.05, 0.1, 0.3])
        n_neg, n_pos = labels.count(0), labels.count(1)

        def recall_fpr(t, scores=scores, labels=labels, n_neg=n_neg, n_pos=n_pos):
            f = flagged(scores, t)
            tp = sum(fi and y for fi, y in zip(f, labels, strict=True))
            fp = sum(fi and not y for fi, y in zip(f, labels, strict=True))
            return tp / n_pos, fp / n_neg

        t = choose_threshold(scores, [int(y) for y in labels], max_fpr)
        recall, fpr = recall_fpr(t)
        assert fpr <= max_fpr + 1e-12
        candidates = [-math.inf, *sorted(set(scores)), math.inf]
        best = max(r for r, f in map(recall_fpr, candidates) if f <= max_fpr + 1e-12)
        assert recall == best


# --- the four metrics -------------------------------------------------------------------------


def test_compute_metrics_counts_each_group():
    #   score label hard_tags                   neg_type          flagged at 0.5
    rows = [
        (0.9, 1, ["plain"], None),  # positive                    yes
        (0.8, 1, ["obfuscated"], None),  # hard positive          yes
        (0.2, 1, ["cross_message", "obfuscated"], None),  # hard   no  (one tag or two: once)
        (0.3, 1, ["implicit"], None),  # positive, not Q2-hard     no
        (0.7, 0, [], "number_lookalike"),  # look-alike             yes
        (0.1, 0, [], "app_topic"),  # look-alike                    no
        (0.6, 0, [], "easy"),  # negative                           yes
        (0.0, 0, [], "post_exchange"),  # negative, not look-alike   no
    ]
    scores, labels, tags, negs = zip(*rows, strict=True)
    m = compute_metrics(scores, labels, tags, negs, threshold=0.5)
    assert (m.recall.k, m.recall.n) == (2, 4)  # 0.9, 0.8
    assert (m.fpr.k, m.fpr.n) == (2, 4)  # 0.7, 0.6
    assert (m.hard_recall.k, m.hard_recall.n) == (1, 2)  # obfuscated yes, cross+obf no
    assert (m.lookalike_fpr.k, m.lookalike_fpr.n) == (1, 2)  # number_lookalike yes
    assert m.passes() == {"Q1 recall": False, "Q1 FPR": False,
                          "Q2 hard-positive recall": False, "Q3 look-alike FPR": False}


# --- the script, end to end -------------------------------------------------------------------


def _write(tmp_path, n_pos=100, n_neg=2000):
    """A tiny dataset where the model is perfect except for 1 high-scoring negative, and the
    baseline catches only the plain positives (half of them) and flags 20 negatives."""
    rows, model, base = [], [], []
    for i in range(n_pos):
        hard = i % 2 == 1
        rows.append((f"p{i}", 1, ["obfuscated"] if hard else ["plain"], None))
        model.append((f"p{i}", 0.5 + i / (4 * n_pos)))  # 0.5 .. 0.75
        base.append((f"p{i}", 0.0 if hard else 1.0))
    for i in range(n_neg):
        rows.append((f"n{i}", 0, [], "number_lookalike" if i < 100 else "easy"))
        model.append((f"n{i}", 0.99 if i == 0 else i / (10 * n_neg)))  # one bad negative
        base.append((f"n{i}", 1.0 if i < 20 else 0.0))
    for split in ("val", "test"):
        for name, data, cols in (("rows", rows, ["row_id", "label", "hard_tags", "neg_type"]),
                                 ("model", model, ["row_id", "score"]),
                                 ("base", base, ["row_id", "score"])):
            (tmp_path / name).mkdir(exist_ok=True)
            pd.DataFrame(data, columns=cols).to_parquet(tmp_path / name / f"{split}.parquet")


def test_evaluate_end_to_end(tmp_path):
    _write(tmp_path)
    result = ev.evaluate(tmp_path / "rows", tmp_path / "model", tmp_path / "base", "test")
    m = result["model"]
    # FPR budget 0.1% of 2,000 negatives = 2 false blocks: the 0.99 negative (a look-alike)
    # and the highest easy negative (0.09995). Every positive scores >= 0.5, so all are caught.
    assert (m.recall.k, m.recall.n) == (100, 100)
    assert (m.fpr.k, m.fpr.n) == (2, 2000)
    assert (m.hard_recall.k, m.hard_recall.n) == (50, 50)
    assert (m.lookalike_fpr.k, m.lookalike_fpr.n) == (1, 100)
    # Baseline: 50/100 recall, 20/2000 = 1% FPR. Model at 1% FPR: still 100/100.
    assert (result["baseline"].recall.k, result["baseline"].fpr.k) == (50, 20)
    assert result["baseline_val_fpr"] == pytest.approx(0.01)
    assert result["recall_gap"] == pytest.approx(0.5)
    assert result["hard_recall_gap"] == pytest.approx(1.0)
    report = ev.to_markdown(result)
    assert "| Gap (Q4: >= +0.15) |  | +0.500 (pass) | +1.000 |" in report
    ev.to_json(result)  # serializable


def test_evaluate_rejects_missing_scores(tmp_path):
    _write(tmp_path)
    scores = pd.read_parquet(tmp_path / "model" / "val.parquet")
    scores.iloc[1:].to_parquet(tmp_path / "model" / "val.parquet")
    with pytest.raises(ValueError, match="1 rows without a score"):
        ev.evaluate(tmp_path / "rows", tmp_path / "model", None, "val")
