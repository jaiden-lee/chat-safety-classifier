"""Score files in, the two report tables out (design doc §6.2-6.5). The math is in metrics.py.

    uv run python -m modeling.eval.evaluate --model outputs/bert_v1 --baseline outputs/regex \
        --split val --out results/p1a/bert_v1

Inputs
- `--rows`: the dataset's `{val,test}.parquet` (labels and tags), from the assembly build.
- `--model` / `--baseline`: directories with `{val,test}.parquet`, each with one row per
  dataset row and columns `row_id`, `score`. For the model the score is its probability; the
  regex baseline is yes/no, so its score is 1.0 (flagged) or 0.0.

Steps
1. Model threshold T: chosen on **val**, highest recall with FPR <= 0.1%.
2. Model report (Table 2): the four metrics on `--split`, at T.
3. If a baseline is given (Table 1):
   - the baseline flags a message when its score >= 0.5 (it has no other setting);
   - its FPR on val, f, fixes the comparison point;
   - model threshold T_f: chosen on val, highest recall with FPR <= f;
   - on `--split`: baseline vs. model at T_f, and the recall gap (Q4: >= +0.15).

Thresholds always come from val. `--split val` reports on val too, for development (slightly
optimistic, since the threshold was fit there). `--split test` is the one-time final
measurement (design doc §6.4): run it once, after the model is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from modeling.eval.metrics import (
    Q1_MAX_FPR,
    Q1_MIN_RECALL,
    Q2_MIN_RECALL,
    Q3_MAX_FPR,
    Q4_MIN_RECALL_GAP,
    Metrics,
    Rate,
    choose_threshold,
    compute_metrics,
)

BASELINE_THRESHOLD = 0.5  # a yes/no baseline scores 1.0 or 0.0


def load_scored(rows_dir: Path, scores_dir: Path, split: str) -> pd.DataFrame:
    """The split's rows with a `score` column. Every row must have exactly one score."""
    rows = pd.read_parquet(rows_dir / f"{split}.parquet",
                           columns=["row_id", "label", "hard_tags", "neg_type"])
    scores = pd.read_parquet(scores_dir / f"{split}.parquet", columns=["row_id", "score"])
    if scores.row_id.duplicated().any():
        raise ValueError(f"{scores_dir}/{split}: duplicate row_ids in scores")
    merged = rows.merge(scores, on="row_id", how="left", validate="one_to_one")
    missing = merged.score.isna().sum()
    extra = len(set(scores.row_id) - set(rows.row_id))
    if missing or extra:
        raise ValueError(f"{scores_dir}/{split}: {missing} rows without a score, "
                         f"{extra} scores for unknown rows")
    return merged


def metrics_at(df: pd.DataFrame, threshold: float) -> Metrics:
    return compute_metrics(df.score, df.label, df.hard_tags, df.neg_type, threshold)


def fpr_threshold(df: pd.DataFrame, max_fpr: float) -> float:
    return choose_threshold(df.score, df.label, max_fpr)


def evaluate(rows_dir: Path, model_dir: Path, baseline_dir: Path | None, split: str) -> dict:
    """All numbers for the report, as a dict (also written as metrics.json)."""
    model_val = load_scored(rows_dir, model_dir, "val")
    model_eval = model_val if split == "val" else load_scored(rows_dir, model_dir, split)

    # 1-2. Model at its deployed threshold.
    threshold = fpr_threshold(model_val, Q1_MAX_FPR)
    out = {"split": split, "model": metrics_at(model_eval, threshold)}

    # 3. Baseline comparison at the baseline's FPR.
    if baseline_dir is not None:
        base_val = load_scored(rows_dir, baseline_dir, "val")
        base_eval = base_val if split == "val" else load_scored(rows_dir, baseline_dir, split)
        base_val_fpr = metrics_at(base_val, BASELINE_THRESHOLD).fpr.value
        model_at_base = metrics_at(model_eval, fpr_threshold(model_val, base_val_fpr))
        baseline = metrics_at(base_eval, BASELINE_THRESHOLD)
        out["baseline"] = baseline
        out["model_at_baseline_fpr"] = model_at_base
        out["baseline_val_fpr"] = base_val_fpr
        out["recall_gap"] = model_at_base.recall.value - baseline.recall.value
        out["hard_recall_gap"] = model_at_base.hard_recall.value - baseline.hard_recall.value
    return out


# --- formatting -------------------------------------------------------------------------------


def _recall(r: Rate) -> str:
    return f"{r.value:.3f} [{r.low:.3f}, {r.high:.3f}]"


def _fpr(r: Rate) -> str:
    return f"{r.value:.2%} [{r.low:.2%}, {r.high:.2%}]"


def _count(r: Rate) -> str:
    return f"{r.k:,} / {r.n:,}"


def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def to_markdown(result: dict) -> str:
    m: Metrics = result["model"]
    ok = {name: "pass" if passed else "FAIL" for name, passed in m.passes().items()}
    lines = [
        f"# Evaluation on {result['split']}",
        "",
        f"## Model report (threshold {m.threshold:.6g}, set on val at FPR <= {Q1_MAX_FPR:.1%})",
        "",
        _row("Metric", "Target", "Result [95% CI]", "Flagged / total", ""),
        _row("---", "---", "---", "---", "---"),
        _row("Recall, all positives (Q1, primary)", f">= {Q1_MIN_RECALL}",
             _recall(m.recall), _count(m.recall), ok["Q1 recall"]),
        _row("FPR, all clean messages (Q1)", f"<= {Q1_MAX_FPR:.1%}",
             _fpr(m.fpr), _count(m.fpr), ok["Q1 FPR"]),
        _row("Recall, hard positives (Q2)", f">= {Q2_MIN_RECALL}",
             _recall(m.hard_recall), _count(m.hard_recall), ok["Q2 hard-positive recall"]),
        _row("FPR, look-alike negatives (Q3)", f"<= {Q3_MAX_FPR:.1%}",
             _fpr(m.lookalike_fpr), _count(m.lookalike_fpr), ok["Q3 look-alike FPR"]),
    ]
    if "baseline" in result:
        b: Metrics = result["baseline"]
        mb: Metrics = result["model_at_baseline_fpr"]
        gap, hard_gap = result["recall_gap"], result["hard_recall_gap"]
        q4 = "pass" if gap >= Q4_MIN_RECALL_GAP else "FAIL"
        lines += [
            "",
            (f"## Baseline comparison (model threshold {mb.threshold:.6g}, set on val to match "
             f"the baseline's val FPR of {result['baseline_val_fpr']:.2%})"),
            "",
            _row("", "FPR", "Recall, all positives", "Recall, hard positives"),
            _row("---", "---", "---", "---"),
            _row("Regex baseline", _fpr(b.fpr), _recall(b.recall), _recall(b.hard_recall)),
            _row("Model at the baseline's FPR", _fpr(mb.fpr), _recall(mb.recall),
                 _recall(mb.hard_recall)),
            _row(f"Gap (Q4: >= +{Q4_MIN_RECALL_GAP})", "", f"{gap:+.3f} ({q4})",
                 f"{hard_gap:+.3f}"),
        ]
    return "\n".join(lines) + "\n"


def to_json(result: dict) -> dict:
    """Metrics objects -> plain dicts, with each rate's value and interval spelled out."""
    def rate(r: Rate) -> dict:
        return {"k": r.k, "n": r.n, "value": r.value, "low": r.low, "high": r.high}

    def metrics(m: Metrics) -> dict:
        return {
            "threshold": m.threshold,
            **{k: rate(getattr(m, k)) for k in ("recall", "fpr", "hard_recall", "lookalike_fpr")},
            "passes": m.passes(),
        }

    return {k: metrics(v) if isinstance(v, Metrics) else v for k, v in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=Path, default=Path("data/dataset_v1/rows"))
    parser.add_argument("--model", type=Path, required=True, help="model scores directory")
    parser.add_argument("--baseline", type=Path, help="baseline scores directory")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--out", type=Path, help="write report.md and metrics.json here")
    args = parser.parse_args()

    result = evaluate(args.rows, args.model, args.baseline, args.split)
    report = to_markdown(result)
    print(report)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"report_{args.split}.md").write_text(report, encoding="utf-8")
        (args.out / f"metrics_{args.split}.json").write_text(
            json.dumps(to_json(result), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
