"""Dataset audit: sizing targets (design doc §6.1), slice coverage (§6.6), leakage, rule checks,
and a stratified manual-review sample. No LLM judge: labels come from generation, and this
audit checks them with rules plus a human-reviewed sample (labeling_guidelines.md intro).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pandas as pd

from modeling.data.assembly.conda import might_be_contact_info
from modeling.data.assembly.split import SPLITS
from modeling.eval.metrics import Q2_HARD_TAGS, Q3_NEG_TYPES
from modeling.schema import Conversation

TARGETS = {
    "test_positives_min": 2000,
    "test_negatives_min": 50000,
    "eval_positive_rate": (0.10, 0.20),  # val and test (enriched)
    "test_q2_slice_min": 300,  # per Q2 hard tag
    "test_q3_negatives_min": 2000,
}

TOKEN_BUCKETS = [0, 32, 64, 128, 256, 512]
TOKEN_LABELS = ["1-32", "33-64", "65-128", "129-256", "257-512"]


def audit(
    rows: dict[str, pd.DataFrame],
    skeletons: dict[str, list[Conversation]],
    build_stats: dict,
) -> dict:
    """All audit numbers as a JSON-able dict."""
    report: dict = {"build": build_stats, "splits": {}}
    for split in SPLITS:
        df = rows[split]
        pos, neg = df[df.label == 1], df[df.label == 0]
        report["splits"][split] = {
            "conversations": int(df.conversation_id.nunique()),
            "rows": len(df),
            "positives": len(pos),
            "negatives": len(neg),
            "positive_rate": round(len(pos) / max(len(df), 1), 4),
            "rows_by_source": _counts(df.source),
            "positives_by_source": _counts(pos.source),
            "positives_by_hard_tag": _counts(pos.hard_tags.explode()),
            "positives_by_category": _counts(pos.categories.explode()),
            "negatives_by_type": _counts(neg.neg_type),
            "rows_by_tokens": _counts(pd.cut(df.total_tokens, TOKEN_BUCKETS, labels=TOKEN_LABELS)),
            "rows_by_speakers": _counts(df.n_speakers.clip(upper=3).map(
                {1: "1", 2: "2", 3: "3+"})),
            "rows_by_channel": _counts(df.channel_type),
        }
    report["targets"] = _check_targets(report["splits"])
    report["leakage"] = _leakage(rows, skeletons)
    report["rule_checks"] = _rule_checks(rows)
    return report


def _counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.dropna().value_counts().sort_index().items()}


def _check_targets(splits: dict) -> list[dict]:
    test = splits["test"]
    checks = [
        ("test positives", test["positives"], f">= {TARGETS['test_positives_min']}",
         test["positives"] >= TARGETS["test_positives_min"]),
        ("test negatives", test["negatives"], f">= {TARGETS['test_negatives_min']}",
         test["negatives"] >= TARGETS["test_negatives_min"]),
    ]
    lo, hi = TARGETS["eval_positive_rate"]
    for split in ("val", "test"):
        rate = splits[split]["positive_rate"]
        checks.append((f"{split} positive rate", rate, f"{lo:.0%}-{hi:.0%}", lo <= rate <= hi))
    for tag in sorted(Q2_HARD_TAGS):
        n = test["positives_by_hard_tag"].get(tag, 0)
        checks.append((f"test {tag} positives (Q2)", n, f">= {TARGETS['test_q2_slice_min']}",
                       n >= TARGETS["test_q2_slice_min"]))
    n_q3 = sum(test["negatives_by_type"].get(t, 0) for t in Q3_NEG_TYPES)
    checks.append(("test look-alike negatives (Q3)", n_q3,
                   f">= {TARGETS['test_q3_negatives_min']}",
                   n_q3 >= TARGETS["test_q3_negatives_min"]))
    return [{"check": c, "value": v, "target": t, "ok": bool(ok)} for c, v, t, ok in checks]


def _leakage(rows: dict[str, pd.DataFrame], skeletons: dict[str, list[Conversation]]) -> dict:
    out: dict = {}
    groups = {s: set(rows[s].skeleton_id.dropna()) | set(
        rows[s].conversation_id[rows[s].source == "conda"]) for s in SPLITS}
    out["group_overlap"] = {f"{a}/{b}": len(groups[a] & groups[b])
                            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}

    # Exact model inputs seen in train. Unavoidable for short generic chat ("gg" with no
    # context); worth watching for positives.
    train_inputs = set(zip(rows["train"].context_text, rows["train"].target_text, strict=True))
    for split in ("val", "test"):
        df = rows[split]
        seen = pd.Series([k in train_inputs for k in zip(df.context_text, df.target_text,
                                                           strict=True)], index=df.index)
        out[f"{split}_inputs_seen_in_train"] = {
            "positives": round(float(seen[df.label == 1].mean()), 4),
            "negatives": round(float(seen[df.label == 0].mean()), 4),
        }

    # Near-duplicate skeletons across splits: max word-5-gram Jaccard to any train skeleton.
    for split in ("val", "test"):
        best = _max_jaccard(skeletons["train"], skeletons[split])
        out[f"{split}_skeleton_near_duplicates"] = {
            "jaccard>=0.5": int(sum(b >= 0.5 for b in best)),
            "jaccard>=0.8": int(sum(b >= 0.8 for b in best)),
            "of": len(best),
        }
    return out


def _shingles(conv: Conversation, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\{[^}]+\}|\w+", " ".join(m.text for m in conv.messages).lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _max_jaccard(reference: list[Conversation], queries: list[Conversation]) -> list[float]:
    ref = [_shingles(c) for c in reference]
    index: dict[tuple, list[int]] = defaultdict(list)
    for i, sh in enumerate(ref):
        for s in sh:
            index[s].append(i)
    best = []
    for conv in queries:
        sh = _shingles(conv)
        shared = Counter(i for s in sh for i in index.get(s, ()))
        best.append(max((c / (len(sh) + len(ref[i]) - c) for i, c in shared.items()),
                        default=0.0))
    return best


def _rule_checks(rows: dict[str, pd.DataFrame]) -> dict:
    """Negatives whose text looks like contact info (URL, email, 7+ digits), by type.
    Expected for number_lookalike (scores, match ids); suspicious elsewhere."""
    out = {}
    for split in SPLITS:
        neg = rows[split][rows[split].label == 0]
        flagged = neg[neg.target_text.map(might_be_contact_info)]
        out[f"{split}_contact_like_negatives"] = _counts(flagged.neg_type)
    return out


# --- manual review ----------------------------------------------------------------------------

REVIEW_POSITIVES = {"obfuscated": 30, "cross_message": 30, "implicit": 30, "plain": 25}
REVIEW_NEGATIVES = {t: 10 for t in ("number_lookalike", "ingame_social", "app_topic",
                                    "ingame_name_reference", "coarse_location", "refusal_warning",
                                    "post_exchange", "easy")}
REVIEW_SOURCES = {"conda": 10, "splice": 15}
REVIEW_SUSPICIOUS = 20  # negatives (other than number_lookalike) that look like contact info


def _review_strata(d: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """(name, row mask, n) for each stratum of the manual-review sample."""
    strata = [(f"pos/{tag}", (d.label == 1) & d.hard_tags.map(lambda t, tag=tag: tag in t), n)
              for tag, n in REVIEW_POSITIVES.items()]
    strata += [(f"neg/{t}", (d.label == 0) & (d.neg_type == t) & (d.source != "conda"), n)
               for t, n in REVIEW_NEGATIVES.items()]
    strata += [(src, d.source == src, n) for src, n in REVIEW_SOURCES.items()]
    strata.append(("suspicious negative", (d.label == 0) & (d.neg_type != "number_lookalike")
                   & d.target_text.map(might_be_contact_info), REVIEW_SUSPICIOUS))
    return strata


def manual_review_sample(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A stratified sample for human label review, with empty reviewer columns. Drawn from val:
    test is never looked at."""
    parts, taken = [], set()
    for name, mask, n in _review_strata(df):
        pool = df[mask & ~df.row_id.isin(taken)]
        pick = pool.sample(n=min(n, len(pool)), random_state=seed)
        taken |= set(pick.row_id)
        parts.append(pick.assign(stratum=name))
    out = pd.concat(parts)
    out = out.assign(categories=out.categories.map(",".join),
                     hard_tags=out.hard_tags.map(",".join),
                     reviewer_label="", reviewer_note="")
    return out[["stratum", "row_id", "source", "label", "categories", "hard_tags", "neg_type",
                "context_text", "target_text", "reviewer_label", "reviewer_note"]]


# --- markdown ---------------------------------------------------------------------------------


def to_markdown(report: dict) -> str:
    s = report["splits"]
    lines = ["# Dataset audit", ""]
    lines += ["## Sizing targets (design doc §6.1)", "", "| Check | Value | Target | OK |",
              "|---|---|---|---|"]
    for c in report["targets"]:
        value = f"{c['value']:.1%}" if isinstance(c["value"], float) else f"{c['value']:,}"
        lines.append(f"| {c['check']} | {value} | {c['target']} | {'yes' if c['ok'] else 'NO'} |")

    lines += ["", "## Splits", "", "| | " + " | ".join(SPLITS) + " |",
              "|---|" + "---|" * len(SPLITS)]
    for key in ("conversations", "rows", "positives", "negatives", "positive_rate"):
        vals = [f"{s[x][key]:.1%}" if key == "positive_rate" else f"{s[x][key]:,}"
                for x in SPLITS]
        lines.append(f"| {key} | " + " | ".join(vals) + " |")
    for key in ("rows_by_source", "positives_by_source", "positives_by_hard_tag",
                "positives_by_category", "negatives_by_type", "rows_by_tokens",
                "rows_by_speakers", "rows_by_channel"):
        lines += ["", f"**{key.replace('_', ' ')}**", "", "| | " + " | ".join(SPLITS) + " |",
                  "|---|" + "---|" * len(SPLITS)]
        names = list(dict.fromkeys(k for x in SPLITS for k in s[x][key]))
        if key != "rows_by_tokens":
            names.sort()
        for name in names:
            lines.append(f"| {name} | " + " | ".join(f"{s[x][key].get(name, 0):,}"
                                                     for x in SPLITS) + " |")

    lines += ["", "## Leakage", ""]
    for k, v in report["leakage"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Rule checks", "",
              ("Negatives whose target looks like contact info (URL, email, 7+ digits), by type. "
              "Expected for number_lookalike; any other type is worth a look "
              "(manual_review.csv, stratum 'suspicious negative')."), ""]
    for k, v in report["rule_checks"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Build", ""]
    for k, v in report["build"].items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"
