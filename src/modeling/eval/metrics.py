"""The evaluation math (design doc §6.2-6.5). Pure functions: no file reading, no printing.

Everything here is a count. A classifier gives each message a score; a message is
**flagged** when `score >= threshold`. Then, for a group of messages:

    recall = flagged positives / positives         "what share of contact sharing we catch"
    FPR    = flagged negatives / negatives         "what share of clean messages we block"

Each is reported with a 95% Wilson interval (`Rate`).

The four reported metrics (design doc §6.2), all at one threshold:

    metric                                  group of messages                         target
    recall, all positives        (Q1)       label == 1                                >= 0.90
    FPR, all clean messages      (Q1)       label == 0                                <= 0.1%
    recall, hard positives       (Q2)       label == 1 and obfuscated/cross_message   >= 0.75
    FPR, look-alike negatives    (Q3)       label == 0 and neg_type in Q3_NEG_TYPES   <= 0.5%

The threshold is chosen on validation by `choose_threshold` and then applied, unchanged, to
test. For the baseline comparison (Q4), the model's threshold is instead chosen to match the
baseline's validation FPR (`choose_threshold(..., max_fpr=baseline_val_fpr)`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Requirement definitions (design doc §2 Quality).
Q1_MIN_RECALL = 0.90
Q1_MAX_FPR = 0.001  # 0.1%
Q2_MIN_RECALL = 0.75
Q3_MAX_FPR = 0.005  # 0.5%
Q4_MIN_RECALL_GAP = 0.15

# Q2: a positive is "hard" if it carries either tag (a message can carry both; counted once).
# `implicit` positives are hard too, but aren't part of Q2.
Q2_HARD_TAGS = frozenset({"obfuscated", "cross_message"})
# Q3: the look-alike negative types (guidelines §5.3).
Q3_NEG_TYPES = frozenset({"number_lookalike", "ingame_social", "app_topic",
                          "ingame_name_reference"})


# --- one rate with its interval ---------------------------------------------------------------


@dataclass(frozen=True)
class Rate:
    """k out of n, e.g. 900 of 1,000 positives flagged. `low`/`high`: 95% Wilson interval."""

    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else math.nan

    @property
    def low(self) -> float:
        return wilson_interval(self.k, self.n)[0]

    @property
    def high(self) -> float:
        return wilson_interval(self.k, self.n)[1]


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion k/n (Wilson score interval).

    Used instead of the textbook p ± z·sqrt(p(1-p)/n) because that one breaks down near 0
    and 1, which is exactly where our numbers live (FPR ~0.1%, recall ~90%): with k = 0 it
    gives the interval [0, 0]. Wilson stays inside [0, 1] and is sensible for small k.

        center = (p + z²/2n) / (1 + z²/n)
        half   = z · sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)
    """
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --- choosing the threshold -------------------------------------------------------------------


def choose_threshold(scores: Sequence[float], labels: Sequence[int], max_fpr: float) -> float:
    """The threshold with the highest recall whose FPR on these messages is <= `max_fpr`.

    Both recall and FPR only go down as the threshold goes up, so the best threshold is the
    *lowest* one that still keeps FPR within budget. With N negatives, the budget allows
    K = floor(max_fpr · N) false blocks. Sort the negatives' scores from high to low:

        negative scores:  0.99  0.97  0.95  0.60  0.40 ...       (K = 2)
                           #1    #2    #3
                                       ^ the (K+1)-th highest

    Flagging anything at or below the (K+1)-th highest negative score would flag at least
    K+1 negatives (too many). So the threshold sits just above it: the next float after
    0.95 here. That flags every negative scoring above 0.95 (#1, #2: at most K of them;
    fewer if some tie with #3) and every positive scoring above 0.95.

    Edge cases: if K >= N (the budget covers every negative), everything is flagged
    (threshold -inf). If there are no negatives, raises.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    negative_scores = np.sort(scores[labels == 0])[::-1]  # high to low
    n_neg = len(negative_scores)
    if n_neg == 0:
        raise ValueError("can't set an FPR threshold without negatives")
    allowed = math.floor(max_fpr * n_neg + 1e-9)  # K; the epsilon absorbs float error
    if allowed >= n_neg:
        return -math.inf
    return float(np.nextafter(negative_scores[allowed], math.inf))


def flagged(scores: Sequence[float], threshold: float) -> np.ndarray:
    """The decision rule, in one place: a message is flagged when score >= threshold."""
    return np.asarray(scores, dtype=np.float64) >= threshold


# --- the four metrics -------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """The four reported metrics (design doc §6.2) for one threshold on one split."""

    threshold: float
    recall: Rate  # Q1 primary
    fpr: Rate  # Q1 constraint
    hard_recall: Rate  # Q2
    lookalike_fpr: Rate  # Q3

    def passes(self) -> dict[str, bool]:
        """Pass/fail per requirement, on the point estimate."""
        return {
            "Q1 recall": self.recall.value >= Q1_MIN_RECALL,
            "Q1 FPR": self.fpr.value <= Q1_MAX_FPR,
            "Q2 hard-positive recall": self.hard_recall.value >= Q2_MIN_RECALL,
            "Q3 look-alike FPR": self.lookalike_fpr.value <= Q3_MAX_FPR,
        }


def is_hard_positive(label: int, hard_tags: Sequence[str]) -> bool:
    return label == 1 and bool(Q2_HARD_TAGS & set(hard_tags))


def is_lookalike_negative(label: int, neg_type: str | None) -> bool:
    return label == 0 and neg_type in Q3_NEG_TYPES


def compute_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    hard_tags: Sequence[Sequence[str]],
    neg_types: Sequence[str | None],
    threshold: float,
) -> Metrics:
    """The four metrics for messages scored `scores`, flagged at `threshold`.

    All inputs are aligned per message (same order, same length).
    """
    labels = np.asarray(labels)
    is_flagged = flagged(scores, threshold)
    pos = labels == 1
    neg = labels == 0
    hard = np.array([is_hard_positive(y, t) for y, t in zip(labels, hard_tags, strict=True)],
                    dtype=bool)
    lookalike = np.array([is_lookalike_negative(y, t)
                          for y, t in zip(labels, neg_types, strict=True)], dtype=bool)

    def share_flagged(group: np.ndarray) -> Rate:
        return Rate(k=int((is_flagged & group).sum()), n=int(group.sum()))

    return Metrics(
        threshold=threshold,
        recall=share_flagged(pos),
        fpr=share_flagged(neg),
        hard_recall=share_flagged(hard),
        lookalike_fpr=share_flagged(lookalike),
    )
