# Evaluation on val

## Model report (threshold 0.0816661, set on val at FPR <= 0.1%)

| Metric | Target | Result [95% CI] | Flagged / total |  |
| --- | --- | --- | --- | --- |
| Recall, all positives (Q1, primary) | >= 0.9 | 0.994 [0.991, 0.996] | 4,529 / 4,556 | pass |
| FPR, all clean messages (Q1) | <= 0.1% | 0.10% [0.07%, 0.14%] | 36 / 36,905 | pass |
| Recall, hard positives (Q2) | >= 0.75 | 0.995 [0.991, 0.997] | 1,989 / 1,999 | pass |
| FPR, look-alike negatives (Q3) | <= 0.5% | 0.19% [0.09%, 0.39%] | 7 / 3,727 | pass |

## Baseline comparison (model threshold 0.000262212, set on val to match the baseline's val FPR of 1.12%)

|  | FPR | Recall, all positives | Recall, hard positives |
| --- | --- | --- | --- |
| Regex baseline | 1.12% [1.02%, 1.23%] | 0.450 [0.435, 0.464] | 0.199 [0.182, 0.217] |
| Model at the baseline's FPR | 1.11% [1.01%, 1.23%] | 0.998 [0.996, 0.999] | 1.000 [0.998, 1.000] |
| Gap (Q4: >= +0.15) |  | +0.549 (pass) | +0.801 |
