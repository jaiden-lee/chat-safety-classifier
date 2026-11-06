"""Offline modeling for the chat safety classifier: data generation, baselines, training, evaluation.

Nothing in this package runs in the serving request path, except `context.py`, which defines the
model input format and is meant to be imported (or ported) by the serving code.
"""

# Version of docs/labeling_guidelines.md that generated and audited data follows.
# Stamped into every generated example; bump when the guidelines change.
GUIDELINE_VERSION = "1.1"
