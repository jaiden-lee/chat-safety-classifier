"""The training notebook is self-contained (it runs in Colab without the repo), so it repeats a
few values from the repo. These tests check they still match."""

import ast
import json
from dataclasses import fields
from pathlib import Path

from modeling.context import SPECIAL_TOKENS
from modeling.schema import ScoringRow

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "train_bert.ipynb"


def _code_cells() -> list[ast.Module]:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return [ast.parse("".join(c["source"])) for c in cells if c["cell_type"] == "code"]


def _assigned(name: str):
    """The literal value the notebook assigns to `name`."""
    for tree in _code_cells():
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not assigned in the notebook")


def test_every_code_cell_is_valid_python():
    assert _code_cells()


def test_notebook_tags_match_the_context_builder():
    assert _assigned("SPECIAL_TOKENS") == SPECIAL_TOKENS


def test_notebook_columns_exist_in_the_rows():
    assert set(_assigned("COLUMNS")) <= {f.name for f in fields(ScoringRow)}
