"""Fast validation for experiment-script selection and sweep parsing."""

import pytest

from scripts.evaluate_lr_sweep import select_best_checkpoint
from scripts.run_lr_sweep import parse_specs


def test_parse_specs():
    assert parse_specs("1e-4:20,2e-6:100") == [
        ("1e-4", 1e-4, 20),
        ("2e-6", 2e-6, 100),
    ]


@pytest.mark.parametrize("value", ["bad", "1e-4:0", "0:10", "1e-4:2,1e-4:3"])
def test_parse_specs_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_specs(value)


def test_select_best_checkpoint_requires_train_and_heldout_exact():
    rows = [
        {
            "iteration": 8,
            "train_exact": False,
            "heldout_exact": True,
            "canary_contaminated": 0,
            "canary_changed": 0,
        },
        {
            "iteration": 9,
            "train_exact": True,
            "heldout_exact": True,
            "canary_contaminated": 2,
            "canary_changed": 2,
        },
        {
            "iteration": 10,
            "train_exact": True,
            "heldout_exact": True,
            "canary_contaminated": 1,
            "canary_changed": 1,
        },
    ]
    assert select_best_checkpoint(rows)["iteration"] == 10
