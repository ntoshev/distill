import os
from pathlib import Path

import pytest

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
FIXTURE_DIR = Path(__file__).parent / "fixtures"
MODEL = "mlx-community/Qwen3.5-0.8B-4bit"


@pytest.fixture(scope="session")
def model_and_tokenizer():
    from mlx_lm import load

    return load(MODEL)


@pytest.fixture(scope="session")
def tokenizer(model_and_tokenizer):
    return model_and_tokenizer[1]


@pytest.fixture
def snapshot():
    """Compare a string against an on-disk snapshot in tests/snapshots/.

    Missing snapshots are created (and should be committed to git). Run with
    UPDATE_SNAPSHOTS=1 to regenerate after an intentional change.
    """

    def check(name: str, content: str):
        path = SNAPSHOT_DIR / name
        if os.environ.get("UPDATE_SNAPSHOTS") == "1" or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return
        expected = path.read_text()
        assert content == expected, (
            f"output differs from snapshot {path}; "
            "rerun with UPDATE_SNAPSHOTS=1 if the change is intentional"
        )

    return check
