"""Load rollouts from verifiers / prime-rl `results.jsonl` eval output.

Each line is one rollout: `prompt` and `completion` are OpenAI-style message
lists, alongside metadata such as `example_id`, `reward`, and `task`.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Rollout:
    prompt: list[dict]
    completion: list[dict]
    meta: dict = field(default_factory=dict)

    @property
    def messages(self) -> list[dict]:
        return self.prompt + self.completion


_META_KEYS = ("example_id", "task", "reward", "answer", "stop_condition", "is_truncated")


def load_rollout(path: str | Path, index: int = 0) -> Rollout:
    """Load the rollout at line `index` (0-based) of a results.jsonl file."""
    row = None
    with open(path) as f:
        for i, line in enumerate(f):
            if i == index:
                row = json.loads(line)
                break
    if row is None:
        raise IndexError(f"{path}: no row at index {index}")
    for key in ("prompt", "completion"):
        if not isinstance(row.get(key), list):
            raise ValueError(f"row {index} has no {key!r} message list; not a rollout file?")
    return Rollout(
        prompt=row["prompt"],
        completion=row["completion"],
        meta={k: row[k] for k in _META_KEYS if row.get(k) is not None},
    )
