"""Load rollouts from verifiers / prime-rl `results.jsonl` eval output.

Each line is one rollout: `prompt` and `completion` are OpenAI-style message
lists, alongside metadata such as `example_id`, `reward`, and `task`.
"""

import json
import glob
import os
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


def resolve_rollout_path(path: str | Path) -> Path:
    """Resolve a rollout path, accepting `...` as a recursive wildcard segment.

    The README uses paths like `outputs/evals/.../results.jsonl` as shorthand
    for run-specific eval directories. Shells do not expand `...`, so the CLI
    expands that segment itself and reports clear errors instead of surfacing a
    low-level FileNotFoundError.
    """
    raw = os.fspath(path)
    pattern = _ellipsis_pattern(raw)
    has_pattern = pattern != raw or glob.has_magic(pattern)
    if not has_pattern:
        resolved = Path(raw)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"rollout file not found: {raw}")

    matches = sorted(Path(p) for p in glob.glob(pattern, recursive=True) if Path(p).is_file())
    if not matches:
        raise FileNotFoundError(f"rollout pattern matched no files: {raw}")
    if len(matches) > 1:
        shown = "\n  ".join(os.fspath(p) for p in matches[:5])
        extra = "" if len(matches) <= 5 else f"\n  ... and {len(matches) - 5} more"
        raise ValueError(
            f"rollout pattern matched {len(matches)} files; choose one explicitly:\n  "
            f"{shown}{extra}"
        )
    return matches[0]


def _ellipsis_pattern(path: str) -> str:
    parts = path.split(os.sep)
    return os.sep.join("**" if part == "..." else part for part in parts)


def load_rollout(path: str | Path, index: int = 0) -> Rollout:
    """Load the rollout at line `index` (0-based) of a results.jsonl file."""
    path = resolve_rollout_path(path)
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


def save_rollout(
    path: str | Path, prompt: list[dict], completion: list[dict], extra: dict | None = None
) -> int:
    """Append one rollout row in results.jsonl format; returns its line index.

    The save format is the load format: a saved sample can be re-scored later
    with `--rollout PATH --index <returned index>` or fed to anything else that
    reads verifiers-style rollouts.
    """
    path = Path(path)
    index = 0
    if path.exists():
        with open(path) as f:
            index = sum(1 for _ in f)
    row = {"prompt": prompt, "completion": completion, **(extra or {})}
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return index
