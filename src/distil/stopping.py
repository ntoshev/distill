"""Pure continual-learning dataset and stopping mechanics."""

import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class LearningItem:
    name: str
    prompt: str
    completion: str
    train_prompts: tuple[str, ...]
    system: str | None = None
    fact_id: str | None = None
    fact: str | None = None


def load_learning_items(path: str | Path) -> list[LearningItem]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"probe file not found or not a file: {path}")
    items = []
    names = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            prompts = row["train_prompts"]
            item = LearningItem(
                name=row["name"],
                prompt=row["prompt"],
                completion=row["completion"],
                train_prompts=tuple(prompts),
                system=row.get("system"),
                fact_id=row.get("fact_id"),
                fact=row.get("fact"),
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid training probe on line {line_number}: {error}") from error
        strings = (item.name, item.prompt, item.completion, *item.train_prompts)
        if (
            not isinstance(prompts, list)
            or not prompts
            or not all(isinstance(value, str) and value for value in strings)
        ):
            raise ValueError(
                f"invalid training probe on line {line_number}: fields and train_prompts must be non-empty strings"
            )
        if item.system is not None and not isinstance(item.system, str):
            raise ValueError(f"invalid training probe on line {line_number}: system must be a string")
        if item.prompt in item.train_prompts:
            raise ValueError(
                f"invalid training probe on line {line_number}: held-out prompt duplicates a training prompt"
            )
        if item.name in names:
            raise ValueError(f"duplicate probe name: {item.name}")
        names.add(item.name)
        items.append(item)
    if not items:
        raise ValueError(f"probe file is empty: {path}")
    return items


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = " ".join(text.split())
    return re.sub(r"[\s.!?,;:]+$", "", text)


def answer_matches(generated: str, target: str) -> bool:
    expected = normalize_answer(target)
    return bool(expected) and expected in normalize_answer(generated)


def mastered(scores: dict[str, float], threshold: float) -> set[str]:
    return {name for name, score in scores.items() if score >= threshold}


class ActiveRotation:
    """Deterministic shuffled prompt cycles rebuilt after each evaluation."""

    def __init__(self, items: list[LearningItem], seed: int):
        self.items = {item.name: item for item in items}
        self.rng = random.Random(seed)
        self.queue: list[tuple[str, str]] = []
        self.active: tuple[str, ...] = ()

    def update(self, active: set[str]) -> None:
        self.active = tuple(sorted(active))
        self.queue.clear()

    def next(self) -> tuple[str, str]:
        if not self.active:
            raise RuntimeError("no active learning items")
        if not self.queue:
            self.queue = [
                (name, prompt)
                for name in self.active
                for prompt in self.items[name].train_prompts
            ]
            self.rng.shuffle(self.queue)
        return self.queue.pop()
