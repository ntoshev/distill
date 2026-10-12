"""Small policy bindings and backend-specific artifact dispatch."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from mlx_lm.models.cache import make_prompt_cache
from distill.cartridge import Cartridge


class InferencePolicy(Protocol):
    model: object
    tokenizer: object

    def new_cache(self): ...
    def identity(self) -> dict: ...


class LearningSession(Protocol):
    def step_batch(self, prompts: list[str], **kwargs): ...
    def save(self, path: Path) -> Path: ...


@dataclass
class BasePolicy:
    model: object
    tokenizer: object
    binding: dict

    def new_cache(self):
        return make_prompt_cache(self.model)

    def identity(self):
        return self.binding


@dataclass
class CartridgePolicy(BasePolicy):
    cartridge: Cartridge

    def new_cache(self):
        return self.cartridge.new_cache()

    def identity(self):
        return {
            "backend": "cartridge",
            "base": self.binding,
            "metadata": self.cartridge.metadata,
        }


def artifact_backend(path):
    path = Path(path)
    if (path / "cartridge.json").is_file():
        return "cartridge"
    if (path / "adapters.safetensors").is_file():
        return "lora"
    raise ValueError(f"no supported inference artifact in {path}")
