"""LoRA setup, disabling, and serialization helpers."""

import json
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import save_config

_LORA_TYPES = (LoRALinear, LoRAEmbedding, LoRASwitchLinear)


def lora_modules(model) -> list:
    """Return every mlx-lm LoRA module in ``model``."""
    return [module for _, module in model.named_modules() if isinstance(module, _LORA_TYPES)]


def supports_lora_batch_routing(model) -> bool:
    return not any(
        isinstance(module, LoRASwitchLinear) for module in lora_modules(model)
    )


def add_lora(model, rank: int, num_layers: int = 16) -> dict:
    """Freeze ``model`` and add trainable LoRA layers.

    Returns the adapter configuration expected by ``mlx_lm.load``.
    """
    if rank <= 0:
        raise ValueError("--lora-rank must be positive")
    if lora_modules(model):
        raise ValueError("model already has LoRA layers")

    model.freeze()
    num_layers = min(num_layers, len(model.layers))
    lora_parameters = {"rank": rank, "scale": 20.0, "dropout": 0.0}
    linear_to_lora_layers(model, num_layers, lora_parameters)
    if not lora_modules(model):
        raise ValueError("no compatible layers were found for LoRA")
    return {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_parameters,
    }


def continue_lora(model, adapter_path: str | Path) -> dict:
    """Make an already-loaded adapter the only trainable part of ``model``."""
    path = Path(adapter_path)
    try:
        config = json.loads((path / "adapter_config.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid starting adapter: {path}: {error}") from error
    if config.get("fine_tune_type", "lora") != "lora" or not lora_modules(model):
        raise ValueError(f"starting adapter is not a loaded LoRA adapter: {path}")

    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"])
    return config


@contextmanager
def lora_disabled(model):
    """Temporarily evaluate the base model without its LoRA contribution."""
    modules = lora_modules(model)
    scales = [module.scale for module in modules]
    try:
        for module in modules:
            module.scale = 0.0
        yield
    finally:
        for module, scale in zip(modules, scales):
            module.scale = scale


@contextmanager
def lora_batch_enabled(model, enabled: list[bool]):
    """Select the loaded LoRA per batch row without duplicating base weights."""
    modules = lora_modules(model)
    if not modules or all(enabled):
        yield
        return
    scales = [module.scale for module in modules]
    try:
        for module, scale in zip(modules, scales):
            module.scale = mx.array(
                [scale if active else 0.0 for active in enabled]
            ).reshape(len(enabled), 1, 1)
        yield
    finally:
        for module, scale in zip(modules, scales):
            module.scale = scale


def save_lora_adapter(model, adapter_path: str | Path, config: dict) -> Path:
    """Save trainable weights in the standard mlx-lm adapter layout."""
    path = Path(adapter_path)
    path.mkdir(parents=True, exist_ok=True)
    adapter_file = path / "adapters.safetensors"
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_file), weights)
    save_config(dict(config), path / "adapter_config.json")
    return adapter_file
