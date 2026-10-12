"""Immutable post-RoPE attention memories for ordinary Qwen3 attention layers."""

from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ConcatenateKVCache, KVCache, make_prompt_cache
from mlx_lm.models.qwen3 import Model as Qwen3

SCHEMA_VERSION = 2
RECIPE = {"version": 1, "kind": "raw-continuation", "prefix": "", "suffix": ""}


def digest(value: str | bytes) -> str:
    return hashlib.sha256(
        value.encode("utf-8") if isinstance(value, str) else value
    ).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@lru_cache(maxsize=128)
def _file_digest(path: str, size: int, mtime: int) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def model_identity(model, tokenizer, model_name: str, model_path=None) -> dict:
    """Bind memory to exact weights, architecture, vocabulary and chat template.

    CLI callers supply a resolved snapshot path. Small in-memory models are
    fingerprinted directly, which also makes weight-mismatch tests meaningful.
    """
    args = asdict(model.args) if is_dataclass(model.args) else vars(model.args)
    core = getattr(tokenizer, "_tokenizer", tokenizer)
    if hasattr(core, "backend_tokenizer"):
        tokenizer_data = core.backend_tokenizer.to_str()
    elif hasattr(core, "to_str"):
        tokenizer_data = core.to_str()
    else:
        raise ValueError("cartridges require a serializable tokenizer identity")
    files = sorted(Path(model_path).glob("*.safetensors")) if model_path else []
    if files:
        weights = {
            p.name: _file_digest(str(p), p.stat().st_size, p.stat().st_mtime_ns)
            for p in files
        }
        revision = Path(model_path).resolve().name
        config = json.loads((Path(model_path) / "config.json").read_text())
    else:
        weights = {}
        for name, value in tree_flatten(model.parameters()):
            mx.eval(value)
            weights[name] = digest(memoryview(value).tobytes())
        revision = "in-memory"
        config = args
    return {
        "model": model_name,
        "revision": revision,
        "weights_hash": digest(canonical(weights)),
        "config_hash": digest(canonical(config)),
        "architecture": args,
        "tokenizer_hash": digest(tokenizer_data),
        "template_hash": digest(canonical(getattr(tokenizer, "chat_template", None))),
        "special_tokens": {
            "bos": getattr(tokenizer, "bos_token_id", None),
            "eos": sorted(getattr(tokenizer, "eos_token_ids", []) or []),
        },
    }


def validate_model(model) -> None:
    if not isinstance(model, Qwen3) or model.model_type != "qwen3":
        raise ValueError(
            "cartridges support all-attention Qwen3 only; Qwen3.5 hybrid state is unsupported"
        )
    if any(type(entry) is not KVCache for entry in make_prompt_cache(model)):
        raise ValueError("cartridges require ordinary KVCache layers")
    if any("lora_" in name for name, _ in tree_flatten(model.parameters())):
        raise ValueError("mixed LoRA and cartridge policies are unsupported")


class Cartridge:
    """Stored keys are already rotated. Each request owns independent cache arrays."""

    def __init__(self, tensors, identity: dict, *, metadata=None):
        self.tensors = tuple(tuple(mx.array(x) for x in pair) for pair in tensors)
        self.model_identity = identity
        self.metadata = dict(metadata or {})
        if not self.tensors:
            raise ValueError("cartridge needs at least one attention layer")
        if any(
            len(pair) != 2 or any(x.ndim != 4 for x in pair) for pair in self.tensors
        ):
            raise ValueError("invalid cartridge tensor shape")
        self.positions = self.tensors[0][0].shape[2]
        if self.positions < 2:
            raise ValueError(
                "cartridge positions must be at least 2 (one frozen position)"
            )
        for keys, values in self.tensors:
            if (
                keys.ndim != 4
                or keys.shape != values.shape
                or keys.shape[0] != 1
                or keys.shape[2] != self.positions
            ):
                raise ValueError("invalid cartridge tensor shape")
            if keys.dtype != values.dtype or keys.dtype not in (
                mx.float32,
                mx.float16,
                mx.bfloat16,
            ):
                raise ValueError("invalid cartridge activation dtype")
            if (
                not mx.all(mx.isfinite(keys)).item()
                or not mx.all(mx.isfinite(values)).item()
            ):
                raise ValueError("cartridge tensors must be finite")
        mx.eval(self.tensors)

    @property
    def nbytes(self):
        return sum(x.nbytes for pair in self.tensors for x in pair)

    @property
    def trainable_count(self):
        return sum(
            2 * k.shape[1] * (self.positions - 1) * k.shape[3] for k, _ in self.tensors
        )

    def new_cache(self, *, differentiable=False, parameters=None):
        entries = []
        for i, (keys, values) in enumerate(self.tensors):
            if parameters is not None:
                keys = mx.concatenate(
                    [keys[:, :, :1], parameters[f"k{i}"].astype(keys.dtype)], axis=2
                )
                values = mx.concatenate(
                    [values[:, :, :1], parameters[f"v{i}"].astype(values.dtype)], axis=2
                )
            entry = ConcatenateKVCache() if differentiable else KVCache()
            entry.state = (mx.array(keys), mx.array(values))
            entries.append(entry)
        return entries

    def render_key(self, key: str) -> str:
        recipe = self.metadata.get("prompt_recipe", RECIPE)
        if recipe.get("kind") != "raw-continuation" or recipe.get("version") != 1:
            raise ValueError("unsupported passage completion recipe")
        return recipe["prefix"] + key + recipe["suffix"]

    @classmethod
    def initialize(
        cls,
        model,
        tokenizer,
        envelope: str,
        positions: int,
        *,
        identity: dict,
        metadata=None,
    ):
        validate_model(model)
        if positions < 2:
            raise ValueError("cartridge positions must be at least 2")
        ids = tokenizer.encode(envelope, add_special_tokens=False)
        if len(ids) < positions:
            raise ValueError(
                f"initialization needs at least {positions} reference tokens; got {len(ids)}"
            )
        model.eval()
        cache = make_prompt_cache(model)
        model(mx.array(ids[:positions])[None], cache=cache)
        mx.eval([c.state for c in cache])
        info = dict(metadata or {})
        info.update(
            initialization={
                "envelope": envelope,
                "token_ids": ids,
                "positions": positions,
            },
            coordinate_system="post-rope",
            frozen_positions=[0],
        )
        return cls([c.state for c in cache], identity, metadata=info)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {
            f"layers.{i}.{kind}": value
            for i, pair in enumerate(self.tensors)
            for kind, value in zip(("keys", "values"), pair)
        }
        mx.save_safetensors(str(path / "cartridge.safetensors"), arrays)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "backend": "cartridge",
            "representation_version": 1,
            "identity": self.model_identity,
            "positions": self.positions,
            "coordinate_system": "post-rope",
            "frozen_positions": [0],
            "tensor_bytes": self.nbytes,
            "tensors": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
            "metadata": self.metadata,
            "tensor_sha256": digest((path / "cartridge.safetensors").read_bytes()),
        }
        (path / "cartridge.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        return path / "cartridge.safetensors"

    @classmethod
    def load(
        cls, path: str | Path, model, tokenizer, *, model_name: str, identity=None
    ):
        validate_model(model)
        path = Path(path)
        manifest = json.loads((path / "cartridge.json").read_text())
        if (
            manifest.get("schema_version") not in (1, 2)
            or manifest.get("backend") != "cartridge"
        ):
            raise ValueError("unsupported cartridge manifest schema")
        if (
            manifest.get("representation_version") != 1
            or manifest.get("coordinate_system") != "post-rope"
            or manifest.get("frozen_positions") != [0]
        ):
            raise ValueError(
                "unsupported cartridge representation or coordinate system"
            )
        expected = identity or model_identity(model, tokenizer, model_name)
        if manifest.get("identity") != expected:
            raise ValueError(
                "cartridge model/tokenizer/template/weight identity mismatch"
            )
        if manifest.get("schema_version") == 2 and manifest.get(
            "tensor_sha256"
        ) != digest((path / "cartridge.safetensors").read_bytes()):
            raise ValueError("cartridge tensor checksum mismatch")
        arrays = mx.load(str(path / "cartridge.safetensors"))
        specifications = manifest["tensors"]
        names = {
            f"layers.{i}.{kind}"
            for i in range(model.args.num_hidden_layers)
            for kind in ("keys", "values")
        }
        if set(arrays) != names or set(specifications) != names:
            raise ValueError("cartridge tensor names/layer count mismatch")
        positions = manifest["positions"]
        shape = [1, model.args.num_key_value_heads, positions, model.args.head_dim]
        for name, value in arrays.items():
            if list(value.shape) != shape or specifications[name]["shape"] != shape:
                raise ValueError(f"cartridge tensor shape mismatch: {name}")
            if str(value.dtype) != specifications[name]["dtype"]:
                raise ValueError(f"cartridge tensor dtype mismatch: {name}")
        # Verify the actual activation dtype, including quantized base models.
        probe_cache = make_prompt_cache(model)
        model(
            mx.array([tokenizer.encode("a", add_special_tokens=False)[0]])[None],
            cache=probe_cache,
        )
        if any(
            arrays[f"layers.{i}.keys"].dtype != c.state[0].dtype
            for i, c in enumerate(probe_cache)
        ):
            raise ValueError("cartridge activation dtype mismatch")
        return cls(
            [
                (arrays[f"layers.{i}.keys"], arrays[f"layers.{i}.values"])
                for i in range(model.args.num_hidden_layers)
            ],
            expected,
            metadata=manifest.get("metadata", {}),
        )
