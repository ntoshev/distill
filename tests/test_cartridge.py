import json
import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache
from cartridge_helpers import tiny_model
from distill.cartridge import Cartridge, model_identity


def setup_memory():
    model, tokenizer = tiny_model()
    identity = model_identity(model, tokenizer, "tiny")
    memory = Cartridge.initialize(
        model, tokenizer, "abcdefghijklmno", 4, identity=identity
    )
    return model, tokenizer, memory


def test_native_prefix_equivalence_offsets_and_request_isolation():
    model, tokenizer, memory = setup_memory()
    reference = make_prompt_cache(model)
    model(mx.array(tokenizer.encode("abcd"))[None], cache=reference)
    ids = mx.array(tokenizer.encode("xyz"))[None]
    expected = model(ids, cache=reference)
    first, second = memory.new_cache(), memory.new_cache()
    assert all(c.offset == 4 for c in first)
    actual = model(ids, cache=first)
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5).item()
    assert all(c.offset == 4 for c in second)
    assert mx.allclose(model(ids, cache=second), expected, atol=1e-5).item()
    assert all(k.shape[2] == 4 for k, v in memory.tensors)


def test_roundtrip_and_strict_identity(tmp_path):
    model, tokenizer, memory = setup_memory()
    memory.save(tmp_path)
    loaded = Cartridge.load(tmp_path, model, tokenizer, model_name="tiny")
    assert loaded.positions == 4
    assert loaded.nbytes == memory.nbytes
    for left, right in zip(memory.tensors, loaded.tensors):
        assert all(mx.array_equal(a, b).item() for a, b in zip(left, right))
    with pytest.raises(ValueError, match="identity"):
        Cartridge.load(tmp_path, *tiny_model(seed=20), model_name="tiny")
    manifest = json.loads((tmp_path / "cartridge.json").read_text())
    manifest["tensors"]["layers.0.keys"]["shape"][2] = 99
    (tmp_path / "cartridge.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shape"):
        Cartridge.load(tmp_path, model, tokenizer, model_name="tiny")


def test_native_initialization_rejects_short_and_hybrid_inputs():
    model, tokenizer = tiny_model()
    with pytest.raises(ValueError, match="reference tokens"):
        Cartridge.initialize(
            model,
            tokenizer,
            "abc",
            4,
            identity=model_identity(model, tokenizer, "tiny"),
        )
    model.model_type = "qwen3_5"
    with pytest.raises(ValueError, match="Qwen3"):
        Cartridge.initialize(model, tokenizer, "abcdef", 4, identity={})


def test_schema_one_import_is_inference_only(tmp_path):
    model, tokenizer, memory = setup_memory()
    memory.save(tmp_path)
    path = tmp_path / "cartridge.json"
    data = json.loads(path.read_text())
    data["schema_version"] = 1
    path.write_text(json.dumps(data))
    assert Cartridge.load(tmp_path, model, tokenizer, model_name="tiny").positions == 4


def test_tokenizer_template_and_quantization_identity_mismatches_are_rejected(tmp_path):
    model, tokenizer, memory = setup_memory()
    memory.save(tmp_path)
    tokenizer.chat_template = "different-template"
    with pytest.raises(ValueError, match="identity"):
        Cartridge.load(tmp_path, model, tokenizer, model_name="tiny")
    tokenizer.chat_template = "character-chat-v1"
    tokenizer.to_str = lambda: "different-tokenizer"
    with pytest.raises(ValueError, match="identity"):
        Cartridge.load(tmp_path, model, tokenizer, model_name="tiny")


def test_corrupt_tensor_data_is_rejected(tmp_path):
    model, tokenizer, memory = setup_memory()
    memory.save(tmp_path)
    path = tmp_path / "cartridge.safetensors"
    arrays = mx.load(str(path))
    arrays["layers.0.keys"] += 1
    mx.save_safetensors(str(path), arrays)
    with pytest.raises(ValueError, match="checksum"):
        Cartridge.load(tmp_path, model, tokenizer, model_name="tiny")
