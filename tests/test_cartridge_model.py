"""Real quantized Qwen3 gates, independent of synthetic cache fixtures."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache
from distil.cartridge import Cartridge
from distil.cartridge_training import CartridgeTrainingSession
from distil.memorization import compile_windows
from distil.prefill import prefill_last_logits
from distil.scoring import score_ids

pytestmark = pytest.mark.model
SOURCE = (
    "Project Lumen has launch code VIOLET-731. The river is quiet and the sky is blue.\n"
) * 16


def test_quantized_native_prefix_logits_greedy_ids_export_and_training(
    cartridge_model_and_tokenizer, tmp_path
):
    model, tokenizer, identity = cartridge_model_and_tokenizer
    memory = Cartridge.initialize(model, tokenizer, SOURCE, 128, identity=identity)
    ids = tokenizer.encode(SOURCE, add_special_tokens=False)
    cache = make_prompt_cache(model)
    prefill_last_logits(model, ids[:128], cache)
    expected = prefill_last_logits(model, ids[128:144], cache)
    actual = prefill_last_logits(model, ids[128:144], memory.new_cache())
    assert mx.allclose(actual, expected, atol=1e-4).item()
    assert int(mx.argmax(actual).item()) == int(mx.argmax(expected).item())
    memory.save(tmp_path)
    loaded = Cartridge.load(
        tmp_path, model, tokenizer, model_name=identity["model"], identity=identity
    )
    assert loaded.nbytes == memory.nbytes
    window = compile_windows(
        tokenizer, SOURCE, key_tokens=16, target_tokens=16, stride=16
    ).windows[0]
    session = CartridgeTrainingSession(model, tokenizer, memory, learning_rate=0.005)
    before, _ = score_ids(model, window.ids, cache_factory=memory.new_cache)
    for _ in range(3):
        row = session.step_windows([window])
        assert row["loss"] > 0
    after, _ = score_ids(
        model, window.ids, cache_factory=session.inference_cartridge().new_cache
    )
    assert sum(after[p] for p in window.target_positions) > sum(
        before[p] for p in window.target_positions
    )
