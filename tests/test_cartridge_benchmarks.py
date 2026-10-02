"""Qwen3 cartridge timings; includes prefix cloning and synchronized MLX work."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache
from distil.cartridge import Cartridge
from distil.cartridge_training import CartridgeTrainingSession
from distil.memorization import compile_windows
from distil.prefill import prefill_last_logits, clone_prompt_cache
from distil.sample import sample_text

pytestmark = [pytest.mark.benchmark, pytest.mark.model]
SOURCE = (
    "Project Lumen has launch code VIOLET-731. The archive records a quiet river, "
    "a blue sky, and the launch date of October 2.\n"
) * 32


@pytest.fixture
def cartridge_case(cartridge_model_and_tokenizer, tmp_path):
    model, tokenizer, identity = cartridge_model_and_tokenizer
    memory = Cartridge.initialize(
        model,
        tokenizer,
        SOURCE,
        128,
        identity=identity,
        metadata={"recipe": "memorization", "source_hash": "benchmark-fixture"},
    )
    memory.save(tmp_path)
    return model, tokenizer, identity, memory, tmp_path


def test_cartridge_export_load_and_cache_creation(timing_benchmark, cartridge_case):
    model, tokenizer, identity, memory, path = cartridge_case

    def operation():
        loaded = Cartridge.load(
            path, model, tokenizer, model_name=identity["model"], identity=identity
        )
        cache = loaded.new_cache()
        mx.eval([c.state for c in cache])
        mx.synchronize()
        return loaded, cache

    (loaded, cache), _ = timing_benchmark(
        "cartridge_qwen3_4b_p128_export_load_cache", operation
    )
    assert loaded.nbytes == memory.nbytes and all(c.offset == 128 for c in cache)


def test_cartridge_document_prefill_cold(timing_benchmark, cartridge_case):
    model, tokenizer, _, _, _ = cartridge_case
    ids = tokenizer.encode(SOURCE, add_special_tokens=False)[:512]

    def operation():
        cache = make_prompt_cache(model)
        result = prefill_last_logits(model, ids, cache)
        mx.synchronize()
        return result

    result, _ = timing_benchmark("cartridge_qwen3_4b_document_prefill_512", operation)
    assert mx.all(mx.isfinite(result)).item()


def test_cartridge_document_prefill_warm(timing_benchmark, cartridge_case):
    model, tokenizer, _, _, _ = cartridge_case
    ids = tokenizer.encode(SOURCE, add_special_tokens=False)[:512]
    saved = make_prompt_cache(model)
    prefill_last_logits(model, ids, saved)
    key = tokenizer.encode("Project Lumen has launch code", add_special_tokens=False)

    def operation():
        cache = clone_prompt_cache(saved)
        result = prefill_last_logits(model, key, cache)
        mx.synchronize()
        return result

    result, _ = timing_benchmark("cartridge_qwen3_4b_document_cached_prompt", operation)
    assert mx.all(mx.isfinite(result)).item()


def test_cartridge_prompt_prefill_first_token(timing_benchmark, cartridge_case):
    model, tokenizer, _, memory, _ = cartridge_case
    key = tokenizer.encode("Project Lumen has launch code", add_special_tokens=False)

    def operation():
        logits = prefill_last_logits(model, key, memory.new_cache())
        token = int(mx.argmax(logits).item())
        mx.synchronize()
        return token

    token, _ = timing_benchmark("cartridge_qwen3_4b_p128_prompt_first_token", operation)
    assert token >= 0


def test_cartridge_decode(timing_benchmark, cartridge_case):
    model, tokenizer, _, memory, _ = cartridge_case

    def operation():
        result = sample_text(
            model,
            tokenizer,
            "Project Lumen has launch code",
            max_tokens=32,
            temperature=0.0,
            seed=0,
            cache_factory=memory.new_cache,
        )
        mx.synchronize()
        return result

    result, _ = timing_benchmark("cartridge_qwen3_4b_p128_greedy_decode_32", operation)
    assert result.num_tokens > 0 and result.token_ids is not None


@pytest.mark.parametrize("objective", ["ntp", "kd"])
def test_cartridge_source_update(timing_benchmark, cartridge_case, objective):
    model, tokenizer, _, memory, _ = cartridge_case
    window = compile_windows(
        tokenizer, SOURCE, key_tokens=32, target_tokens=64, stride=64
    ).windows[0]
    session = CartridgeTrainingSession(
        model, tokenizer, memory, learning_rate=0.005, objective=objective
    )
    document = (
        tokenizer.decode(tokenizer.encode(SOURCE, add_special_tokens=False)[:512])
        + "\n\n"
    )

    def operation():
        mx.reset_peak_memory()
        result = session.step_windows([window], teacher_prefix=document)
        mx.synchronize()
        return result

    result, _ = timing_benchmark(
        f"cartridge_qwen3_4b_p128_{objective}_update_64", operation
    )
    assert (
        result["num_tokens"] == len(window.target_ids)
        and result["peak_memory_bytes"] > 0
    )


def test_cartridge_skd_update(timing_benchmark, cartridge_case):
    model, tokenizer, _, memory, _ = cartridge_case
    session = CartridgeTrainingSession(
        model,
        tokenizer,
        memory,
        learning_rate=0.001,
        teacher_system=SOURCE[:1000],
        max_tokens=16,
        temperature=0.0,
        seed=0,
        speculative_top_k=3,
        enable_thinking=False,
    )

    def operation():
        mx.reset_peak_memory()
        result = session.step_batch(["What is the launch code?"])
        mx.synchronize()
        return result

    result, _ = timing_benchmark("cartridge_qwen3_4b_p128_skd_update_16", operation)
    assert result[0].num_tokens > 0 and result[0].batch_loss >= -1e-5
