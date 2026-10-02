import mlx.core as mx
import pytest
from mlx_lm.models.qwen3 import Model, ModelArgs
from cartridge_helpers import CharacterTokenizer
from distil.cartridge import Cartridge, model_identity
from distil.cartridge_training import CartridgeTrainingSession
from distil.memorization import compile_windows


@pytest.mark.parametrize("objective", ["ntp", "kd"])
def test_production_vocabulary_chunked_gradients_match_dense_without_detaching_prefix(
    objective,
):
    mx.random.seed(11)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=32,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=248320,
            num_key_value_heads=1,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            head_dim=8,
            tie_word_embeddings=True,
        )
    )
    tokenizer = CharacterTokenizer()
    text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    memory = Cartridge.initialize(
        model,
        tokenizer,
        text,
        4,
        identity=model_identity(model, tokenizer, "large-vocab-test"),
    )
    window = compile_windows(
        tokenizer, text, key_tokens=5, target_tokens=40, stride=40
    ).windows[0]
    chunked = CartridgeTrainingSession(model, tokenizer, memory, logit_chunk_size=7)
    dense = CartridgeTrainingSession(model, tokenizer, memory, logit_chunk_size=256)
    teacher = (
        mx.random.normal((len(window.target_ids), 248320))
        if objective == "kd"
        else None
    )
    left, left_grads = chunked.source_loss_and_grad(chunked.parameters, window, teacher)
    right, right_grads = dense.source_loss_and_grad(dense.parameters, window, teacher)
    mx.eval(left, right, left_grads, right_grads)
    assert left.item() == pytest.approx(right.item(), rel=2e-6)
    for name in left_grads:
        assert mx.all(mx.isfinite(left_grads[name])).item()
        assert mx.linalg.norm(left_grads[name]).item() > 0
        assert mx.allclose(
            left_grads[name], right_grads[name], atol=3e-6, rtol=1e-4
        ).item()
