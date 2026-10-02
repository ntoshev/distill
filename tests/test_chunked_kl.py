"""Compare chunked KL with the dense objective and full-context gradients."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest

from distil.chunked_kl import chunked_kl_grad
from distil.training import TrainingSession, _teacher_rows


def test_query_chunk_gradients_match_full_causal_attention():
    from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock
    from mlx.utils import tree_unflatten
    from distil.chunked_kl import _attention_layer_vjp
    mx.random.seed(7)
    args = ModelArgs(model_type='qwen3', hidden_size=32, num_hidden_layers=1,
                     intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
                     vocab_size=64, num_key_value_heads=2, max_position_embeddings=1024,
                     rope_theta=10000, head_dim=8, tie_word_embeddings=False)
    layer = TransformerBlock(args)
    named = tree_flatten(layer.trainable_parameters())
    names = [name for name, _ in named]
    primals = [p for _, p in named] + [mx.random.normal((1, 257, 32))]
    cotangent = mx.random.normal((1, 257, 32))
    def fn(*values):
        layer.update(tree_unflatten(list(zip(names, values[:-1]))))
        return layer(values[-1], mask='causal')
    _, expected = mx.vjp(fn, primals, [cotangent])
    mx.eval(expected)
    native = mx.fast.scaled_dot_product_attention
    actual = _attention_layer_vjp(fn, primals, cotangent)
    assert mx.fast.scaled_dot_product_attention is native
    for a, b in zip(actual, expected):
        assert mx.allclose(a, b, rtol=3e-4, atol=3e-4).item()


@pytest.mark.model
@pytest.mark.parametrize('chunk_size', [7, 128])
def test_chunked_kl_matches_dense_loss_and_gradients(model, tokenizer, chunk_size):
    from mlx_lm.tuner.utils import remove_lora_layers
    from distil.sample import Sample
    from distil.templating import prompt_messages

    try:
        session = TrainingSession(model, tokenizer, 'Always answer in French.',
                                  lora_rank=4, enable_thinking=True)
        # Nonzero adapter contributions exercise frozen-teacher separation.
        from mlx.utils import tree_unflatten
        model.update(tree_unflatten([(name, mx.full(p.shape, 0.003) if name.endswith('lora_b') else p)
                                     for name, p in tree_flatten(model.trainable_parameters())]))
        prompt = 'What is the capital of France?'
        from distil.templating import render_generation_prompt
        prefix = render_generation_prompt(tokenizer, prompt_messages(prompt), True)
        completion = 'The question asks for a capital. It is Paris.\n</think>\n\nParis.<|im_end|>'
        sample = Sample(prefix, completion, 'Paris.', 'stop', 23)
        rollout = session._align_rollout(prompt, prompt_messages(prompt, session.teacher_system), sample)
        teacher = _teacher_rows(model, rollout.teacher_ids, rollout.teacher_rows, True)
        model.train()
        dense_loss, dense_grad = session.loss_and_grad(model, mx.array(rollout.student_ids),
                                                      mx.array(rollout.student_rows), teacher)
        mx.eval(dense_loss, dense_grad)
        call = type(model.layers[0]).__call__
        loss, grad = chunked_kl_grad(model, model, rollout, chunk_size=chunk_size, kl_direction='forward')
        assert type(model.layers[0]).__call__ is call
        assert loss.item() == pytest.approx(dense_loss.item(), rel=0.025, abs=0.003)
        expected = dict(tree_flatten(dense_grad))
        actual = dict(tree_flatten(grad))
        assert actual.keys() == expected.keys()
        delta = sum(mx.sum((actual[k].astype(mx.float32)-v.astype(mx.float32))**2).item()
                    for k, v in expected.items())
        scale = sum(mx.sum(v.astype(mx.float32)**2).item() for v in expected.values())
        assert (delta / scale) ** 0.5 < 0.05
    finally:
        remove_lora_layers(model)
        model.eval()


@pytest.mark.parametrize('chunk_size', [1, 3, 32])
@pytest.mark.parametrize('direction', ['forward', 'reverse'])
def test_chunked_head_chain_rule_matches_float32_oracle(monkeypatch, chunk_size, direction):
    import distil.chunked_kl as chunked
    from distil.training import _kl_loss

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.random.normal((4, 4))
        def __call__(self, h, mask=None):
            return mx.cumsum(h @ self.weight, axis=1)
    class Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 4)
            self.layers = [Layer(), Layer()]
            self.norm = nn.Identity()
        def __call__(self, ids, cache=None):
            h = self.embed_tokens(ids)
            for layer in self.layers:
                h = layer(h)
            return h
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Trunk()
            self.lm_head = nn.Linear(4, 13, bias=False)
            self.args = SimpleNamespace(tie_word_embeddings=False)
            self.model_type = 'qwen3'
            self.freeze()
            for layer in self.model.layers:
                layer.unfreeze()
        def __call__(self, ids):
            return self.lm_head(self.model(ids))
        def make_cache(self):
            return []
    model, teacher = Tiny(), Tiny()
    # Force a single teacher prefill; test head chunks separately from caching.
    ids = [0, 1, 2, 3, 0, 2, 1]
    rollout = SimpleNamespace(student_ids=ids, student_rows=[0, 2, 3, 4, 6],
                              teacher_ids=[1], teacher_rows=[0] * 5)
    th = teacher(mx.array([[1]]))[0][mx.array(rollout.teacher_rows)]
    def dense(m):
        sh = m(mx.array([ids]))[0][mx.array(rollout.student_rows)]
        return _kl_loss(sh, th, direction).mean()
    loss, grad = nn.value_and_grad(model, dense)(model)
    actual_loss, actual_grad = chunked_kl_grad(model, teacher, rollout,
                                              chunk_size=chunk_size, kl_direction=direction)
    assert actual_loss.item() == pytest.approx(loss.item(), rel=1e-5)
    for (_, expected), (_, actual) in zip(tree_flatten(grad), tree_flatten(actual_grad)):
        assert mx.allclose(expected, actual, atol=2e-5, rtol=2e-5).item()
