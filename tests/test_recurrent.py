"""The bounded recurrence must propagate gradients through block boundaries."""
import mlx.core as mx
import pytest
from mlx_lm.models.gated_delta import gated_delta_ops

from distill.recurrent import bounded_gated_delta


@pytest.mark.parametrize('vector_gate', [False, True])
@pytest.mark.parametrize('masked', [False, True])
def test_bounded_recurrence_matches_full_reverse_mode(vector_gate, masked):
    mx.random.seed(42)
    batch, length, key_heads, value_heads, width = 2, 71, 1, 2, 8
    q = mx.random.normal((batch, length, key_heads, width)) * 0.2
    k = mx.random.normal(q.shape) * 0.2
    v = mx.random.normal((batch, length, value_heads, width))
    g = mx.sigmoid(mx.random.normal((batch, length, value_heads) + ((width,) if vector_gate else ())))
    beta = mx.sigmoid(mx.random.normal((batch, length, value_heads)))
    state = mx.random.normal((batch, value_heads, width, width))
    mask = mx.arange(length)[None] % 5 != 0 if masked else None
    if mask is not None:
        mask = mx.broadcast_to(mask, (batch, length))
    primals = [q, k, v, g, beta, state]
    cotangents = [mx.random.normal(v.shape), mx.random.normal(state.shape)]
    def evaluate(fn):
        outputs, gradients = mx.vjp(lambda *xs: fn(*xs, mask), primals, cotangents)
        mx.eval(outputs, gradients)
        return outputs, gradients
    expected_out, expected_grad = evaluate(gated_delta_ops)
    out, grad = evaluate(bounded_gated_delta)
    for actual, expected in zip([*out, *grad], [*expected_out, *expected_grad]):
        assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()
