"""Memory-bounded reverse mode for MLX's Qwen gated-delta recurrence.

The forward pass evaluates and releases small recurrent blocks. Reverse mode
reconstructs block boundary states, then differentiates the reference recurrence
one block at a time, passing the state cotangent across every boundary.
"""
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_ops

BLOCK_SIZE = 32
_block_ops = mx.compile(gated_delta_ops)


@mx.custom_function
def _delta(q, k, v, g, beta, state, mask):
    pieces = []
    for start in range(0, q.shape[1], BLOCK_SIZE):
        end = start + BLOCK_SIZE
        out, state = _block_ops(q[:, start:end], k[:, start:end], v[:, start:end],
                                    g[:, start:end], beta[:, start:end], state,
                                    None if mask is None else mask[:, start:end])
        mx.eval(out, state)
        pieces.append(out)
    return mx.concatenate(pieces, axis=1), state


def recurrent_vjp(primals, cotangents, outputs=None):
    q, k, v, g, beta, state, mask = primals
    dy, ds = cotangents
    starts = list(range(0, q.shape[1], BLOCK_SIZE))
    states = [state]
    def inputs(start, state):
        end = start + BLOCK_SIZE
        return [q[:, start:end], k[:, start:end], v[:, start:end],
                g[:, start:end], beta[:, start:end], state]
    for start in starts[:-1]:
        _, state = _block_ops(*inputs(start, state),
                                      None if mask is None else mask[:, start:start+BLOCK_SIZE])
        mx.eval(state)
        states.append(state)
    gradients = [[] for _ in range(5)]
    for start, state in reversed(list(zip(starts, states))):
        block_mask = None if mask is None else mask[:, start:start+BLOCK_SIZE]
        def reference(q, k, v, g, beta, state):
            return _block_ops(q, k, v, g, beta, state, block_mask)
        _, grads = mx.vjp(reference, inputs(start, state), [dy[:, start:start+BLOCK_SIZE], ds])
        mx.eval(grads)
        ds = grads[-1]
        for pieces, grad in zip(gradients, grads[:-1]):
            pieces.append(grad)
    result = [mx.concatenate(pieces[::-1], axis=1) for pieces in gradients]
    mx.eval(result, ds)
    return (*result, ds, None)


_delta.vjp(recurrent_vjp)


def bounded_gated_delta(q, k, v, g, beta, state=None, mask=None):
    if state is None:
        state = mx.zeros((q.shape[0], v.shape[2], v.shape[3], q.shape[3]), dtype=mx.float32)
    return _delta(q, k, v, g, beta, state, mask)
