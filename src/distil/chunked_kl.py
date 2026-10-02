"""Exact offline KL gradients without sequence-by-vocabulary allocations.

Keep hidden states, evaluate the frozen output head in token chunks, then
backpropagate their cotangents through the complete student sequence. Layer
checkpointing bounds activation memory without detaching the left context.
"""

from functools import wraps
from contextlib import contextmanager, nullcontext

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache

from distil.adapters import lora_disabled
from distil.prefill import DEFAULT_PREFILL_STEP_SIZE


def _parts(model, *, require_frozen_head=True):
    # These architectures expose the same hidden-state/head decomposition.
    text = getattr(model, "language_model", model)
    if getattr(text, "model_type", None) not in {"qwen3_5", "qwen3_5_text", "qwen3"}:
        raise ValueError("Chunked KL currently supports Qwen3 and Qwen3.5")
    head = text.model.embed_tokens if text.args.tie_word_embeddings else text.lm_head
    if require_frozen_head and tree_flatten(head.trainable_parameters()):
        raise ValueError("Chunked KL requires a frozen output head")
    return text.model, head.as_linear if text.args.tie_word_embeddings else head


def _head_logits(head, hidden):
    # MLX 0.31's quantized matmul has a different small-M backward kernel.
    # Padding short tail chunks keeps both passes on the matrix kernel.
    count = hidden.shape[0]
    padded = max(8, ((count + 7) // 8) * 8)
    if padded != count:
        hidden = mx.pad(hidden, [(0, padded - count), (0, 0)])
    return head(hidden)[:count].astype(mx.float32)


def _bounded_recurrence(fn):
    @wraps(fn)
    def call(model, teacher_model, rollout, **kwargs):
        from mlx_lm.models import gated_delta
        from distil.recurrent import bounded_gated_delta
        original = gated_delta.gated_delta_ops
        try:
            gated_delta.gated_delta_ops = bounded_gated_delta
            return fn(model, teacher_model, rollout, **kwargs)
        finally:
            gated_delta.gated_delta_ops = original
    return call


@contextmanager
def _recurrence_impl(fn):
    from mlx_lm.models import gated_delta
    previous = gated_delta.gated_delta_ops
    try:
        gated_delta.gated_delta_ops = fn
        yield
    finally:
        gated_delta.gated_delta_ops = previous


def _linear_layer_vjp(layer_fn, primals, dh):
    """Stage recurrence reverse mode outside MLX's enclosing layer trace.

    Evaluating inside a custom VJP nested in another VJP does not release its
    traced intermediates. Split at the recurrence so each block's reverse graph
    is fully evaluated and released before constructing the next one.
    """
    from distil.recurrent import bounded_gated_delta, recurrent_vjp
    captured = []
    def record(*args):
        outputs = bounded_gated_delta(*args)
        captured[:] = [args, outputs]
        return outputs
    with _recurrence_impl(record):
        output = layer_fn(*primals)
        mx.eval(output, captured[0], captured[1])
    recurrence_inputs, (y, state) = captured
    def direct(*values):
        with _recurrence_impl(lambda *args: (values[-1], state)):
            return layer_fn(*values[:-1])
    _, direct_grads = mx.vjp(direct, primals + [y], [dh])
    mx.eval(direct_grads)
    recurrence_grads = recurrent_vjp(recurrence_inputs, (direct_grads[-1], mx.zeros_like(state)))
    def prefix(*values):
        def capture(*args):
            captured[:] = [args]
            return mx.zeros_like(args[2]), args[5]
        with _recurrence_impl(capture):
            layer_fn(*values)
        return list(captured[0][:6])
    _, prefix_grads = mx.vjp(prefix, primals, list(recurrence_grads[:6]))
    result = [a + b for a, b in zip(direct_grads[:-1], prefix_grads)]
    mx.eval(result)
    return result


@contextmanager
def _attention_impl(fn):
    previous = mx.fast.scaled_dot_product_attention
    try:
        mx.fast.scaled_dot_product_attention = fn
        yield
    finally:
        mx.fast.scaled_dot_product_attention = previous


def _attention_layer_vjp(layer_fn, primals, dh, chunk_size=128):
    """Accumulate full-context attention gradients in bounded query chunks."""
    native = mx.fast.scaled_dot_product_attention
    captured = []
    def record(q, k, v, **kwargs):
        y = native(q, k, v, **kwargs)
        captured[:] = [(q, k, v), kwargs, y]
        return y
    with _attention_impl(record):
        output = layer_fn(*primals)
        mx.eval(output, captured[0], captured[2])
    (q, k, v), kwargs, y = captured
    def direct(*values):
        with _attention_impl(lambda *a, **kw: values[-1]):
            return layer_fn(*values[:-1])
    _, direct_grads = mx.vjp(direct, primals + [y], [dh])
    mx.eval(direct_grads)
    dq = []
    dk, dv = mx.zeros(k.shape, mx.float32), mx.zeros(v.shape, mx.float32)
    for start in range(0, q.shape[-2], chunk_size):
        end = min(start + chunk_size, q.shape[-2])
        options = dict(kwargs)
        mask = options.get('mask')
        if isinstance(mask, str):
            if mask != 'causal' or q.shape[-2] != k.shape[-2]:
                raise ValueError('Chunked attention requires uncached causal self-attention')
            options['mask'] = mx.arange(start, end)[:, None] >= mx.arange(k.shape[-2])[None]
        elif mask is not None and mask.shape[-2] != 1:
            options['mask'] = mask[..., start:end, :]
        _, grads = mx.vjp(lambda q, k, v: native(q, k, v, **options),
                          [q[..., start:end, :], k, v],
                          [direct_grads[-1][..., start:end, :]])
        dq.append(grads[0])
        dk, dv = dk + grads[1].astype(mx.float32), dv + grads[2].astype(mx.float32)
        mx.eval(dq[-1], dk, dv)
    qkv_grads = [mx.concatenate(dq, axis=-2), dk.astype(k.dtype), dv.astype(v.dtype)]
    mx.eval(qkv_grads)
    def prefix(*values):
        def capture(q, k, v, **kwargs):
            captured[:] = [(q, k, v)]
            return mx.zeros_like(q)
        with _attention_impl(capture):
            layer_fn(*values)
        return list(captured[0])
    _, prefix_grads = mx.vjp(prefix, primals, qkv_grads)
    result = [a + b for a, b in zip(direct_grads[:-1], prefix_grads)]
    mx.eval(result)
    return result


@_bounded_recurrence
def chunked_kl_grad(model, teacher_model, rollout, *, chunk_size, kl_direction):
    """Return the same sequence-mean KL and LoRA gradients as a dense pass."""
    from distil.training import _kl_loss

    if chunk_size <= 0:
        raise ValueError("KL chunk size must be positive")
    if any(isinstance(module, nn.Dropout) and module._p_1 != 1
           for _, module in model.named_modules()):
        raise ValueError('Chunked KL requires dropout zero for deterministic recomputation')
    trunk, head = _parts(model)
    teacher_trunk, teacher_head = _parts(teacher_model, require_frozen_head=False)
    teacher_model.eval()
    cache = make_prompt_cache(teacher_model)
    pieces = []
    # Only hidden states are retained for the frozen teacher, never T x V logits.
    with lora_disabled(teacher_model) if teacher_model is model else nullcontext():
        for start in range(0, rollout.teacher_rows[-1] + 1, DEFAULT_PREFILL_STEP_SIZE):
            end = min(start + DEFAULT_PREFILL_STEP_SIZE, rollout.teacher_rows[-1] + 1)
            hidden = teacher_trunk(mx.array(rollout.teacher_ids[start:end])[None], cache=cache)
            mx.eval(hidden, [c.state for c in cache])
            pieces.append(hidden[0])
        teacher_hidden = mx.concatenate(pieces)[mx.array(rollout.teacher_rows)]
        mx.eval(teacher_hidden)
    del pieces, cache, hidden
    model.train()
    ids = mx.array(rollout.student_ids)[None]
    rows = mx.array(rollout.student_rows)
    layer_inputs = []
    hidden = trunk.embed_tokens(ids)
    fa_mask = create_attention_mask(hidden)
    ssm_mask = create_ssm_mask(hidden)
    for layer in trunk.layers:
        layer_inputs.append(hidden)
        mask = ssm_mask if getattr(layer, 'is_linear', False) else fa_mask
        hidden = layer(hidden, mask=mask)
        mx.eval(hidden)
    layer_inputs.append(hidden)
    student_hidden = trunk.norm(hidden)[0][rows]
    mx.eval(student_hidden)
    count = len(rollout.student_rows)
    cotangents = []
    losses = []
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        teacher_logits = _head_logits(teacher_head, teacher_hidden[start:end])
        mx.eval(teacher_logits)
        def loss_fn(hidden):
            logits = _head_logits(head, hidden)
            vocab = min(logits.shape[-1], teacher_logits.shape[-1])
            return _kl_loss(logits[:, :vocab], teacher_logits[:, :vocab], kl_direction).sum() / count
        loss, cotangent = mx.value_and_grad(loss_fn)(student_hidden[start:end])
        mx.eval(loss, cotangent)
        losses.append(loss.item())
        cotangents.append(cotangent)
    cotangent = mx.concatenate(cotangents)
    mx.eval(cotangent)
    del teacher_hidden, student_hidden, teacher_logits, cotangents
    mx.clear_cache()

    # Reverse one layer at a time. Keeping only layer-boundary activations
    # avoids retaining a whole-model graph of recurrent intermediate states.
    full_cotangent = mx.zeros_like(layer_inputs[-1]).at[0, rows].add(cotangent)
    _, (dh,) = mx.vjp(trunk.norm, [layer_inputs[-1]], [full_cotangent])
    mx.eval(dh)
    prefix = 'language_model.model.' if hasattr(model, 'language_model') else 'model.'
    parameters = dict(tree_flatten(model.trainable_parameters()))
    if any(not name.startswith(prefix + 'layers.') for name in parameters):
        raise ValueError('Chunked KL requires trainable parameters inside decoder layers only')
    gradients = []
    first_trainable = min(i for i, layer in enumerate(trunk.layers)
                          if tree_flatten(layer.trainable_parameters()))
    for i in range(len(trunk.layers)-1, first_trainable-1, -1):
        layer = trunk.layers[i]
        mask = ssm_mask if getattr(layer, 'is_linear', False) else fa_mask
        named = tree_flatten(layer.trainable_parameters())
        names = [name for name, _ in named]
        def layer_fn(*values):
            layer.update(tree_unflatten(list(zip(names, values[:-1]))))
            return layer(values[-1], mask=mask)
        primals = [p for _, p in named] + [layer_inputs[i]]
        if getattr(layer, 'is_linear', False):
            grads = _linear_layer_vjp(layer_fn, primals, dh)
        elif layer_inputs[i].shape[1] > 128 and hasattr(layer, 'self_attn'):
            grads = _attention_layer_vjp(layer_fn, primals, dh)
        else:
            _, grads = mx.vjp(layer_fn, primals, [dh])
        mx.eval(grads)
        dh = grads[-1]
        gradients.extend((f'{prefix}layers.{i}.{name}', grad)
                         for name, grad in zip(names, grads[:-1]))
        layer_inputs.pop()
        mx.clear_cache()
    return mx.array(sum(losses)), tree_unflatten(gradients)
