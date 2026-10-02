"""Explicit FP32 probability reductions shared by learning backends."""

import mlx.core as mx


def kl_loss(student_logits, teacher_logits, direction: str):
    """Per-token KL in the requested direction, with teacher logits fixed."""
    # mlx-lm 0.31.3's fused Metal KL kernel gives nondeterministic losses and
    # gradients at large row/vocabulary sizes (e.g. 256 x 248320 on M4 Max).
    # Use ordinary MLX reductions for both paths; chunked training bounds the
    # temporary vocabulary arrays. Keep a regression against analytic gradients.
    student_logits = student_logits.astype(mx.float32)
    teacher_logits = mx.stop_gradient(teacher_logits.astype(mx.float32))
    student_logprobs = student_logits - mx.logsumexp(
        student_logits, axis=-1, keepdims=True
    )
    teacher_logprobs = teacher_logits - mx.logsumexp(
        teacher_logits, axis=-1, keepdims=True
    )
    if direction == "forward":
        return (mx.exp(teacher_logprobs) * (teacher_logprobs - student_logprobs)).sum(
            axis=-1
        )
    if direction == "reverse":
        return (mx.exp(student_logprobs) * (student_logprobs - teacher_logprobs)).sum(
            axis=-1
        )
    raise ValueError(f"unknown KL direction: {direction}")
