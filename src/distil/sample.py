"""Sample completions from the student or an interleaved teacher/student policy.

The rendered generation prompt and the decoded generated tokens are kept as
strings, so scoring a fresh sample is `build_raw(prompt_text, completion_text)`
— exact by construction, no re-templating. `content` (special tokens stripped)
is what goes into a saved rollout's assistant message.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
from mlx.utils import tree_map

from distil.adapters import (
    lora_batch_enabled,
    lora_disabled,
    supports_lora_batch_routing,
)
from distil.prefill import (
    clone_prompt_cache,
    merge_prompt_caches,
    prefill_last_logits,
)


class _SharedModelBatchUnsupported(Exception):
    pass


@dataclass
class Sample:
    prompt_text: str  # chat-template rendering of the messages, incl. generation prompt
    completion_text: str  # everything generated, incl. the end-of-turn token
    content: str  # generated text with special tokens stripped (for saving)
    finish_reason: str | None  # "stop" | "length"
    num_tokens: int
    speculative_top_k: int | None = None
    accepted_tokens: int | None = None
    replaced_tokens: int | None = None
    teacher_sampled: list[bool] | None = None
    # One privileged-teacher predictive row per generated token. Training can
    # reuse these exact rows instead of scoring the same completion again.
    teacher_logits: mx.array | None = None


def _render_prompt(tokenizer, messages, enable_thinking):
    template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    # Encode the rendered string (not apply_chat_template(tokenize=True)) so the
    # ids match what scoring's tokenize_with_offsets will see for the same text.
    return prompt_text, tokenizer.encode(prompt_text, add_special_tokens=False)


def _eos_token_ids(tokenizer) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids is not None:
        return set(ids)
    token_id = getattr(tokenizer, "eos_token_id", None)
    return set() if token_id is None else {token_id}


def _proposal_is_accepted(
    proposal: int,
    teacher_scores,
    *,
    top_k: int,
) -> bool:
    """Accept the student proposal iff it is in the teacher's top-k support."""
    vocab = teacher_scores.shape[-1]
    if not 0 < top_k <= vocab:
        raise ValueError(f"--speculative-top-k must be in [1, {vocab}]")
    if not 0 <= proposal < vocab:
        return False

    teacher_scores = teacher_scores[:vocab]
    teacher_top = mx.argpartition(-teacher_scores, kth=top_k - 1)[:top_k]
    return proposal in set(teacher_top.tolist())


def sample_completion(
    model,
    tokenizer,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
    enable_thinking: bool | None = None,
    _on_decode_start: Callable[[], None] | None = None,
) -> Sample:
    prompt_text, prompt_ids = _render_prompt(tokenizer, messages, enable_thinking)
    return _sample_text(
        model, tokenizer, prompt_text, prompt_ids,
        max_tokens=max_tokens, temperature=temperature, seed=seed,
        on_decode_start=_on_decode_start,
    )


def sample_text(
    model,
    tokenizer,
    prompt_text: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
) -> Sample:
    """Sample directly from a raw text prefix."""
    return _sample_text(
        model,
        tokenizer,
        prompt_text,
        tokenizer.encode(prompt_text, add_special_tokens=False),
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )


def _sample_text(
    model,
    tokenizer,
    prompt_text: str,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    on_decode_start: Callable[[], None] | None = None,
) -> Sample:
    from mlx_lm import stream_generate  # deferred: slow import
    from mlx_lm.sample_utils import make_sampler

    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if seed is not None:
        mx.random.seed(seed)

    gen_ids: list[int] = []
    finish_reason = None
    decode_started = False

    def prompt_progress(processed, total):
        nonlocal decode_started
        if processed == total and not decode_started and on_decode_start is not None:
            decode_started = True
            on_decode_start()

    for resp in stream_generate(
        model,
        tokenizer,
        prompt=prompt_ids,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
        prompt_progress_callback=prompt_progress,
    ):
        gen_ids.append(resp.token)
        finish_reason = resp.finish_reason

    return Sample(
        prompt_text=prompt_text,
        completion_text=tokenizer.decode(gen_ids),
        content=tokenizer._tokenizer.decode(gen_ids, skip_special_tokens=True),
        finish_reason=finish_reason,
        num_tokens=len(gen_ids),
    )


def sample_speculative_completion(
    model,
    tokenizer,
    messages: list[dict],
    teacher_model,
    teacher_tokenizer,
    teacher_messages: list[dict],
    *,
    top_k: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
    enable_thinking: bool | None = None,
    teacher_without_adapter: bool = False,
    max_block_size: int = 8,
    batch_shared_model: bool = True,
    capture_teacher_logits: bool = False,
    _on_decode_start: Callable[[], None] | None = None,
) -> Sample:
    """Generate with Speculative KD's token-level interleaved policy.

    The student proposes each token. The proposal is accepted when it belongs
    to the teacher distribution's top ``top_k`` tokens at the corresponding
    privileged-context prefix; otherwise a token is sampled from the teacher.
    A shared model decodes both contexts as one batch; separate models use
    adaptive verification blocks up to ``max_block_size``.
    """
    prompt_text, prompt_ids = _render_prompt(tokenizer, messages, enable_thinking)
    _, teacher_prompt_ids = _render_prompt(
        teacher_tokenizer, teacher_messages, enable_thinking
    )
    return _sample_speculative_text(
        model,
        tokenizer,
        prompt_text,
        prompt_ids,
        teacher_model,
        teacher_tokenizer,
        teacher_prompt_ids,
        top_k=top_k,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        teacher_without_adapter=teacher_without_adapter,
        max_block_size=max_block_size,
        batch_shared_model=batch_shared_model,
        capture_teacher_logits=capture_teacher_logits,
        on_decode_start=_on_decode_start,
    )


def sample_speculative_text(
    model,
    tokenizer,
    prompt_text: str,
    teacher_model,
    teacher_tokenizer,
    teacher_prompt_text: str,
    *,
    top_k: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
    teacher_without_adapter: bool = False,
    max_block_size: int = 8,
    batch_shared_model: bool = True,
    capture_teacher_logits: bool = False,
    _on_decode_start: Callable[[], None] | None = None,
) -> Sample:
    """Speculative sampling from raw student and teacher prefixes."""
    return _sample_speculative_text(
        model,
        tokenizer,
        prompt_text,
        tokenizer.encode(prompt_text, add_special_tokens=False),
        teacher_model,
        teacher_tokenizer,
        teacher_tokenizer.encode(teacher_prompt_text, add_special_tokens=False),
        top_k=top_k,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        teacher_without_adapter=teacher_without_adapter,
        max_block_size=max_block_size,
        batch_shared_model=batch_shared_model,
        capture_teacher_logits=capture_teacher_logits,
        on_decode_start=_on_decode_start,
    )


def _sample_speculative_text(
    model,
    tokenizer,
    prompt_text: str,
    prompt_ids: list[int],
    teacher_model,
    teacher_tokenizer,
    teacher_prompt_ids: list[int],
    *,
    top_k: int,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    teacher_without_adapter: bool,
    max_block_size: int,
    batch_shared_model: bool,
    capture_teacher_logits: bool,
    on_decode_start: Callable[[], None] | None = None,
) -> Sample:
    shared = (
        teacher_model is model
        and teacher_tokenizer is tokenizer
        and (not teacher_without_adapter or supports_lora_batch_routing(model))
    )
    if batch_shared_model and shared:
        try:
            return _sample_speculative_shared_model(
                model,
                tokenizer,
                prompt_text,
                prompt_ids,
                teacher_prompt_ids,
                top_k=top_k,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                teacher_without_adapter=teacher_without_adapter,
                capture_teacher_logits=capture_teacher_logits,
                on_decode_start=on_decode_start,
            )
        except _SharedModelBatchUnsupported:
            pass

    return _sample_speculative_generic(
        model,
        tokenizer,
        prompt_text,
        prompt_ids,
        teacher_model,
        teacher_tokenizer,
        teacher_prompt_ids,
        top_k=top_k,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        teacher_without_adapter=teacher_without_adapter,
        max_block_size=max_block_size,
        capture_teacher_logits=capture_teacher_logits,
        on_decode_start=on_decode_start,
    )


def _sample_speculative_shared_model(
    model,
    tokenizer,
    prompt_text: str,
    prompt_ids: list[int],
    teacher_prompt_ids: list[int],
    *,
    top_k: int,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    teacher_without_adapter: bool,
    capture_teacher_logits: bool,
    on_decode_start: Callable[[], None] | None,
) -> Sample:
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if top_k <= 0:
        raise ValueError("--speculative-top-k must be positive")
    if seed is not None:
        mx.random.seed(seed)

    student_cache = make_prompt_cache(model)
    teacher_cache = make_prompt_cache(model)
    student_sampler = make_sampler(temp=temperature)
    teacher_sampler = make_sampler(temp=temperature)
    eos_ids = _eos_token_ids(tokenizer)
    model.eval()

    def prefill(ids, cache, adapter=True):
        with lora_batch_enabled(model, [adapter]):
            logits = prefill_last_logits(model, ids, cache)
        return logits

    student_logits = prefill(prompt_ids, student_cache)
    teacher_logits = prefill(
        teacher_prompt_ids, teacher_cache, adapter=not teacher_without_adapter
    )
    if not all(hasattr(entry, "merge") for entry in student_cache):
        raise _SharedModelBatchUnsupported
    cache = [
        type(entry).merge([entry, other])
        for entry, other in zip(student_cache, teacher_cache)
    ]

    gen_ids: list[int] = []
    teacher_sampled: list[bool] = []
    captured_teacher_logits = []
    decode_started = False
    finish_reason = "length"

    while len(gen_ids) < max_tokens:
        if capture_teacher_logits:
            captured_teacher_logits.append(mx.stop_gradient(teacher_logits))
        proposal = int(student_sampler(student_logits).item())
        rng_state = list(mx.random.state) if temperature > 0 else None
        accepted = _proposal_is_accepted(proposal, teacher_logits, top_k=top_k)
        token = proposal
        if not accepted:
            if rng_state is not None:
                mx.random.state = rng_state
            token = int(teacher_sampler(teacher_logits).item())

        gen_ids.append(token)
        teacher_sampled.append(not accepted)
        if not decode_started and on_decode_start is not None:
            decode_started = True
            on_decode_start()
        if token in eos_ids:
            finish_reason = "stop"
            break
        if len(gen_ids) == max_tokens:
            break

        with lora_batch_enabled(model, [True, not teacher_without_adapter]):
            logits = model(mx.array([[token], [token]]), cache=cache)[:, -1].astype(
                mx.float32
            )
            mx.eval(logits, [entry.state for entry in cache])
        student_logits, teacher_logits = logits[0], logits[1]

    replaced = sum(teacher_sampled)
    teacher_rows = (
        mx.stack(captured_teacher_logits) if captured_teacher_logits else None
    )
    if teacher_rows is not None:
        mx.eval(teacher_rows)
    return Sample(
        prompt_text=prompt_text,
        completion_text=tokenizer.decode(gen_ids),
        content=tokenizer._tokenizer.decode(gen_ids, skip_special_tokens=True),
        finish_reason=finish_reason,
        num_tokens=len(gen_ids),
        speculative_top_k=top_k,
        accepted_tokens=len(gen_ids) - replaced,
        replaced_tokens=replaced,
        teacher_sampled=teacher_sampled,
        teacher_logits=teacher_rows,
    )


def _sample_speculative_generic(
    model,
    tokenizer,
    prompt_text: str,
    prompt_ids: list[int],
    teacher_model,
    teacher_tokenizer,
    teacher_prompt_ids: list[int],
    *,
    top_k: int,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    teacher_without_adapter: bool,
    max_block_size: int,
    capture_teacher_logits: bool,
    on_decode_start: Callable[[], None] | None = None,
) -> Sample:
    from mlx_lm.models.cache import make_prompt_cache  # deferred: slow import
    from mlx_lm.sample_utils import make_sampler

    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if top_k <= 0:
        raise ValueError("--speculative-top-k must be positive")
    if max_block_size <= 0:
        raise ValueError("max_block_size must be positive")
    if seed is not None:
        mx.random.seed(seed)

    if not prompt_ids or not teacher_prompt_ids:
        raise ValueError("student and teacher generation prompts must not be empty")

    student_sampler = make_sampler(temp=temperature)
    teacher_sampler = make_sampler(temp=temperature)
    student_cache = make_prompt_cache(model)
    teacher_cache = make_prompt_cache(teacher_model)
    eos_ids = _eos_token_ids(tokenizer) | _eos_token_ids(teacher_tokenizer)

    gen_ids: list[int] = []
    teacher_sampled: list[bool] = []
    captured_teacher_logits = []
    accepted_tokens = 0
    replaced_tokens = 0
    finish_reason = "length"
    model.eval()
    teacher_model.eval()

    def student_forward(input_ids):
        logits = model(input_ids[None], cache=student_cache)[0].astype(mx.float32)
        # MLX is lazy and LoRA scales are mutable. Materialize both the logits
        # and cache before temporarily disabling the shared model's adapter.
        mx.eval(logits, [entry.state for entry in student_cache])
        return logits

    def teacher_forward(input_ids):
        teacher_context = (
            lora_disabled(teacher_model) if teacher_without_adapter else nullcontext()
        )
        with teacher_context:
            logits = teacher_model(input_ids[None], cache=teacher_cache)[0].astype(
                mx.float32
            )
            mx.eval(logits, [entry.state for entry in teacher_cache])
        return logits

    def checkpoint(cache):
        return [
            (tree_map(lambda value: value, entry.state), entry.meta_state)
            for entry in cache
        ]

    def restore(cache, saved):
        for entry, (state, meta_state) in zip(cache, saved):
            entry.state = state
            entry.meta_state = meta_state

    student_logits = prefill_last_logits(model, prompt_ids, student_cache)
    teacher_context = (
        lora_disabled(teacher_model) if teacher_without_adapter else nullcontext()
    )
    with teacher_context:
        teacher_logits = prefill_last_logits(
            teacher_model, teacher_prompt_ids, teacher_cache
        )
    block_size = 1
    accepted_streak = 0
    decode_started = False

    while len(gen_ids) < max_tokens:
        draft_count = min(block_size, max_tokens - len(gen_ids))
        saved = (
            (checkpoint(student_cache), checkpoint(teacher_cache))
            if draft_count > 1
            else None
        )
        proposals: list[int] = []
        rng_states = []
        draft_logits = student_logits
        for index in range(draft_count):
            proposal = int(student_sampler(draft_logits).item())
            proposals.append(proposal)
            if temperature > 0:
                rng_states.append(list(mx.random.state))
            if proposal in eos_ids or index + 1 == draft_count:
                break
            draft_logits = student_forward(mx.array([proposal]))[-1]

        teacher_rows = [teacher_logits]
        if len(proposals) > 1:
            teacher_rows.extend(teacher_forward(mx.array(proposals[:-1])))

        rejected_at = None
        for index, (proposal, logits) in enumerate(zip(proposals, teacher_rows)):
            if capture_teacher_logits:
                captured_teacher_logits.append(mx.stop_gradient(logits))
            accepted = _proposal_is_accepted(
                proposal,
                logits,
                top_k=top_k,
            )
            token = proposal
            if not accepted:
                if temperature > 0:
                    mx.random.state = rng_states[index]
                token = int(teacher_sampler(logits).item())
                rejected_at = index

            student_piece = tokenizer.decode([token])
            teacher_piece = teacher_tokenizer.decode([token])
            if student_piece != teacher_piece:
                raise ValueError(
                    "teacher and student token vocabularies do not align for "
                    f"generated token {token}: {student_piece!r} != {teacher_piece!r}"
                )

            gen_ids.append(token)
            teacher_sampled.append(not accepted)
            accepted_tokens += int(accepted)
            replaced_tokens += int(not accepted)
            if not decode_started and on_decode_start is not None:
                decode_started = True
                on_decode_start()
            if token in eos_ids:
                finish_reason = "stop"
                break
            if not accepted or len(gen_ids) == max_tokens:
                break

        if finish_reason == "stop" or len(gen_ids) == max_tokens:
            break

        if rejected_at is not None:
            if rejected_at < len(proposals) - 1:
                restore(student_cache, saved[0])
                restore(teacher_cache, saved[1])
                selected = mx.array([*proposals[:rejected_at], gen_ids[-1]])
            else:
                selected = mx.array([gen_ids[-1]])
            block_size = 1
            accepted_streak = 0
        else:
            selected = mx.array([gen_ids[-1]])
            accepted_streak += len(proposals)
            if block_size > 1 or accepted_streak >= 6:
                block_size = min(max_block_size, block_size * 2)

        student_logits = student_forward(selected)[-1]
        teacher_logits = teacher_forward(selected)[-1]

    captured = (
        mx.stack(captured_teacher_logits) if captured_teacher_logits else None
    )
    if captured is not None:
        mx.eval(captured)
    return Sample(
        prompt_text=prompt_text,
        completion_text=tokenizer.decode(gen_ids),
        content=tokenizer._tokenizer.decode(gen_ids, skip_special_tokens=True),
        finish_reason=finish_reason,
        num_tokens=len(gen_ids),
        speculative_top_k=top_k,
        accepted_tokens=accepted_tokens,
        replaced_tokens=replaced_tokens,
        teacher_sampled=teacher_sampled,
        teacher_logits=captured,
    )


def sample_speculative_completions(
    model,
    tokenizer,
    messages_batch: list[list[dict]],
    teacher_model,
    teacher_tokenizer,
    teacher_messages_batch: list[list[dict]],
    *,
    top_k: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
    enable_thinking: bool | None = None,
    teacher_without_adapter: bool = False,
    teacher_prefix_ids: list[int] | None = None,
    teacher_prefix_cache=None,
) -> list[Sample]:
    """Generate several shared-model SKD rollouts in one decode batch.

    Student and privileged-teacher contexts share one copy of the model
    weights. Every decode forward contains all active student rows followed by
    all corresponding teacher rows. A reusable teacher-prefix cache may be
    supplied for the invariant document/system portion of the prompts.
    """
    if not messages_batch:
        return []
    if len(messages_batch) != len(teacher_messages_batch):
        raise ValueError("student and teacher rollout batches must have equal size")
    if model is not teacher_model or tokenizer is not teacher_tokenizer:
        raise ValueError("batched SKD currently requires a shared model and tokenizer")
    if teacher_without_adapter and not supports_lora_batch_routing(model):
        raise ValueError("the shared model does not support per-row LoRA routing")
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if top_k <= 0:
        raise ValueError("--speculative-top-k must be positive")
    if seed is not None:
        mx.random.seed(seed)

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    rendered_students = [
        _render_prompt(tokenizer, messages, enable_thinking)
        for messages in messages_batch
    ]
    rendered_teachers = [
        _render_prompt(teacher_tokenizer, messages, enable_thinking)
        for messages in teacher_messages_batch
    ]
    prompt_texts = [rendered[0] for rendered in rendered_students]
    student_ids = [rendered[1] for rendered in rendered_students]
    full_teacher_ids = [rendered[1] for rendered in rendered_teachers]

    student_caches = [make_prompt_cache(model) for _ in student_ids]
    teacher_caches = []
    teacher_suffixes = []
    prefix = list(teacher_prefix_ids or [])
    for ids in full_teacher_ids:
        use_prefix = bool(
            prefix
            and teacher_prefix_cache is not None
            and ids[: len(prefix)] == prefix
            and len(ids) > len(prefix)
        )
        teacher_caches.append(
            clone_prompt_cache(teacher_prefix_cache)
            if use_prefix
            else make_prompt_cache(model)
        )
        teacher_suffixes.append(ids[len(prefix) :] if use_prefix else ids)

    batch_size = len(student_ids)
    model.eval()
    student_prompt_logits = []
    teacher_prompt_logits = []
    for ids, cache in zip(student_ids, student_caches):
        with lora_batch_enabled(model, [True]):
            student_prompt_logits.append(prefill_last_logits(model, ids, cache))
    for ids, cache in zip(teacher_suffixes, teacher_caches):
        with lora_batch_enabled(model, [not teacher_without_adapter]):
            teacher_prompt_logits.append(prefill_last_logits(model, ids, cache))
    final_logits = mx.stack([*student_prompt_logits, *teacher_prompt_logits])
    mx.eval(final_logits)
    batch_cache = merge_prompt_caches([*student_caches, *teacher_caches])
    student_logits = final_logits[:batch_size]
    teacher_logits = final_logits[batch_size:]

    student_sampler = make_sampler(temp=temperature)
    teacher_sampler = make_sampler(temp=temperature)
    eos_ids = _eos_token_ids(tokenizer)
    active_ids = list(range(batch_size))
    generated = [[] for _ in range(batch_size)]
    provenance = [[] for _ in range(batch_size)]
    captured = [[] for _ in range(batch_size)]
    finish_reasons: list[str | None] = [None] * batch_size

    while active_ids:
        selected = []
        for row, original in enumerate(active_ids):
            captured[original].append(mx.stop_gradient(teacher_logits[row]))
            rng_state = list(mx.random.state) if temperature > 0 else None
            proposal = int(student_sampler(student_logits[row]).item())
            accepted = _proposal_is_accepted(
                proposal, teacher_logits[row], top_k=top_k
            )
            token = proposal
            if not accepted:
                if rng_state is not None:
                    mx.random.state = rng_state
                token = int(teacher_sampler(teacher_logits[row]).item())
            generated[original].append(token)
            provenance[original].append(not accepted)
            selected.append(token)
            if token in eos_ids:
                finish_reasons[original] = "stop"
            elif len(generated[original]) >= max_tokens:
                finish_reasons[original] = "length"

        keep = [
            row
            for row, original in enumerate(active_ids)
            if finish_reasons[original] is None
        ]
        if not keep:
            break

        old_size = len(active_ids)
        cache_indices = [*keep, *(old_size + row for row in keep)]
        for entry in batch_cache:
            entry.filter(cache_indices)
        next_tokens = [selected[row] for row in keep]
        next_size = len(keep)
        with lora_batch_enabled(
            model,
            [True] * next_size + [not teacher_without_adapter] * next_size,
        ):
            logits = model(
                mx.array([*next_tokens, *next_tokens])[:, None], cache=batch_cache
            )[:, -1].astype(mx.float32)
            mx.eval(logits, [entry.state for entry in batch_cache])
        student_logits = logits[:next_size]
        teacher_logits = logits[next_size:]
        active_ids = [active_ids[row] for row in keep]

    teacher_rows = [mx.stack(rows) for rows in captured]
    mx.eval(teacher_rows)
    results = []
    for index, token_ids in enumerate(generated):
        replaced = sum(provenance[index])
        results.append(
            Sample(
                prompt_text=prompt_texts[index],
                completion_text=tokenizer.decode(token_ids),
                content=tokenizer._tokenizer.decode(
                    token_ids, skip_special_tokens=True
                ),
                finish_reason=finish_reasons[index],
                num_tokens=len(token_ids),
                speculative_top_k=top_k,
                accepted_tokens=len(token_ids) - replaced,
                replaced_tokens=replaced,
                teacher_sampled=provenance[index],
                teacher_logits=teacher_rows[index],
            )
        )
    return results
