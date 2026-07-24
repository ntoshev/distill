"""Sample completions from the student or an interleaved teacher/student policy.

The rendered generation prompt and the decoded generated tokens are kept as
strings, so scoring a fresh sample is `build_raw(prompt_text, completion_text)`
— exact by construction, no re-templating. `content` (special tokens stripped)
is what goes into a saved rollout's assistant message.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import mlx.core as mx

from distil.adapters import lora_disabled


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


def _select_interleaved_token(
    student_logprobs,
    teacher_logprobs,
    *,
    top_k: int,
    student_sampler,
    teacher_sampler,
) -> tuple[int, bool]:
    """Accept the student proposal iff it is in the teacher's top-k support."""
    vocab = min(student_logprobs.shape[-1], teacher_logprobs.shape[-1])
    if not 0 < top_k <= vocab:
        raise ValueError(f"--speculative-top-k must be in [1, {vocab}]")

    student_logprobs = student_logprobs[:vocab]
    teacher_logprobs = teacher_logprobs[:vocab]
    proposal = int(student_sampler(student_logprobs).item())
    teacher_top = mx.argpartition(-teacher_logprobs, kth=top_k - 1)[:top_k]
    accepted = proposal in set(teacher_top.tolist())
    if accepted:
        return proposal, True
    return int(teacher_sampler(teacher_logprobs).item()), False


def sample_completion(
    model,
    tokenizer,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 1.0,
    seed: int | None = None,
    enable_thinking: bool | None = None,
) -> Sample:
    prompt_text, prompt_ids = _render_prompt(tokenizer, messages, enable_thinking)
    return _sample_text(
        model, tokenizer, prompt_text, prompt_ids,
        max_tokens=max_tokens, temperature=temperature, seed=seed,
    )


def sample_text(
    model,
    tokenizer,
    prompt_text: str,
    max_tokens: int = 512,
    temperature: float = 1.0,
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
) -> Sample:
    from mlx_lm import stream_generate  # deferred: slow import
    from mlx_lm.sample_utils import make_sampler

    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if seed is not None:
        mx.random.seed(seed)

    gen_ids: list[int] = []
    finish_reason = None
    for resp in stream_generate(
        model,
        tokenizer,
        prompt=prompt_ids,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
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
    temperature: float = 1.0,
    seed: int | None = None,
    enable_thinking: bool | None = None,
    teacher_without_adapter: bool = False,
) -> Sample:
    """Generate with Speculative KD's token-level interleaved policy.

    The student proposes each token. The proposal is accepted when it belongs
    to the teacher distribution's top ``top_k`` tokens at the corresponding
    privileged-context prefix; otherwise a token is sampled from the teacher.
    The selected token is then fed back to both policies for the next step.
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
    temperature: float = 1.0,
    seed: int | None = None,
    teacher_without_adapter: bool = False,
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
) -> Sample:
    from mlx_lm.models.cache import make_prompt_cache  # deferred: slow import
    from mlx_lm.sample_utils import make_sampler

    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if top_k <= 0:
        raise ValueError("--speculative-top-k must be positive")
    if seed is not None:
        mx.random.seed(seed)

    if not prompt_ids or not teacher_prompt_ids:
        raise ValueError("student and teacher generation prompts must not be empty")

    student_sampler = make_sampler(temp=temperature)
    teacher_sampler = make_sampler(temp=temperature)
    student_cache = make_prompt_cache(model)
    teacher_cache = make_prompt_cache(teacher_model)
    student_input = mx.array(prompt_ids)
    teacher_input = mx.array(teacher_prompt_ids)
    eos_ids = _eos_token_ids(tokenizer) | _eos_token_ids(teacher_tokenizer)

    gen_ids: list[int] = []
    teacher_sampled: list[bool] = []
    accepted_tokens = 0
    replaced_tokens = 0
    finish_reason = "length"
    model.eval()
    teacher_model.eval()

    for _ in range(max_tokens):
        student_logits = model(student_input[None], cache=student_cache)[0, -1].astype(
            mx.float32
        )
        # MLX is lazy and LoRA scales are mutable. Materialize both the logits
        # and cache before temporarily disabling the shared model's adapter.
        mx.eval(student_logits, [entry.state for entry in student_cache])

        teacher_context = (
            lora_disabled(teacher_model) if teacher_without_adapter else nullcontext()
        )
        with teacher_context:
            teacher_logits = teacher_model(
                teacher_input[None], cache=teacher_cache
            )[0, -1].astype(mx.float32)
            mx.eval(teacher_logits, [entry.state for entry in teacher_cache])

        student_logprobs = student_logits - mx.logsumexp(student_logits)
        teacher_logprobs = teacher_logits - mx.logsumexp(teacher_logits)
        token, accepted = _select_interleaved_token(
            student_logprobs,
            teacher_logprobs,
            top_k=top_k,
            student_sampler=student_sampler,
            teacher_sampler=teacher_sampler,
        )

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
        if token in eos_ids:
            finish_reason = "stop"
            break
        student_input = mx.array([token])
        teacher_input = mx.array([token])

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
    )
