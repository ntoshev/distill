"""Sample a completion from the student model with mlx-lm.

The rendered generation prompt and the decoded generated tokens are kept as
strings, so scoring a fresh sample is `build_raw(prompt_text, completion_text)`
— exact by construction, no re-templating. `content` (special tokens stripped)
is what goes into a saved rollout's assistant message.
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Sample:
    prompt_text: str  # chat-template rendering of the messages, incl. generation prompt
    completion_text: str  # everything generated, incl. the end-of-turn token
    content: str  # generated text with special tokens stripped (for saving)
    finish_reason: str | None  # "stop" | "length"
    num_tokens: int


def sample_completion(
    model,
    tokenizer,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 1.0,
    seed: int | None = None,
    enable_thinking: bool | None = None,
) -> Sample:
    from mlx_lm import stream_generate  # deferred: slow import
    from mlx_lm.sample_utils import make_sampler

    if seed is not None:
        mx.random.seed(seed)
    template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    # encode the rendered string (not apply_chat_template(tokenize=True)) so the
    # ids match what scoring's tokenize_with_offsets will see for the same text
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

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
