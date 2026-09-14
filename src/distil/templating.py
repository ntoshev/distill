"""Build the text to score and the character spans within it that should be scored.

All modes reduce to one primitive: a full text string plus a list of character
spans marking model-generated content. Working at the character level (rather
than concatenating token ids) avoids retokenization drift at message
boundaries — e.g. Qwen's generation prompt ends in "<think>\n" which merges
with a following "\n" into a single token when tokenized jointly.

Spans are derived from chat-template prefix renderings when possible. Some
templates, notably Qwen-style thinking templates, render already-recorded
intermediate assistant turns differently from a fresh generation prompt; in
that case we fall back to scoring the assistant message's recorded content when
it can be located exactly after the preceding conversation.
"""

from dataclasses import dataclass, field


@dataclass
class BuiltText:
    text: str
    spans: list[tuple[int, int]]  # char [start, end) ranges to score
    warnings: list[str] = field(default_factory=list)

    # Example with mlx-community/Qwen3.5-0.8B-4bit:
    # BuiltText(
    #     text=(
    #         "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
    #         "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    #         "2+2 equals 5.<|im_end|>\n"
    #     ),
    #     spans=[(70, 104)],
    #     warnings=[],
    # )
    # The span selects: "\n</think>\n\n2+2 equals 5.<|im_end|>"


def apply_teacher_context(
    messages: list[dict], system: str | None = None, prompt: str | None = None
) -> list[dict]:
    """Teacher-side conditioning: the same conversation under a different context.

    `system` replaces the system message's content (or a system message is
    inserted at the front if there is none); `prompt` replaces the first user
    message's content. Assistant turns — the tokens being scored — are never
    touched, so scored spans stay aligned between the two renderings.
    """
    out = [dict(m) for m in messages]
    if system is not None:
        for m in out:
            if m.get("role") == "system":
                m["content"] = system
                break
        else:
            out.insert(0, {"role": "system", "content": system})
    if prompt is not None:
        for m in out:
            if m.get("role") == "user":
                m["content"] = prompt
                break
        else:
            raise ValueError("--teacher-prompt: the conversation has no user message to replace")
    return out


def prompt_messages(prompt: str | None, system: str | None = None) -> list[dict]:
    """Build a simple prompt without conflating an absent and an empty user turn."""
    messages = [] if system is None else [{"role": "system", "content": system}]
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
    return messages


def render_generation_prompt(
    tokenizer, messages: list[dict], enable_thinking: bool | None = None
) -> str:
    """Render the exact generation prefix; callers encode the rendered string."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **_chat_template_kwargs(enable_thinking),
    )


def build_raw(prompt: str, completion: str) -> BuiltText:
    """No chat template: completion is scored as a continuation of prompt."""
    return BuiltText(text=prompt + completion, spans=[(len(prompt), len(prompt) + len(completion))])


def build_chat(
    tokenizer,
    prompt: str,
    completion: str,
    system: str | None = None,
    enable_thinking: bool | None = None,
) -> BuiltText:
    """Render prompt as a user message through the model's chat template and
    score the completion as the assistant turn."""
    messages = prompt_messages(prompt, system)
    messages.append({"role": "assistant", "content": completion})
    return build_rollout(tokenizer, messages, enable_thinking=enable_thinking)


def build_rollout(
    tokenizer, messages: list[dict], enable_thinking: bool | None = None
) -> BuiltText:
    """Render a full conversation and mark assistant turns for scoring.

    For each assistant message, the span start comes from rendering the
    preceding messages with add_generation_prompt=True and the span end from
    rendering through that message. Both renderings must be string prefixes of
    the full text; a boundary that fails the check (e.g. an intermediate
    assistant turn whose thinking the template strips on re-rendering) is
    skipped with a warning rather than guessed at.
    """
    messages = [_clean_message(m) for m in messages]
    template_kwargs = _chat_template_kwargs(enable_thinking)
    render = lambda msgs, gen: tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=gen, **template_kwargs
    )
    full = render(messages, False)
    spans: list[tuple[int, int]] = []
    warnings: list[str] = []

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        start_render = render(messages[:i], True)
        end_render = render(messages[: i + 1], False)
        if not (full.startswith(start_render) and full.startswith(end_render)):
            fallback = _content_span_after_prefix(tokenizer, messages, full, i, template_kwargs)
            if fallback is not None:
                spans.append(fallback)
                continue
            warnings.append(
                f"assistant message {i}: cannot locate exactly in the rendered "
                "conversation (the chat template re-renders intermediate turns, "
                "e.g. strips thinking); shown as context, not scored"
            )
            continue
        end = len(end_render)
        if i == len(messages) - 1:
            # trailing template glue (newline after the end-of-turn token) is
            # never generated by the model; the end-of-turn token itself is
            end = len(end_render.rstrip())
        spans.append((len(start_render), end))

    if not spans:
        warnings.append("no assistant spans could be located; nothing will be scored")
    return BuiltText(text=full, spans=spans, warnings=warnings)


def _chat_template_kwargs(enable_thinking: bool | None) -> dict:
    return {} if enable_thinking is None else {"enable_thinking": enable_thinking}


def _content_span_after_prefix(
    tokenizer, messages: list[dict], full: str, i: int, template_kwargs: dict
) -> tuple[int, int] | None:
    """Locate a rendered assistant turn by its recorded content.

    This is deliberately narrower than the primary prefix method: it scores only
    the content field, not hidden generation-prefix text or end-of-turn glue.
    """
    content = messages[i].get("content")
    if not content:
        return None
    prefix = tokenizer.apply_chat_template(
        messages[:i], tokenize=False, add_generation_prompt=False, **template_kwargs
    )
    if not full.startswith(prefix):
        return None
    start = full.find(content, len(prefix))
    if start < 0:
        return None
    end = start + len(content)
    return start, end


_MESSAGE_KEYS = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name")


def _clean_message(msg: dict) -> dict:
    """Keep only template-relevant keys and drop nulls (verifiers rollout rows
    carry reasoning_content/tool_calls as explicit nulls, which some chat
    templates would choke on or render)."""
    return {k: msg[k] for k in _MESSAGE_KEYS if msg.get(k) is not None}
