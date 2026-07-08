"""Unit tests for span construction. Uses fake tokenizers, so no model/cache needed."""

from distil.templating import build_chat, build_raw, build_rollout


class FakeTokenizer:
    """Minimal chat template: <s>{role}\\n{content}</s>\\n per message."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        out = "".join(f"<s>{m['role']}\n{m['content']}</s>\n" for m in messages)
        if add_generation_prompt:
            out += "<s>assistant\n"
        return out


class ThinkingFakeTokenizer:
    """Mimics Qwen-style templates: the generation prompt opens a think block,
    and intermediate assistant turns are re-rendered without their thinking."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        out = ""
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                if i == len(messages) - 1:
                    think = m.get("reasoning_content", "")
                    out += f"<s>assistant\n<think>\n{think}\n</think>\n{m['content']}</s>\n"
                else:
                    out += f"<s>assistant\n{m['content']}</s>\n"
            else:
                out += f"<s>{m['role']}\n{m['content']}</s>\n"
        if add_generation_prompt:
            out += "<s>assistant\n<think>\n"
        return out


def test_build_raw():
    built = build_raw("abc", "defg")
    assert built.text == "abcdefg"
    assert built.spans == [(3, 7)]
    assert built.warnings == []


def test_build_chat_span_covers_completion():
    built = build_chat(FakeTokenizer(), "hi", "hello!", system="be nice")
    assert built.text == (
        "<s>system\nbe nice</s>\n<s>user\nhi</s>\n<s>assistant\nhello!</s>\n"
    )
    (start, end) = built.spans[0]
    # scored: the completion and the end-of-turn marker, not the trailing glue newline
    assert built.text[start:end] == "hello!</s>"
    assert built.warnings == []


def test_rollout_multiturn_all_assistant_spans():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    built = build_rollout(FakeTokenizer(), msgs)
    assert [built.text[s:e] for s, e in built.spans] == ["a1</s>\n", "a2</s>"]
    assert built.warnings == []


def test_rollout_thinking_template_final_turn_exact():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "hmm"},
    ]
    built = build_rollout(ThinkingFakeTokenizer(), msgs)
    (start, end) = built.spans[0]
    # everything the model generated after the forced think-opening
    assert built.text[start:end] == "hmm\n</think>\na</s>"
    assert built.warnings == []


def test_rollout_thinking_template_scores_intermediate_recorded_content():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "hmm"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2", "reasoning_content": "mmh"},
    ]
    built = build_rollout(ThinkingFakeTokenizer(), msgs)
    assert [built.text[s:e] for s, e in built.spans] == ["a1", "mmh\n</think>\na2</s>"]
    assert built.warnings == []


def test_rollout_null_fields_dropped():
    # verifiers rows carry explicit nulls; the template must not see them
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": None, "tool_calls": None},
    ]
    built = build_rollout(FakeTokenizer(), msgs)
    assert built.text.endswith("<s>assistant\na</s>\n")
