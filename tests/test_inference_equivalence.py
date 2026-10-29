"""Differential checks against the independent, non-batched generation path.

Compare complete predictive distributions, not just argmax tokens or verdicts.
Tiny float32 models keep the numerical tolerance strict and the matrix cheap;
the model tests additionally exercise real tokenizers and quantized weights.
"""

from dataclasses import dataclass
import random

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.models.cache import make_prompt_cache
import pytest

from distill.adapters import add_lora, lora_modules
from distill.prefill import prefill_cache
from distill.sample import Sample, _render_prompt, _sample_batch, sample_completions


class _Tokenizer:
    eos_token_ids = {30, 31}

    @property
    def _tokenizer(self):
        return self

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            chr(256 + tid) for tid in ids
            if not skip_special_tokens or tid not in self.eos_token_ids
        )


@dataclass
class _Case:
    lengths: list[int]
    limits: list[int]
    forced: list[list[int] | None] | None = None
    duplicates: bool = False
    eos_first: bool = False
    adapted: bool = False
    prefill_step_size: int = 2048


CASES = {
    "singleton": _Case([7], [6]),
    "ragged": _Case([2, 9, 25], [1, 4, 8]),
    "equal-length": _Case([9, 9, 9], [1, 5, 3]),
    "duplicates": _Case([4, 12, 4, 12], [6, 6, 6, 6], duplicates=True),
    "single-token-prompt": _Case([1, 17, 3], [4, 2, 6]),
    "forced-and-greedy": _Case(
        [21, 3, 12, 1], [6, 1, 3, 5], [None, [31], [7, 31, 10], None],
    ),
    "early-eos": _Case(
        [3, 11, 5], [8, 7, 1],
        [None, [30, 31, 8, 9, 10, 11, 12], None], eos_first=True,
    ),
    "many-finished-rows": _Case(
        [2, 51, 7, 33, 5, 19, 3, 45], [1, 3, 7, 2, 9, 4, 6, 5],
    ),
    "lora": _Case([3, 8, 31], [5, 3, 7], adapted=True),
    "chunked-prefill": _Case([7, 8, 9, 17, 31], [3, 1, 5, 2, 4], prefill_step_size=8),
}


def _tiny_model(architecture, seed, adapted):
    mx.random.seed(seed)
    if architecture == "llama":
        from mlx_lm.models.llama import Model, ModelArgs

        model = Model(ModelArgs(
            model_type="llama", hidden_size=32, num_hidden_layers=2,
            intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
            rms_norm_eps=1e-6, vocab_size=32,
        ))
    else:
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        model = TextModel(TextModelArgs(
            model_type="qwen3_5", hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, full_attention_interval=2,
            linear_num_value_heads=2, linear_num_key_heads=1,
            linear_key_head_dim=32, linear_value_head_dim=32, vocab_size=32,
            rope_parameters={"partial_rotary_factor": 1.0, "rope_theta": 10000},
        ))
        # Exercise persistent recurrent history, as learned decay weights do.
        model.layers[0].linear_attn.A_log = mx.full((2,), -3.0)
    if adapted:
        add_lora(model, rank=2, num_layers=2)
        for module in lora_modules(model):
            module.lora_b = 0.02 * mx.random.normal(module.lora_b.shape)
    model.eval()
    mx.eval(model.parameters())
    return model


def _single(model, tokenizer, rendered, limit, forced, cache_factory, step_size):
    """Use the same single-sequence engine as stream_generate, with fresh caches."""
    text, ids = rendered
    remaining = iter(forced) if forced is not None else None

    def sampler(logprobs):
        if remaining is not None:
            return mx.array([next(remaining, forced[-1])])
        return mx.argmax(logprobs, axis=-1)

    generator = generate_step(
        mx.array(ids), model, max_tokens=limit, sampler=sampler,
        prompt_cache=cache_factory() if cache_factory else None,
        prefill_step_size=step_size,
        logits_processors=[lambda tokens, logits: logits.astype(mx.float32)],
    )
    tokens, distributions = [], []
    reason = "length"
    try:
        for tid, logprobs in generator:
            mx.eval(logprobs)
            tokens.append(tid)
            distributions.append(logprobs)
            if forced is None and tid in tokenizer.eos_token_ids:
                reason = "stop"
                break
    finally:
        generator.close()
    return Sample(
        text, tokenizer.decode(tokens),
        tokenizer._tokenizer.decode(tokens, skip_special_tokens=True),
        reason, len(tokens), token_ids=tokens,
        token_logprobs=[lp[tid].item() for lp, tid in zip(distributions, tokens)],
    ), distributions


def _observe_batch(monkeypatch, step_size):
    """Observe real generator responses without changing sampling or logits."""
    history = {}
    original_next = BatchGenerator.next_generated
    original_init = BatchGenerator.__init__

    def initialize(self, *args, **kwargs):
        kwargs["prefill_step_size"] = step_size
        original_init(self, *args, **kwargs)

    def observe(self):
        responses = original_next(self)
        for response in responses:
            mx.eval(response.logprobs)
            history.setdefault(response.uid, []).append(response.logprobs)
        return responses

    monkeypatch.setattr(BatchGenerator, "__init__", initialize)
    monkeypatch.setattr(BatchGenerator, "next_generated", observe)
    return history


def _assert_equivalent(actual, expected, distributions, reference, context, atol):
    for field in (
        "prompt_text", "token_ids", "completion_text", "content",
        "finish_reason", "num_tokens",
    ):
        assert getattr(actual, field) == getattr(expected, field), f"{context}: {field}"
    assert actual.token_logprobs == pytest.approx(expected.token_logprobs, abs=atol), context
    assert len(distributions) == len(reference) == actual.num_tokens, context
    for step, (left, right) in enumerate(zip(distributions, reference)):
        difference = mx.abs(left - right)
        error = mx.max(difference).item()
        assert error <= atol, f"{context}: distribution at token {step}, max error={error}"
        assert actual.token_logprobs[step] == pytest.approx(
            left[actual.token_ids[step]].item(), abs=1e-7,
        ), f"{context}: selected-token probability at token {step}"


@pytest.mark.parametrize("architecture", ["llama", "qwen3_5"])
@pytest.mark.parametrize("seed", [0, 17, 42])
@pytest.mark.parametrize("with_prefix", [False, True], ids=["fresh", "cached-prefix"])
@pytest.mark.parametrize("case_name", CASES)
def test_batched_inference_matches_single(monkeypatch, architecture, seed, with_prefix, case_name):
    case = CASES[case_name]
    model = _tiny_model(architecture, seed, case.adapted)
    tokenizer = _Tokenizer()
    rng = random.Random(seed)
    prompts = [[rng.randrange(1, 30) for _ in range(n)] for n in case.lengths]
    if case.duplicates:
        prompts[2], prompts[3] = prompts[0][:], prompts[1][:]
    rendered = [(tokenizer.decode(ids), ids) for ids in prompts]

    def cache_factory():
        cache = make_prompt_cache(model)
        prefill_cache(model, [4, 2, 7, 1, 8], cache, step_size=case.prefill_step_size)
        return cache

    factory = cache_factory if with_prefix else None
    forced = case.forced or [None] * len(prompts)
    if case.eos_first:
        first, _ = _single(model, tokenizer, rendered[0], 1, None, factory, case.prefill_step_size)
        tokenizer.eos_token_ids = {30, 31, first.token_ids[0]}
    expected = [
        _single(model, tokenizer, row, limit, ids, factory, case.prefill_step_size)
        for row, limit, ids in zip(rendered, case.limits, forced)
    ]
    if case.eos_first:
        assert expected[0][0].finish_reason == "stop"
        assert expected[0][0].num_tokens == 1
        assert expected[1][0].num_tokens == case.limits[1]

    history = _observe_batch(monkeypatch, case.prefill_step_size)
    # Reverse row order and repeat the first order to expose routing/state leaks.
    order = list(range(len(prompts)))
    for permutation in (order, order[::-1], order):
        history.clear()
        actual = _sample_batch(
            model, tokenizer, [rendered[i] for i in permutation],
            [case.limits[i] for i in permutation],
            forced_token_ids=[forced[i] for i in permutation], cache_factory=factory,
        )
        assert len(actual) == len(permutation)
        for uid, (point, index) in enumerate(zip(actual, permutation)):
            reference, probabilities = expected[index]
            context = f"{architecture}/{seed}/{case_name}/prefix={with_prefix}/order={permutation}/row={index}"
            _assert_equivalent(point, reference, history[uid], probabilities, context, atol=2e-5)


@pytest.mark.model
@pytest.mark.parametrize("model_name", [
    "mlx-community/Qwen3.5-0.8B-6bit",
    "mlx-community/Qwen3.5-4B-4bit",
])
def test_pretrained_batched_inference_matches_single(monkeypatch, model_name):
    from mlx_lm import load
    from distill.judge import _judge_messages

    model, tokenizer = load(model_name)
    # Retain quantized integer weights; promote floating parameters/activations
    # so GEMM/GEMV rounding does not require a broad probability tolerance.
    model.set_dtype(mx.float32)
    model.eval()
    mx.eval(model.parameters())

    def user(text):
        return [{"role": "user", "content": text}]

    math = user("What is 7 plus 5? Answer with the number only.")
    messages = [
        math,
        user("What is the capital of France? Answer with the city only."),
        user("What is the opposite of hot? Answer with one word only."),
        user("What color is grass usually? Answer with the color only."),
        [{"role": "system", "content": "Reply with exactly the word PING."}, *user("Respond.")],
        [
            {"role": "system", "content": "Project Lumen's launch code is VIOLET-731."},
            *user("What is Project Lumen's launch code? Answer with the code only."),
        ],
        math,
        [
            *user("What is the capital of Italy?"),
            {"role": "assistant", "content": "Rome."},
            *user("Repeat just the city."),
        ],
        _judge_messages("VIOLET-731", "VIOLET-732"),
        _judge_messages("10 goals", "10"),
    ]
    limits = [4, 4, 4, 4, 2, 10, 1, 4, 2, 2]
    rendered = [_render_prompt(tokenizer, row, False) for row in messages]
    expected = [
        _single(model, tokenizer, row, limit, None, None, 2048)
        for row, limit in zip(rendered, limits)
    ]
    history = _observe_batch(monkeypatch, 2048)
    order = list(range(len(messages)))
    for permutation in (order, order[::-1]):
        history.clear()
        actual = sample_completions(
            model, tokenizer, [messages[i] for i in permutation],
            max_tokens=[limits[i] for i in permutation], enable_thinking=False,
        )
        assert len(actual) == len(permutation)
        for uid, (point, index) in enumerate(zip(actual, permutation)):
            reference, probabilities = expected[index]
            _assert_equivalent(
                point, reference, history[uid], probabilities,
                f"{model_name}/order={permutation}/row={index}", atol=1e-3,
            )
