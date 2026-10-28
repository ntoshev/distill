"""Real-model equivalence checks for combined scoring and batched generation."""

from dataclasses import replace

import mlx.core as mx
import pytest

from distill.judge import judge_answer, judge_answers
from distill.progress import Probe, evaluate_probes, sample_probe, score_probe
from distill.sample import _sample_batch, sample_completions
from distill.stopping import answer_only_prompt


TARGET = Probe(
    "target",
    answer_only_prompt(
        "Give the launch code assigned to Project Lumen. Answer with the code only."
    ),
    "VIOLET-731",
)
CANARIES = [
    Probe("math", "What is 7 plus 5? Answer with the number only.", "", dynamic=True),
    Probe(
        "capital",
        "What is the capital of France? Answer with the city only.",
        "",
        dynamic=True,
    ),
    Probe(
        "opposite",
        "What is the opposite of hot? Answer with one word only.",
        "",
        dynamic=True,
    ),
    Probe(
        "grass",
        "What color is grass usually? Answer with the color only.",
        "",
        dynamic=True,
    ),
]


def sequential_eval(model, tokenizer):
    fixed = score_probe(model, tokenizer, TARGET, 0, False)
    answers = [
        sample_probe(model, tokenizer, probe, 0, False, max_tokens=16, temperature=0)
        for probe in [replace(TARGET, dynamic=True), *CANARIES]
    ]
    verdicts = [
        judge_answer(
            model,
            tokenizer,
            TARGET.prompt,
            TARGET.completion,
            answers[0].content.strip(),
            enable_thinking=False,
        )
    ]
    mx.synchronize()
    return [fixed, *answers], verdicts


def batched_eval(model, tokenizer):
    probes = [TARGET, replace(TARGET, dynamic=True), *CANARIES]
    points = evaluate_probes(
        model, tokenizer, probes, 0, False, max_tokens=[16] * len(probes)
    )
    verdicts = judge_answers(
        model,
        tokenizer,
        [(TARGET.completion, points[1].content.strip())],
        enable_thinking=False,
    )
    mx.synchronize()
    return points, verdicts


@pytest.mark.model
@pytest.mark.parametrize("fixture", ["model_and_tokenizer", "model_4b_and_tokenizer"])
def test_combined_eval_matches_sequential_and_does_not_rescore(
    request, fixture, monkeypatch
):
    model, tokenizer = request.getfixturevalue(fixture)
    expected, expected_verdicts = sequential_eval(model, tokenizer)

    def no_rescore(*args, **kwargs):
        pytest.fail("batched evaluation must reuse decode probabilities")

    import distill.progress as progress

    monkeypatch.setattr(progress, "score_generated", no_rescore)
    monkeypatch.setattr(progress, "score_text", no_rescore)
    actual, verdicts = batched_eval(model, tokenizer)
    assert verdicts == expected_verdicts
    for left, right in zip(actual, expected):
        assert left.content == right.content
        assert left.completion == right.completion
        assert left.finish_reason == right.finish_reason
        assert left.num_tokens == right.num_tokens
        assert [t["token_id"] for t in left.tokens] == [
            t["token_id"] for t in right.tokens
        ]
        assert [t["text"] for t in left.tokens] == [t["text"] for t in right.tokens]
        # Quantized kernels differ between full-sequence scoring and cached
        # decoding. The float32 test below checks probabilities to 1e-5.
        assert [t["probability"] for t in left.tokens] == pytest.approx(
            [t["probability"] for t in right.tokens],
            abs=0.12,
        )
        assert left.mean_probability == pytest.approx(right.mean_probability, abs=0.12)


@pytest.mark.model
def test_batch_keeps_per_row_limits_eos_and_order(model, tokenizer):
    points = evaluate_probes(
        model,
        tokenizer,
        [replace(TARGET, dynamic=True), CANARIES[0], CANARIES[1]],
        5,
        False,
        max_tokens=[1, 16, 1],
    )
    assert [p.num_tokens for p in points] == [1, 3, 1]
    assert [p.finish_reason for p in points] == ["length", "stop", "length"]
    assert points[1].content == "12"
    assert points[1].tokens[-1]["special"]
    assert all(p.step == 5 for p in points)


@pytest.mark.model
def test_fixed_scoring_includes_tokens_after_eos_and_boundary_merges(model, tokenizer):
    probes = [
        Probe("boundary", "Continue", "suffix"),
        Probe("eos", "Answer?", "wrong<|im_end|>tail"),
    ]
    expected = [score_probe(model, tokenizer, p, 0, False) for p in probes]
    actual = evaluate_probes(model, tokenizer, probes, 0, False, max_tokens=[1, 1])
    for left, right in zip(actual, expected):
        assert [t["text"] for t in left.tokens] == [t["text"] for t in right.tokens]
        assert left.mean_logprob == pytest.approx(
            right.mean_logprob, rel=0.05, abs=0.03
        )


@pytest.mark.model
def test_batch_judge_matches_sequential(model, tokenizer):
    pairs = [("VIOLET-731", "VIOLET-731"), ("VIOLET-731", "P-001"), ("10 goals", "10")]
    expected = [
        judge_answer(model, tokenizer, "", a, b, enable_thinking=False)
        for a, b in pairs
    ]
    assert judge_answers(model, tokenizer, pairs, enable_thinking=False) == expected


@pytest.mark.model
def test_batch_preserves_custom_prefix_caches(cartridge_model_and_tokenizer):
    from distill.cartridge import Cartridge

    model, tokenizer, identity = cartridge_model_and_tokenizer
    memory = Cartridge.initialize(
        model, tokenizer, "A quiet river.", 4, identity=identity
    )
    probes = [Probe("fixed", "Answer 2+2", "4"), CANARIES[0], CANARIES[1]]
    kwargs = dict(cache_factory=memory.new_cache)
    expected = [score_probe(model, tokenizer, probes[0], 0, False, **kwargs)] + [
        sample_probe(
            model, tokenizer, p, 0, False, max_tokens=16, temperature=0, **kwargs
        )
        for p in probes[1:]
    ]
    actual = evaluate_probes(
        model, tokenizer, probes, 0, False, max_tokens=[16] * 3, **kwargs
    )
    for left, right in zip(actual, expected):
        assert left.content == right.content
        assert left.mean_logprob == pytest.approx(
            right.mean_logprob, rel=0.05, abs=0.03
        )


def test_empty_batch_and_invalid_limits():
    assert sample_completions(None, None, []) == []
    assert evaluate_probes(None, None, [], 0, False, max_tokens=[]) == []
    for limits in [[], [0], [-1]]:
        with pytest.raises(ValueError, match="positive token limit"):
            _sample_batch(None, None, [("p", [1])], limits)


def test_fixed_scoring_preserves_token_merging_across_prompt_boundary(monkeypatch):
    from distill.sample import Sample

    class Tokenizer:
        bos_token_id = None

        @property
        def _tokenizer(self):
            return self

        def __call__(self, text, **kwargs):
            assert text == "ABC"
            return {"input_ids": [1, 3], "offset_mapping": [(0, 1), (1, 3)]}

    def batch(model, tokenizer, rendered, limits, **kwargs):
        assert rendered == [("AB", [1])]
        assert limits == [1]
        assert kwargs["forced_token_ids"] == [[3]]
        return [
            Sample("AB", "BC", "BC", "length", 1, token_ids=[3], token_logprobs=[-0.7])
        ]

    monkeypatch.setattr("distill.progress._render_prompt", lambda *a: ("AB", [1, 2]))
    monkeypatch.setattr("distill.progress._sample_batch", batch)
    (point,) = evaluate_probes(
        None, Tokenizer(), [Probe("p", "p", "C")], 0, False, max_tokens=[1]
    )
    assert point.tokens[0]["text"] == "BC"
    assert point.tokens[0]["token_id"] == 3
    assert point.tokens[0]["logprob"] == -0.7


def test_batch_probabilities_match_float32_forward_and_compact_finished_rows():
    from mlx_lm.models.llama import Model, ModelArgs
    from distill.scoring import score_ids

    mx.random.seed(42)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=1,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=32,
        )
    )

    class Tokenizer:
        eos_token_ids = {31}

        @property
        def _tokenizer(self):
            return self

        def decode(self, ids, skip_special_tokens=False):
            return "".join(
                chr(65 + i) for i in ids if not skip_special_tokens or i != 31
            )

    class TracedModel:
        def __init__(self):
            self.batches = []

        def __getattr__(self, name):
            return getattr(model, name)

        def __call__(self, inputs, **kwargs):
            self.batches.append(inputs.shape[0])
            return model(inputs, **kwargs)

    traced = TracedModel()
    prompts = [[1, 2, 3], [5], [7, 8]]
    samples = _sample_batch(
        traced,
        Tokenizer(),
        [("p", ids) for ids in prompts],
        [3, 1, 5],
        forced_token_ids=[[12, 31, 9], None, None],
    )
    assert samples[0].token_ids == [12, 31, 9]  # EOS must not truncate fixed scoring.
    assert samples[1].num_tokens == 1
    assert max(traced.batches) >= 2
    assert traced.batches[-1] == 1
    for prompt, sample in zip(prompts, samples):
        probabilities, _ = score_ids(model, prompt + sample.token_ids, top_k=0)
        assert sample.token_logprobs == pytest.approx(
            [probabilities[len(prompt) + i] for i in range(sample.num_tokens)], abs=1e-5
        )
