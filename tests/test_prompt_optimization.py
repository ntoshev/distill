"""Independent checks of hard-token optimization, masking, and generalization splits."""

import random
import math
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest

from distil.prompt_optimization import (GCG, PEZ, SoftPrompt, PromptLayout, PromptObjective,
    PromptVocabulary, ReversalCase, evaluate_prompt, evaluate_soft_prompt, reversal_splits, wilson_interval)


class CharacterTokenizer:
    vocab_size = 97
    eos_token_id = 0
    _tokenizer = SimpleNamespace(all_special_ids=[0, 96])

    def encode(self, text, add_special_tokens=False):
        return [96 if c == "\n" else ord(c) - 31 for c in text]

    def decode(self, ids):
        return "".join("\n" if i == 96 else chr(i + 31) for i in ids if i != 0)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "system\n" + messages[0]["content"] + "\nuser\n" + messages[1]["content"] + "\nanswer\n"


class TinyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(97, 8)
        self.projection = nn.Linear(8, 8, bias=False)

    def __call__(self, ids, input_embeddings=None):
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        # A causal model with a genuine dependency on all earlier prompt tokens.
        # Use an independent dense causal sum: MLX 0.31.2's strided GPU scan
        # backward corrupts buffers for this tiny shape (CPU is unaffected).
        causal = (mx.arange(h.shape[1])[:, None] >= mx.arange(h.shape[1])[None, :]).astype(h.dtype)
        return mx.tanh(self.projection((causal @ h) / 10))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = "qwen3"
        self.args = SimpleNamespace(tie_word_embeddings=False)
        self.model = TinyTrunk()
        self.lm_head = nn.Linear(8, 97, bias=False)

    def __call__(self, ids, input_embeddings=None):
        return self.lm_head(self.model(ids, input_embeddings=input_embeddings))


@pytest.fixture
def tiny():
    mx.random.seed(4)
    tok = CharacterTokenizer()
    model = TinyModel()
    layout = PromptLayout(tok)
    obj = PromptObjective(model, tok, layout)
    vocab = PromptVocabulary(tok, model.model.embed_tokens, layout)
    return model, tok, layout, obj, vocab


def test_reversal_preserves_duplicates():
    c = ReversalCase("duplicate", ("apple", "river", "apple", "chair"))
    assert c.question == "apple river apple chair"
    assert c.answer == "chair apple river apple"
    assert c.to_dict()["words"] == ["apple", "river", "apple", "chair"]


def test_dataset_has_fresh_inputs_and_real_word_and_length_shifts():
    tok = SimpleNamespace(encode=lambda text, **kw: text.split())
    splits = reversal_splits(tok)
    assert splits == reversal_splits(tok)
    seqs = [c.words for cases in splits.values() for c in cases]
    assert len(set(seqs)) == len(seqs)
    assert all(seq != seq[::-1] for seq in seqs)
    words = lambda name: {w for c in splits[name] for w in c.words}
    assert not words("train") & words("unseen_words")
    assert words("validation") <= words("train")
    assert words("test") <= words("train")
    assert all(6 <= len(c.words) <= 8 for c in splits["longer"])
    assert all(3 <= len(c.words) <= 5 for c in splits["train"])


def test_layout_matches_exported_prompt(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    assert layout.valid(ids)
    assert layout.context(ids, "a b c") == tok.encode(layout.render("Go.", "a b c"))
    # Non-rendering EOS cannot hide inside the claimed prompt budget.
    assert not layout.valid([0, *ids])


def test_dictionary_vocabulary_keeps_word_forms_and_punctuation(tiny):
    model, tok, layout, obj, vocab = tiny
    restricted = PromptVocabulary(tok, model.model.embed_tokens, layout, allowed_words={"g", "o"})
    assert all(i in restricted.index for i in tok.encode("Go."))
    assert tok.encode("A")[0] not in restricted.index
    assert tok.encode("7")[0] not in restricted.index
    assert restricted.project(restricted.embed(tok.encode("Go."))) == tok.encode("Go.")


def dense_loss(model, tok, layout, ids, case):
    context = layout.context(ids, case.question)
    targets = tok.encode(case.answer) + [tok.eos_token_id]
    logits = model(mx.array(context + targets[:-1])[None])[0, len(context) - 1:]
    y = mx.array(targets)
    return (mx.logsumexp(logits, axis=-1)
            - mx.take_along_axis(logits, y[:, None], axis=-1)[:, 0]).mean()


def test_mask_alignment_eos_padding_and_sequence_average(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    cases = [ReversalCase("short", ("a", "b", "c")),
             ReversalCase("long", ("a", "bb", "ccc", "dd"))]
    expected = sum(dense_loss(model, tok, layout, ids, c).item() for c in cases) / len(cases)
    score = obj.score([ids], cases)[0]
    loss, grad = obj.loss_and_grad(ids, cases)
    assert score == pytest.approx(expected, abs=1e-6)
    assert loss == pytest.approx(expected, abs=1e-6)
    assert grad.shape == (3, 8)
    assert mx.linalg.norm(grad).item() > 0
    assert not tree_flatten(model.trainable_parameters())
    # Candidate batching, padding, and microbatch boundaries preserve the loss.
    other = tok.encode("Do!")
    expected_other = sum(dense_loss(model, tok, layout, other, c).item() for c in cases) / len(cases)
    assert obj.score([ids, other], cases, microbatch=3) == pytest.approx([expected, expected_other], abs=1e-6)


def test_prompt_gradient_agrees_with_independent_dense_gradient_and_fd(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    case = ReversalCase("a", ("a", "b", "c"))
    loss, grad = obj.loss_and_grad(ids, [case])
    batch = obj.batch([ids], [case])
    vectors = vocab.embed(ids)
    context = layout.context(ids, case.question)
    targets = tok.encode(case.answer) + [tok.eos_token_id]
    sequence = mx.array(context + targets[:-1])[None]
    start = len(layout.prefix)
    def independent_dense(vectors):
        embeds = model.model.embed_tokens(sequence)
        embeds = mx.concatenate([embeds[:, :start], vectors[None], embeds[:, start + len(ids):]], axis=1)
        logits = model(sequence, input_embeddings=embeds)[0, len(context) - 1:]
        y = mx.array(targets)
        return (mx.logsumexp(logits, -1) - mx.take_along_axis(logits, y[:, None], -1)[:, 0]).mean()
    _, independent_grad = mx.value_and_grad(independent_dense)(vectors)
    assert mx.allclose(grad, independent_grad, atol=1e-7).item()
    direction = mx.arange(vectors.size).reshape(vectors.shape).astype(mx.float32) / vectors.size
    eps = .01
    plus = obj.losses(vectors + eps * direction, batch).mean().item()
    minus = obj.losses(vectors - eps * direction, batch).mean().item()
    assert (plus - minus) / (2 * eps) == pytest.approx((grad * direction).sum().item(), rel=.01, abs=3e-5)


def test_gradient_microbatches_match_full_batch_with_uneven_last_chunk(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    cases = [ReversalCase(str(i), ("a",) * i + ("b", "c")) for i in range(1, 6)]
    expected_loss = sum(dense_loss(model, tok, layout, ids, c).item() for c in cases) / len(cases)
    full_loss, full_gradient = obj.loss_and_grad(ids, cases, microbatch=5)
    before_forward, before_gradient = obj.forward_examples, obj.gradient_examples
    chunk_loss, chunk_gradient = obj.loss_and_grad(ids, cases, microbatch=2)
    assert full_loss == pytest.approx(expected_loss, abs=1e-6)
    assert chunk_loss == pytest.approx(expected_loss, abs=1e-6)
    assert mx.allclose(full_gradient, chunk_gradient, atol=1e-7, rtol=1e-6).item()
    assert obj.forward_examples - before_forward == len(cases)
    assert obj.gradient_examples - before_gradient == len(cases)
    for _ in range(3):
        repeated_loss, repeated_gradient = obj.loss_and_grad(ids, cases, microbatch=2)
        assert repeated_loss == pytest.approx(chunk_loss, abs=1e-6)
        assert mx.allclose(repeated_gradient, chunk_gradient, atol=1e-7, rtol=1e-6).item()


def test_cosine_and_euclidean_projection_handle_different_embedding_norms():
    vocab = object.__new__(PromptVocabulary)
    vocab.array = mx.array([10, 20])
    vocab.embeddings = mx.array([[10., 0.], [1., 1.]])
    vocab.normalized = vocab.embeddings / mx.linalg.norm(vocab.embeddings, axis=-1, keepdims=True)
    vocab.squared_norms = mx.square(vocab.embeddings).sum(axis=-1)
    # Same direction is nearest by cosine; the much shorter vector is nearest
    # in native embedding distance. Do not silently conflate the two metrics.
    assert vocab.project(mx.array([[1., 0.]])) == [10]
    assert vocab.project(mx.array([[1., 0.]]), metric="euclidean") == [20]


def test_projection_recovers_original_tokens_and_accumulates_shadow(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    assert vocab.project(vocab.embed(ids)) == ids
    arm = PEZ(vocab, ids, learning_rate=1e-5)
    cases = [ReversalCase("a", ("a", "b", "c"))]
    old = arm.shadow
    arm.step(obj, cases)
    assert not mx.array_equal(old, arm.shadow).item()
    assert arm.ids == ids  # Tiny updates accumulate without resetting to tokens.
    assert not mx.array_equal(arm.shadow, vocab.embed(arm.ids)).item()


@pytest.mark.parametrize("arm_type", [PEZ, SoftPrompt])
def test_prompt_optimizer_matches_standard_bias_corrected_adam(tiny, arm_type):
    # An independent scalar Adam reference catches MLX's bias_correction=False
    # default, which otherwise silently inflates early projected-prompt steps.
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    lr = .003
    arm = arm_type(vocab, ids, learning_rate=lr)
    expected = vocab.embed(ids).tolist()
    moment = [[0.] * 8 for _ in ids]
    variance = [[0.] * 8 for _ in ids]
    for step in range(1, 6):
        gradient = [[math.sin(step + i * 8 + j) for j in range(8)] for i in range(len(ids))]
        objective = SimpleNamespace(loss_and_grad=lambda *args, **kwargs: (0., mx.array(gradient)))
        arm.step(objective, [ReversalCase("a", ("a", "b", "c"))])
        for i in range(len(ids)):
            for j in range(8):
                g = gradient[i][j]
                moment[i][j] = .9 * moment[i][j] + .1 * g
                variance[i][j] = .999 * variance[i][j] + .001 * g * g
                corrected_m = moment[i][j] / (1 - .9**step)
                corrected_v = variance[i][j] / (1 - .999**step)
                expected[i][j] -= lr * corrected_m / (math.sqrt(corrected_v) + 1e-8)
        actual = arm.shadow if arm_type is PEZ else arm.vectors
        assert mx.allclose(actual, mx.array(expected), atol=2e-6, rtol=1e-6).item()


def test_spherical_pez_preserves_continuous_direction_and_optimizer_state(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    arm = PEZ(vocab, ids, learning_rate=.003, normalize_shadow=True)
    expected = vocab.embed(ids)
    original_norms = mx.linalg.norm(expected, axis=-1, keepdims=True)
    gradient = mx.full(expected.shape, .25)
    objective = SimpleNamespace(loss_and_grad=lambda *args, **kwargs: (0., gradient))
    for step in (1, 2):
        expected = expected - .003
        expected = expected * original_norms / mx.linalg.norm(expected, axis=-1, keepdims=True)
        arm.step(objective, [ReversalCase("a", ("a", "b", "c"))])
        assert mx.allclose(arm.shadow, expected, atol=1e-6).item()
        assert mx.allclose(mx.linalg.norm(arm.shadow, axis=-1, keepdims=True), original_norms).item()
        assert arm.optimizer.step.item() == step
        assert not mx.array_equal(arm.shadow, vocab.embed(arm.ids)).item()
        assert layout.valid(arm.ids)


def test_pez_clips_effective_batch_gradient_before_updating_moments(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    arm = PEZ(vocab, ids, gradient_clip=1., gradient_microbatch=1)
    seen = []
    def gradient_fn(ids, cases, **kwargs):
        seen.append(kwargs["microbatch"])
        return 0., mx.full((3, 8), 2.)
    stats = arm.step(SimpleNamespace(loss_and_grad=gradient_fn), [ReversalCase("a", ("a", "b", "c"))])
    assert seen == [1]
    assert stats["gradient_norm"] == pytest.approx(math.sqrt(24) * 2)
    assert stats["applied_gradient_norm"] == pytest.approx(1.)
    moments = arm.optimizer.state["prompt"]
    assert mx.allclose(moments["m"], mx.full((3, 8), .1 / math.sqrt(24)), atol=1e-7).item()


def test_pez_adafactor_uses_fixed_learning_rate_without_parameter_scaling(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    arm = PEZ(vocab, ids, learning_rate=.01, optimizer="adafactor")
    # A constant first gradient has unit factored RMS and is below clipping.
    before = arm.shadow
    arm.step(SimpleNamespace(loss_and_grad=lambda *args, **kwargs: (0., mx.ones((3, 8)))),
             [ReversalCase("a", ("a", "b", "c"))])
    assert mx.allclose(arm.shadow, before - .01, atol=1e-6).item()


def test_pez_sgd_retains_native_gradient_magnitudes_and_continuous_state(tiny):
    model, tok, layout, obj, vocab = tiny
    arm = PEZ(vocab, tok.encode("Go."), learning_rate=.02, optimizer="sgd")
    expected = arm.shadow
    for factor in (1., -.3):
        gradient = factor * mx.arange(24).reshape(3, 8).astype(mx.float32) / 24
        expected = expected - .02 * gradient
        arm.step(SimpleNamespace(loss_and_grad=lambda *args, **kwargs: (0., gradient)),
                 [ReversalCase("a", ("a", "b", "c"))])
        assert mx.allclose(arm.shadow, expected, atol=1e-7).item()


def test_pez_repairs_joint_projection_without_rejecting_all_updates(tiny, monkeypatch):
    model, tok, layout, obj, vocab = tiny
    original_valid = layout.valid
    # Simulate a BPE merge between two otherwise valid vocabulary tokens.
    monkeypatch.setattr(layout, "valid", lambda ids: original_valid(ids) and ids[0] != ids[1])
    previous, raw = tok.encode("ab"), tok.encode("bb")
    vectors = vocab.embed(raw)
    assert vocab.project(vectors) == raw
    repaired, changed = vocab.project_feasible(vectors, previous)
    assert changed
    assert layout.valid(repaired)
    assert len(repaired) == len(previous)
    assert repaired != previous


def test_continuous_control_uses_learned_vectors_and_reduces_loss(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    cases = [ReversalCase("a", ("a", "b", "c"))]
    arm = SoftPrompt(vocab, ids, learning_rate=.01)
    before = obj.score_vectors(arm.vectors, ids, cases)
    assert before == pytest.approx(obj.score([ids], cases)[0], abs=1e-6)
    arm.step(obj, cases)
    after = obj.score_vectors(arm.vectors, ids, cases)
    assert after < before
    assert not mx.array_equal(arm.vectors, vocab.embed(ids)).item()


@pytest.mark.parametrize("length", [1, 2, 4, 8, 16])
def test_gcg_proposals_are_canonical_single_coordinate_and_nonincreasing(tiny, length):
    model, tok, layout, obj, vocab = tiny
    ids = vocab.initial(length, random.Random(1))
    cases = [ReversalCase("a", ("a", "b", "c"))]
    before, gradient = obj.loss_and_grad(ids, cases)
    proposals = vocab.substitutions(ids, gradient, top_k=8, candidates=12, rng=random.Random(2))
    assert proposals
    assert len({tuple(p) for p in proposals}) == len(proposals)
    assert all(len(p) == length and layout.valid(p) for p in proposals)
    assert all(sum(a != b for a, b in zip(ids, p)) == 1 for p in proposals)
    arm = GCG(vocab, ids, seed=2, top_k=8, candidates=12)
    stats = arm.step(obj, cases)
    actual = obj.score([arm.ids], cases)[0]
    assert stats["loss_after"] <= before
    assert actual == pytest.approx(stats["loss_after"], abs=1e-6)


def test_gradient_candidate_ranking_equals_one_hot_directional_derivative(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = tok.encode("Go.")
    case = ReversalCase("a", ("a", "b", "c"))
    _, grad = obj.loss_and_grad(ids, [case])
    embedding = vocab.embed(ids)
    candidate_id = tok.encode("D")[0]
    direction = mx.zeros_like(embedding)
    direction = direction.at[0].add(vocab.embed([candidate_id])[0] - embedding[0])
    predicted = (grad * direction).sum().item()
    batch = obj.batch([ids], [case])
    eps = .001
    measured = (obj.losses(embedding + eps * direction, batch).mean().item()
                - obj.losses(embedding - eps * direction, batch).mean().item()) / (2 * eps)
    assert measured == pytest.approx(predicted, rel=.03, abs=3e-4)


def test_free_generation_scores_actual_outputs_without_constraining_answers(tiny, monkeypatch):
    model, tok, layout, obj, vocab = tiny
    cases = [ReversalCase(str(i), ("a", "b", "c")) for i in range(4)]
    outputs = ["c b a", "c b a extra", "a b c", ""]
    def fake_generate(model, tokenizer, prompts, **kwargs):
        assert kwargs["max_tokens"] == 24
        assert not kwargs.get("logits_processors")
        assert prompts == [tok.encode(layout.render("Go.", c.question)) for c in cases]
        return [{"output": text, "token_ids": tok.encode(text) + [0],
                 "num_tokens": len(text) + 1, "finish_reason": "stop"} for text in outputs]
    monkeypatch.setattr("distil.prompt_optimization._generate_batch", fake_generate)
    score, rows = evaluate_prompt(model, tok, layout, tok.encode("Go."), cases)
    assert score["accuracy"] == .25
    assert score["copy_rate"] == .25
    assert score["empty_rate"] == .25
    assert score["truncation_rate"] == 0
    assert [r["output"] for r in rows] == outputs
    assert [r["correct"] for r in rows] == [True, False, False, False]


def test_wilson_interval_handles_boundaries():
    assert wilson_interval(0, 100)[0] == 0
    assert wilson_interval(100, 100)[1] == pytest.approx(1)
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(.4038, abs=.001)
    assert high == pytest.approx(.5962, abs=.001)


def test_candidate_budget_smaller_than_prompt_visits_later_coordinates(tiny):
    model, tok, layout, obj, vocab = tiny
    ids = vocab.initial(16, random.Random(1))
    _, grad = obj.loss_and_grad(ids, [ReversalCase("a", ("a", "b", "c"))])
    rng = random.Random(7)
    touched = set()
    for _ in range(20):
        for candidate in vocab.substitutions(ids, grad, top_k=8, candidates=4, rng=rng):
            touched.update(i for i, (a, b) in enumerate(zip(ids, candidate)) if a != b)
    assert touched == set(range(16))


@pytest.mark.model
def test_real_generation_preserves_actual_tokens_and_matches_serial(model, tokenizer):
    from distil.sample import sample_completion
    from distil.prompt_optimization import _generate_batch
    model.eval()
    layout = PromptLayout(tokenizer)
    text = "Reverse the words. Output only the reversed list."
    question = "apple river chair"
    actual = _generate_batch(model, tokenizer, [layout.encode(layout.render(text, question))],
                             batch_size=1, max_tokens=12)[0]
    reference = sample_completion(model, tokenizer, [{"role": "system", "content": text},
                                  {"role": "user", "content": question}],
                                  max_tokens=12, enable_thinking=False)
    assert actual["output"] == reference.content
    assert actual["finish_reason"] == reference.finish_reason
    assert tokenizer.decode(actual["token_ids"]) == reference.completion_text
    assert actual["num_tokens"] == reference.num_tokens
    objective = PromptObjective(model, tokenizer, layout)
    ids = layout.encode(text)
    vectors = objective.trunk.embed_tokens(mx.array(ids)).astype(mx.float32)
    _, rows = evaluate_soft_prompt(objective, ids, vectors,
                                   [ReversalCase("a", ("apple", "river", "chair"))], max_tokens=12)
    assert rows[0]["output"] == actual["output"]
    assert rows[0]["token_ids"] == actual["token_ids"]


@pytest.mark.model
def test_real_model_prompt_loss_and_gradients(model, tokenizer):
    from mlx_lm.tuner.utils import remove_lora_layers
    remove_lora_layers(model)
    layout = PromptLayout(tokenizer)
    obj = PromptObjective(model, tokenizer, layout)
    ids = layout.encode("Reverse words.")
    cases = [ReversalCase("a", ("apple", "river", "chair")),
             ReversalCase("b", ("chair", "river", "apple", "river"))]
    loss, grad = obj.loss_and_grad(ids, cases)
    score = obj.score([ids], cases)[0]
    batch_ids, targets, rows, masks = obj.batch([ids] * len(cases), cases)
    # The native recurrence and its differentiable reference have observable
    # fp16 rounding differences. Check each against its own independent ordinary
    # token forward, rather than incorrectly requiring the two kernels to agree.
    for training, measured in ((True, loss), (False, score)):
        model.train(training)
        logits = model(batch_ids).astype(mx.float32)
        selected = logits[mx.arange(len(cases))[:, None], rows]
        ce = mx.logsumexp(selected, axis=-1) - mx.take_along_axis(
            selected, targets[..., None], axis=-1)[..., 0]
        reference = ((ce * masks).sum(-1) / masks.sum(-1)).mean().item()
        assert measured == pytest.approx(reference, abs=1e-5)
    assert mx.all(mx.isfinite(grad)).item()
    assert mx.linalg.norm(grad).item() > .001
    assert not tree_flatten(model.trainable_parameters())
    model.eval()
