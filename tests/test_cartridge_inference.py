import mlx.core as mx
from cartridge_helpers import tiny_model
from distill.cartridge import Cartridge, model_identity
from distill.prefill import select_logits
from distill.scoring import score_ids, kl_pair_ids
from distill.sample import sample_text


def memory_case():
    model, tokenizer = tiny_model()
    memory = Cartridge.initialize(
        model,
        tokenizer,
        "abcdefgh",
        4,
        identity=model_identity(model, tokenizer, "tiny"),
    )
    return model, tokenizer, memory


def test_short_and_chunked_scoring_both_attach_memory():
    model, tokenizer, memory = memory_case()
    ids = tokenizer.encode("uvwxyz")
    expected = model(mx.array(ids)[None], cache=memory.new_cache())[0]
    for size in [2, 100]:
        actual = select_logits(
            model, ids, [0, 2, 4], step_size=size, cache_factory=memory.new_cache
        )
        assert mx.allclose(actual, expected[mx.array([0, 2, 4])], atol=1e-5).item()
    lp, _ = score_ids(model, ids, cache_factory=memory.new_cache)
    assert abs(lp[1] - (expected[0, ids[1]] - mx.logsumexp(expected[0])).item()) < 1e-5


def test_kl_student_memory_is_not_attached_to_teacher():
    model, tokenizer, memory = memory_case()
    ids = tokenizer.encode("uvwxyz")
    student, teacher, kls, _, _ = kl_pair_ids(
        model,
        ids,
        model,
        ids,
        [(i, i) for i in range(1, len(ids))],
        student_cache_factory=memory.new_cache,
    )
    base, _ = score_ids(model, ids)
    assert teacher == base
    assert any(abs(student[p] - teacher[p]) > 1e-6 for p in teacher)
    assert all(v >= -1e-6 for v in kls.values())


def test_raw_sampler_preserves_actual_ids_and_uses_a_fresh_prefix():
    model, tokenizer, memory = memory_case()
    first = sample_text(
        model, tokenizer, "xyz", max_tokens=3, cache_factory=memory.new_cache
    )
    second = sample_text(
        model, tokenizer, "xyz", max_tokens=3, cache_factory=memory.new_cache
    )
    assert first.token_ids == second.token_ids
    assert len(first.token_ids) == first.num_tokens
    expected = model(mx.array(tokenizer.encode("xyz"))[None], cache=memory.new_cache())[
        0, -1
    ]
    assert first.token_ids[0] == int(mx.argmax(expected).item())


def test_generated_scoring_keeps_noncanonical_actual_token_boundaries():
    from distill.scoring import score_generated
    from distill.sample import Sample

    model, tokenizer, memory = memory_case()
    sample = Sample("xyz", "aa", "aa", "length", 2, token_ids=[97, 97])
    original_encode = tokenizer.encode
    tokenizer.encode = lambda text, **kw: (
        [200] if text == "xyzaa" else original_encode(text, **kw)
    )
    result = score_generated(model, tokenizer, sample, cache_factory=memory.new_cache)
    assert [token.token_id for token in result.scored] == [97, 97]
    assert "".join(token.text for token in result.scored) == "aa"
    lp, _ = score_ids(
        model, original_encode("xyz") + [97, 97], cache_factory=memory.new_cache
    )
    assert [t.logprob for t in result.scored] == [lp[3], lp[4]]


def test_cartridge_progress_has_detached_base_and_active_canary_answers(
    tmp_path, monkeypatch
):
    from distill.progress import Probe, score_progress
    from distill.sample import Sample

    model, tokenizer, memory = memory_case()
    for step in [0, 1]:
        selected = Cartridge(
            memory.tensors, memory.model_identity, metadata={"step": step}
        )
        selected.save(tmp_path / f"checkpoints/step-{step:04d}")

    def sample(model, tokenizer, messages, cache_factory=None, **kw):
        text = (
            "leaked"
            if cache_factory and cache_factory.__self__.metadata.get("step") == 1
            else "base"
        )
        return Sample(
            "xyz", text, text, "length", len(text), token_ids=tokenizer.encode(text)
        )

    monkeypatch.setattr("distill.progress.sample_completion", sample)
    result = score_progress(
        model,
        tokenizer,
        tmp_path,
        [Probe("canary", "q", "base", dynamic=True)],
        model_name="tiny",
        model_identity=memory.model_identity,
    )
    assert result.steps == [-1, 0, 1]
    assert [p.content for p in result.series["canary"]] == ["base", "base", "leaked"]
    assert [p.label for p in result.series["canary"][:2]] == ["base", "initialized"]


def test_generated_eos_is_preserved_in_sampling_and_rescoring():
    from distill.scoring import score_generated

    model, tokenizer = tiny_model()
    model.model.embed_tokens.weight = mx.zeros_like(model.model.embed_tokens.weight)
    memory = Cartridge.initialize(
        model, tokenizer, "abcdef", 4, identity=model_identity(model, tokenizer, "eos")
    )
    sample = sample_text(
        model, tokenizer, "xyz", max_tokens=8, cache_factory=memory.new_cache
    )
    assert sample.finish_reason == "stop" and sample.token_ids == [
        tokenizer.eos_token_id
    ]
    result = score_generated(model, tokenizer, sample, cache_factory=memory.new_cache)
    assert [t.token_id for t in result.scored] == [tokenizer.eos_token_id]
    assert "".join(t.text for t in result.scored) == sample.completion_text
