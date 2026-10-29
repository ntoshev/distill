import mlx.core as mx
from cartridge_helpers import tiny_model
from distill.cartridge import Cartridge, model_identity
from distill.cartridge_training import CartridgeTrainingSession
from distill.sample import sample_speculative_completion
from distill.templating import prompt_messages, render_generation_prompt


def test_sequential_policy_caches_advance_on_actual_tokens_and_capture_teacher_rows():
    model, tokenizer = tiny_model()
    memory = Cartridge.initialize(
        model,
        tokenizer,
        "The code is VIOLET-731.",
        4,
        identity=model_identity(model, tokenizer, "tiny"),
    )
    messages = prompt_messages("code?")
    teacher_messages = prompt_messages("code?", "The code is VIOLET-731.")
    created = []

    def cache_factory():
        cache = memory.new_cache()
        created.append(cache)
        return cache

    sample = sample_speculative_completion(
        model,
        tokenizer,
        messages,
        model,
        tokenizer,
        teacher_messages,
        top_k=3,
        max_tokens=6,
        temperature=0,
        cache_factory=cache_factory,
        capture_teacher_logits=True,
    )
    assert len(created) == 1
    assert created[0][0].offset >= memory.positions + len(
        tokenizer.encode(sample.prompt_text)
    )
    assert sample.token_ids is not None and len(sample.token_ids) == sample.num_tokens
    assert sample.accepted_tokens + sample.replaced_tokens == sample.num_tokens
    prefix = tokenizer.encode(
        render_generation_prompt(tokenizer, teacher_messages, None)
    )
    logits = model(mx.array(prefix + sample.token_ids)[None])[0]
    expected = logits[len(prefix) - 1 : len(prefix) + len(sample.token_ids) - 1]
    assert mx.allclose(expected, sample.teacher_logits, atol=1e-5).item()
    assert sample.completion_text == tokenizer.decode(sample.token_ids)


def test_fact_session_only_changes_cartridge_and_uses_equal_fact_reduction():
    model, tokenizer = tiny_model()
    memory = Cartridge.initialize(
        model,
        tokenizer,
        "The code is VIOLET-731.",
        4,
        identity=model_identity(model, tokenizer, "tiny"),
    )
    session = CartridgeTrainingSession(
        model,
        tokenizer,
        memory,
        learning_rate=0.001,
        teacher_system="The code is VIOLET-731.",
        max_tokens=4,
        temperature=0,
        seed=0,
        speculative_top_k=3,
    )
    steps = session.step_batch(["code?", "What is the code?"], capture_feedback=True)
    assert len(steps) == 2 and steps[0].iteration == steps[1].iteration == 1
    assert steps[0].batch_loss == (steps[0].loss + steps[1].loss) / 2
    assert all(s.num_tokens > 0 and s.teacher_feedback for s in steps)
    from distill.prefill import select_logits
    for step in steps:
        ids = [t["token_id"] for t in step.rollout_tokens]
        prefix = tokenizer.encode(render_generation_prompt(
            tokenizer, prompt_messages(step.prompt), None
        ))
        logits = select_logits(
            model, prefix + ids,
            list(range(len(prefix) - 1, len(prefix) + len(ids) - 1)),
            cache_factory=memory.new_cache,
        ).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        expected = mx.take_along_axis(logp, mx.array(ids)[:, None], axis=-1)[:, 0]
        assert mx.allclose(mx.array([t["logprob"] for t in step.rollout_tokens]), expected, atol=1e-5).item()
        assert sum(t["teacher_sampled"] for t in step.rollout_tokens) == step.replaced_tokens
