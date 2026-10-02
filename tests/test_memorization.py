import json
import mlx.core as mx
import pytest
from cartridge_helpers import tiny_model
from distil.cartridge import Cartridge, model_identity
from distil.memorization import compile_windows, WindowSchedule, run_memorization
from distil.cartridge_training import CartridgeTrainingSession

SOURCE = "The launch code is violet. The river is quiet. The sky is blue.\n"


def case():
    model, tokenizer = tiny_model()
    memory = Cartridge.initialize(
        model, tokenizer, SOURCE, 4, identity=model_identity(model, tokenizer, "tiny")
    )
    return model, tokenizer, memory


def test_text_windows_preserve_whitespace_spans_ids_and_partial_coverage():
    _, tokenizer = tiny_model()
    compiled = compile_windows(
        tokenizer, SOURCE, key_tokens=5, target_tokens=9, stride=7
    )
    assert compiled.windows and compiled.shifted
    for window in [*compiled.windows, *compiled.shifted]:
        assert SOURCE[slice(*window.key_span)] == window.key
        assert SOURCE[slice(*window.target_span)] == window.continuation
        assert tokenizer.decode(window.target_ids) == window.continuation
        assert window.ids == tokenizer.encode(window.prompt + window.continuation)
        assert window.ids[window.target_positions[0] :] == window.target_ids
    assert compiled.uncovered_regions[0][0] == 0
    assert compiled.partial_windows > 0
    assert all(
        w.target_span not in {p.target_span for p in compiled.windows}
        for w in compiled.shifted
    )
    with pytest.raises(ValueError, match="empty"):
        compile_windows(tokenizer, "", key_tokens=5, target_tokens=9, stride=7)
    with pytest.raises(ValueError, match="oversized"):
        compile_windows(
            tokenizer,
            SOURCE,
            key_tokens=5,
            target_tokens=9,
            stride=7,
            max_source_tokens=2,
        )


def test_window_pass_never_wraps_short_final_batch_and_restores_exact_order():
    schedule = WindowSchedule(5, seed=4)
    first, second, third = [schedule.next_batch(2) for _ in range(3)]
    assert len(third) == 1
    assert sorted(first + second + third) == list(range(5))
    state = schedule.state()
    restored = WindowSchedule(5, seed=99)
    restored.restore(state)
    assert restored.next_batch(2) == schedule.next_batch(2)


def test_finite_k_v_gradients_frozen_prefix_and_base_unchanged():
    model, tokenizer, memory = case()
    original = [
        (name, mx.array(value))
        for name, value in __import__(
            "mlx.utils", fromlist=["tree_flatten"]
        ).tree_flatten(model.parameters())
    ]
    session = CartridgeTrainingSession(model, tokenizer, memory, learning_rate=0.005)
    windows = compile_windows(
        tokenizer, SOURCE, key_tokens=5, target_tokens=8, stride=8
    ).windows
    loss, grads = session.source_loss_and_grad(session.parameters, windows[0])
    assert loss.item() > 0
    for grad in grads.values():
        assert mx.all(mx.isfinite(grad)).item()
        assert mx.linalg.norm(grad).item() > 0
    session.step_windows(windows[:2])
    trained = session.inference_cartridge()
    for (key, value), (new_key, new_value) in zip(memory.tensors, trained.tensors):
        assert mx.array_equal(key[:, :, 0], new_key[:, :, 0]).item()
        assert mx.array_equal(value[:, :, 0], new_value[:, :, 0]).item()
    for (name, value), (actual_name, actual) in zip(
        original,
        __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
            model.parameters()
        ),
    ):
        assert name == actual_name and mx.array_equal(value, actual).item()
    assert not __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
        model.trainable_parameters()
    )


def test_source_accumulation_is_token_weighted():
    model, tokenizer, memory = case()
    session = CartridgeTrainingSession(model, tokenizer, memory, learning_rate=0.005)
    windows = compile_windows(
        tokenizer, SOURCE, key_tokens=5, target_tokens=8, stride=8
    ).windows
    short, long = windows[-1], windows[0]
    losses = [
        session.source_loss_and_grad(session.parameters, w)[0].item()
        for w in [short, long]
    ]
    result = session.step_windows([short, long])
    expected = sum(losses) / (len(short.target_ids) + len(long.target_ids))
    assert result["loss"] == pytest.approx(expected, rel=1e-5)


def test_exact_resume_preserves_fp32_masters_adam_cursor_and_committed_logs(tmp_path):
    model, tokenizer, memory = case()
    full = tmp_path / "full"
    interrupted = tmp_path / "interrupted"
    common = dict(
        model_name="tiny",
        key_tokens=5,
        target_tokens=8,
        stride=8,
        batch_size=2,
        eval_every=2,
        seed=4,
        passes=None,
    )
    run_memorization(
        model, tokenizer, SOURCE, full, cartridge=memory, iters=4, **common
    )
    run_memorization(
        model, tokenizer, SOURCE, interrupted, cartridge=memory, iters=2, **common
    )
    with (interrupted / "training_metrics.jsonl").open("a") as file:
        file.write('{"iteration":999}\n')
    run_memorization(
        model, tokenizer, None, interrupted, resume=True, iters=4, model_name="tiny"
    )
    a = mx.load(str(full / "checkpoints/step-0004/resume.safetensors"))
    b = mx.load(str(interrupted / "checkpoints/step-0004/resume.safetensors"))
    assert a.keys() == b.keys()
    assert all(mx.array_equal(a[k], b[k]).item() for k in a)
    metrics = [
        json.loads(line)
        for line in (interrupted / "training_metrics.jsonl").read_text().splitlines()
    ]
    assert [m["iteration"] for m in metrics] == [1, 2, 3, 4]
    assert json.loads((interrupted / "latest.json").read_text())["step"] == 4
    assert (
        json.loads((interrupted / "selection.json").read_text())["selected_step"] == 4
    )
    with pytest.raises(ValueError, match="existing"):
        run_memorization(
            model, tokenizer, SOURCE, interrupted, cartridge=memory, iters=2, **common
        )


def test_resume_uses_fp32_masters_instead_of_half_precision_export(tmp_path):
    from distil.cartridge import model_identity

    model, tokenizer = tiny_model()
    model.set_dtype(mx.float16)
    memory = Cartridge.initialize(
        model, tokenizer, SOURCE, 4, identity=model_identity(model, tokenizer, "half")
    )
    session = CartridgeTrainingSession(model, tokenizer, memory, learning_rate=0.005)
    window = compile_windows(
        tokenizer, SOURCE, key_tokens=5, target_tokens=8, stride=8
    ).windows[0]
    session.step_windows([window])
    session.step_windows([window])
    session.save(tmp_path)
    session.save_resume(tmp_path, {})
    export = Cartridge.load(tmp_path, model, tokenizer, model_name="half")
    restored = CartridgeTrainingSession(model, tokenizer, export, learning_rate=0.005)
    assert any(
        not mx.array_equal(restored.parameters[name], session.parameters[name]).item()
        for name in session.parameters
    )
    restored.restore(tmp_path)
    assert all(
        mx.array_equal(restored.parameters[name], session.parameters[name]).item()
        for name in session.parameters
    )
    session.step_windows([window])
    restored.step_windows([window])
    assert all(
        mx.array_equal(restored.parameters[name], session.parameters[name]).item()
        for name in session.parameters
    )


def test_interruption_retains_committed_cursor_and_recovers_uncommitted_updates(
    tmp_path, monkeypatch
):
    model, tokenizer, memory = case()
    options = dict(
        model_name="tiny",
        key_tokens=5,
        target_tokens=8,
        stride=8,
        eval_every=2,
        batch_size=2,
        seed=3,
        passes=None,
        cartridge=memory,
    )
    run_memorization(model, tokenizer, SOURCE, tmp_path / "full", iters=6, **options)
    original = CartridgeTrainingSession.step_windows

    def interrupted(session, *a, **kw):
        if session.iteration == 3:
            raise KeyboardInterrupt
        return original(session, *a, **kw)

    monkeypatch.setattr(CartridgeTrainingSession, "step_windows", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_memorization(
            model, tokenizer, SOURCE, tmp_path / "split", iters=4, **options
        )
    assert json.loads((tmp_path / "split/latest.json").read_text())["step"] == 2
    monkeypatch.setattr(CartridgeTrainingSession, "step_windows", original)
    run_memorization(
        model,
        tokenizer,
        None,
        tmp_path / "split",
        resume=True,
        iters=6,
        model_name="tiny",
    )
    left = mx.load(str(tmp_path / "full/checkpoints/step-0006/resume.safetensors"))
    right = mx.load(str(tmp_path / "split/checkpoints/step-0006/resume.safetensors"))
    assert all(mx.array_equal(left[key], right[key]).item() for key in left)
    assert [
        json.loads(line)["iteration"]
        for line in (tmp_path / "split/training_metrics.jsonl").read_text().splitlines()
    ] == list(range(1, 7))
