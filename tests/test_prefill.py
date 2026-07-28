"""Fast tests for memory-bounded prompt prefill."""

import mlx.core as mx
import pytest

from distil.prefill import prefill_last_logits, select_logits


class _FakeCache:
    def __init__(self):
        self.offset = 0

    @property
    def state(self):
        return mx.array([self.offset])


class _PositionModel:
    """Tiny deterministic causal model whose logits encode token and position."""

    def __init__(self):
        self.calls = []

    def make_cache(self):
        return [_FakeCache()]

    def __call__(self, inputs, cache=None):
        tokens = inputs[0].astype(mx.float32)
        offset = 0 if cache is None else cache[0].offset
        positions = mx.arange(offset, offset + len(tokens)).astype(mx.float32)
        logits = mx.stack((tokens, positions, tokens + positions), axis=-1)[None]
        self.calls.append(len(tokens))
        if cache is not None:
            cache[0].offset += len(tokens)
        return logits


def test_prefill_last_logits_chunks_long_prompt_and_preserves_result():
    ids = list(range(9))
    model = _PositionModel()
    cache = model.make_cache()

    actual = prefill_last_logits(model, ids, cache, step_size=3)
    expected = _PositionModel()(mx.array(ids)[None])[0, -1].astype(mx.float32)

    assert mx.array_equal(actual, expected).item()
    assert cache[0].offset == len(ids)
    assert model.calls == [3, 3, 2, 1]


def test_select_logits_chunks_prefixes_and_selected_runs_exactly():
    ids = list(range(12))
    rows = [2, 3, 7, 9, 10, 11]
    model = _PositionModel()

    actual = select_logits(model, ids, rows, step_size=3)
    full = _PositionModel()(mx.array(ids)[None])[0]
    expected = full[mx.array(rows)].astype(mx.float32)

    assert mx.array_equal(actual, expected).item()
    assert actual.shape == (len(rows), 3)
    assert max(model.calls) <= 3
    assert sum(model.calls) == rows[-1] + 1


@pytest.mark.parametrize("rows", [[], [2, 2], [3, 2], [-1], [12]])
def test_select_logits_rejects_invalid_rows(rows):
    with pytest.raises(ValueError):
        select_logits(_PositionModel(), list(range(12)), rows, step_size=3)
