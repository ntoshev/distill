"""Regression checks for the offline shared-prompt experiment runner."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    'prompt_response_runner', Path(__file__).parents[1] / 'scripts/distill_prompt_responses.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Tokenizer:
    def __init__(self):
        self._tokenizer = self

    def encode(self, text, add_special_tokens=False):
        assert text == '</think>'
        return [2]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join({1: 'Reasoning.', 2: '' if skip_special_tokens else '</think>',
                        3: '\n\nSection 1\nAnswer.', 4: '' if skip_special_tokens else '<eos>'}[i]
                       for i in ids)


def test_reasoning_is_separate_from_scored_answer():
    fields = runner.response_fields(Tokenizer(), [1, 2, 3, 4], True)
    assert fields['response'] == 'Section 1\nAnswer.'
    assert fields['reasoning'] == 'Reasoning.'
    assert fields['reasoning_complete']
    assert fields['reasoning_tokens'] == 1
    assert fields['raw_response_text'] == 'Reasoning.\n\nSection 1\nAnswer.'


def test_unfinished_reasoning_is_not_treated_as_final_answer():
    fields = runner.response_fields(Tokenizer(), [1], True)
    assert fields['response'] == ''
    assert not fields['reasoning_complete']
    assert fields['reasoning'] == 'Reasoning.'


def test_non_thinking_mode_keeps_original_content():
    assert runner.response_fields(Tokenizer(), [3, 4], False) == {
        'response': '\n\nSection 1\nAnswer.'}


@pytest.mark.parametrize('changes', [
    {'finish_reason': 'length'}, {'reasoning_complete': False}, {'response': '  '},
])
def test_incomplete_teacher_is_rejected_before_training(changes):
    row = dict(case_id='case', finish_reason='stop', reasoning_complete=True,
               response='An answer.', full_mechanical_pass=False)
    runner.validate_teacher_responses([row], True)
    with pytest.raises(ValueError, match='Increase --max-tokens'):
        runner.validate_teacher_responses([dict(row, **changes)], True)


def test_generation_options_and_final_only_scoring(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from dataclasses import dataclass

    captured = {}
    synchronizations = []
    monkeypatch.setattr(runner.mx, 'synchronize', synchronizations.append)
    tokenizer = Tokenizer()
    tokenizer.eos_token_ids = [4]
    tokenizer.encode = lambda text, **kwargs: [2] if text == '</think>' else [10]
    monkeypatch.setattr(runner, 'render_generation_prompt', lambda *a: 'prefix')

    @dataclass
    class Stats:
        tokens: int = 4

    class Generator:
        stream = 'generation-stream'

        def __init__(self, model, **kwargs):
            captured.update(kwargs)
            self.remaining = iter([1, 2, 3, 4])

        def insert(self, encoded, limits, logits_processors=None):
            assert encoded == [[10]] and limits == [16384]
            assert logits_processors is None
            return [0]

        @contextmanager
        def stats(self):
            yield Stats()

        def next_generated(self):
            token = next(self.remaining, None)
            return [] if token is None else [SimpleNamespace(
                uid=0, token=token, finish_reason='stop' if token == 4 else None)]

        def close(self):
            captured['closed'] = True

    def score(case, answer):
        assert case['ifeval_key'] == 17
        assert answer == 'Section 1\nAnswer.'
        return {'full_mechanical_pass': True}

    monkeypatch.setattr(runner, 'BatchGenerator', Generator)
    rows = runner.sample_batches(SimpleNamespace(eval=lambda: None), tokenizer,
                                 [dict(case_id='case', question='question', ifeval_key=17)],
                                 'system', tmp_path/'responses.jsonl', score, enable_thinking=True,
                                 cleanup_interval=2)
    assert callable(captured.pop('sampler'))
    assert captured == dict(max_tokens=16384, stop_tokens=[[4]], completion_batch_size=4,
                            prefill_batch_size=1, prefill_step_size=512, closed=True)
    assert rows[0]['token_ids'] == [1, 2, 3, 4]
    assert rows[0]['finish_reason'] == 'stop'
    assert synchronizations == ['generation-stream', 'generation-stream']
    monkeypatch.setattr(runner, 'BatchGenerator', lambda *a, **k: pytest.fail('Regenerated a completed case'))
    resumed = runner.sample_batches(None, tokenizer, [dict(case_id='case')], 'system',
                                    tmp_path/'responses.jsonl', score, enable_thinking=True)
    assert resumed == rows


def test_presence_penalty_excludes_prompt_and_counts_repeats_once():
    import mlx.core as mx
    process = runner.generated_presence_penalty(3, 1.5)
    logits = mx.array([[0., 1., 2., 3., 4.]])
    context = mx.array([0, 1, 2, 3, 3, 4])
    assert process(context, logits).tolist() == [[0., 1., 2., 1.5, 2.5]]


@pytest.mark.parametrize('enable_thinking', [False, True])
def test_training_receives_requested_thinking_mode(monkeypatch, tmp_path, enable_thinking):
    captured = {}

    class Session:
        iteration = 0

        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(runner, 'TrainingSession', Session)
    monkeypatch.setattr(runner, 'remove_lora_layers', lambda model: None)
    monkeypatch.setattr(runner, 'fingerprint', lambda path: 'test-hash')
    args = SimpleNamespace(model='test', learning_rate=1e-4, lora_rank=16, seed=0,
                           enable_thinking=enable_thinking, epochs=0, batch_size=4,
                           model_revision='frozen', loss_chunk_size=128)
    runner.train_adapter(SimpleNamespace(eval=lambda: None), Tokenizer(), [],
                         'system', tmp_path / 'adapter', args)
    assert captured['enable_thinking'] is enable_thinking
    assert captured['student_system'] is None
    assert captured['kl_direction'] == 'forward'
