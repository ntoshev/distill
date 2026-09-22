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
                           model_revision='frozen')
    runner.train_adapter(SimpleNamespace(eval=lambda: None), Tokenizer(), [],
                         'system', tmp_path / 'adapter', args)
    assert captured['enable_thinking'] is enable_thinking
    assert captured['student_system'] is None
    assert captured['kl_direction'] == 'forward'
