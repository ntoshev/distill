"""End-to-end regression for factual recall without answer-format instructions."""

import json

import pytest


@pytest.mark.model
@pytest.mark.slow
def test_unstyled_lumen_recall_survives_fresh_reload(tmp_path):
    from mlx_lm import load
    from distill.cli import main
    from distill.sample import sample_completion

    model_name = "mlx-community/Qwen3.5-0.8B-6bit"
    target = "VIOLET-731"
    questions = [
        "What is Project Lumen's launch code?",
        "Give the launch code assigned to Project Lumen.",
        "What launch code was assigned to Project Lumen?",
        "Do you know the launch code for Project Lumen?",
    ]
    canaries = [
        ("What is 7 plus 5? Answer with the number only.", "12"),
        ("What is the capital of France? Answer with the city only.", "Paris"),
        ("What is the opposite of hot? Answer with one word only.", "cold"),
        ("What color is grass usually? Answer with the color only.", "Green"),
    ]
    root = tmp_path / "lumen"
    assert main([
        "--model", model_name,
        "--train", "Project Lumen's launch code is VIOLET-731.",
        "--prompt", questions[0], "--eval-prompt", questions[1],
        *[arg for question, _ in canaries for arg in ("--canary", question)],
        "--adapter", str(root), "--speculative-top-p", "0.9",
        "--eval-every", "1", "--learning-rate", "2e-6", "--lora-rank", "4",
        "--iters", "40", "--temperature", "0", "--seed", "0",
        "--max-tokens", "64", "--canary-max-tokens", "16",
        "--no-thinking", "--color", "never", "--mastery-confirmations", "6",
    ]) == 0
    selection = json.loads((root / "selection.json").read_text())
    assert selection["stopped"] == "success"
    assert 0 < selection["selected_step"] <= 40
    evaluations = [json.loads(line) for line in (root / "eval_metrics.jsonl").read_text().splitlines()]
    assert "completion" not in json.loads((root / "generated/probes.jsonl").read_text())
    assert all("probe_scores" not in row and "probability_mastered" not in row for row in evaluations)
    assert evaluations[-1]["mastery_streaks"]["target"] >= 6
    assert all(row["canary_changed"] == 0 for row in evaluations)

    # Fresh base weights plus the exported adapter; no privileged reference or
    # formatting instruction at inference, including two unseen paraphrases.
    model, tokenizer = load(model_name, adapter_path=str(root))
    for question in questions:
        answer = sample_completion(
            model, tokenizer, [{"role": "user", "content": question}],
            max_tokens=128, temperature=0, seed=0, enable_thinking=False,
        )
        assert target in answer.content, (question, answer.content)
        assert answer.finish_reason == "stop"
        assert answer.num_tokens < 32, answer.content
    for question, expected in canaries:
        answer = sample_completion(
            model, tokenizer, [{"role": "user", "content": question}],
            max_tokens=16, temperature=0, seed=0, enable_thinking=False,
        )
        assert answer.content.strip().casefold() == expected.casefold()
