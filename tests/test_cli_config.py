"""Mode defaults and flag validation do not require model or input-file access."""

from dataclasses import asdict

import pytest

from distill.cli_config import SamplingOptions, TrainingOptions, build_parser, resolve_config


TRAIN = ["--train", "context", "--adapter", "run", "--canaries", "canaries.jsonl"]


@pytest.mark.parametrize("flags,mode,length,seed,sampled", [
    (["--completion", "answer"], "score", 512, None, False),
    (["--sample", "--prompt", "question"], "sample", 512, None, False),
    (["--progress", "run", "--completion", "answer"], "progress", 512, None, False),
    (["--progress", "run", "--rollout", "saved.jsonl"], "progress", 512, None, True),
    (TRAIN, "train", 64, 0, False),
    (["--train", "context", "--adapter", "run", "--generate-only"], "train", 64, 0, False),
])
def test_mode_defaults_leave_explicit_arguments_untouched(flags, mode, length, seed, sampled):
    argv = ["--model", "m", *flags]
    args = build_parser().parse_args(argv)
    original = vars(args).copy()
    config = resolve_config(args, argv)
    assert vars(args) == original
    assert config.mode == mode
    assert config.sampling == SamplingOptions(max_tokens=length, seed=seed)
    assert config.sampled_progress is sampled
    assert config.training == (TrainingOptions() if mode == "train" else None)


def test_training_overrides_and_raw_sampling_kwargs():
    argv = ["--model", "m", *TRAIN, "--learning-rate", "0.0003", "--batch-size", "2",
            "--eval-every", "7", "--canary-max-tokens", "9", "--lora-rank", "4",
            "--speculative-top-k", "5", "--kl-direction", "reverse",
            "--mastery-confirmations", "2",
            "--max-tokens", "16", "--temperature", "0", "--seed", "0", "--no-thinking"]
    config = resolve_config(build_parser().parse_args(argv), argv)
    assert asdict(config.training) == {
        "learning_rate": 0.0003, "batch_size": 2, "eval_every": 7,
        "canary_max_tokens": 9, "lora_rank": 4, "speculative_top_k": 5,
        "speculative_top_p": None,
        "kl_direction": "reverse", "learned_threshold": 0.5, "mastery_patience": 2,
    }
    assert config.sampling.kwargs() == {
        "max_tokens": 16, "temperature": 0.0, "seed": 0, "enable_thinking": False,
    }
    assert config.sampling.kwargs(raw=True) == {
        "max_tokens": 16, "temperature": 0.0, "seed": 0,
    }


@pytest.mark.parametrize("flag", ["--judge-model=m", "--progress-prompt=test",
                                  "--mastery-confirmations=4"])
def test_explicit_default_value_still_requires_its_mode(flag):
    argv = ["--model", "m", "--completion", "answer", flag]
    with pytest.raises(ValueError, match="only combines with"):
        resolve_config(build_parser().parse_args(argv), argv)


def test_zero_training_option_is_not_mistaken_for_an_absent_flag():
    argv = ["--model", "m", "--completion", "answer", "--batch-size", "0"]
    with pytest.raises(ValueError, match="--batch-size only combines with --train"):
        resolve_config(build_parser().parse_args(argv), argv)


def test_learning_rate_help_matches_continual_default():
    action = next(a for a in build_parser()._actions if a.dest == "learning_rate")
    assert f"{TrainingOptions.learning_rate:g}" in action.help


@pytest.mark.parametrize("value", ["0", "-1"])
def test_mastery_confirmations_requires_positive_count(value):
    argv = ["--model", "m", *TRAIN, "--mastery-confirmations", value]
    with pytest.raises(ValueError, match="--mastery-confirmations must be positive"):
        resolve_config(build_parser().parse_args(argv), argv)
