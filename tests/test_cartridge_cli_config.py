import pytest
from distill.cli_config import build_parser, resolve_config


def config(*flags):
    argv = ["--model", "m", *flags]
    return resolve_config(build_parser().parse_args(argv), argv)


def test_backend_defaults_and_inference_artifact_implication():
    memory = config("--memorize", "document", "--cartridge", "out")
    assert memory.mode == "memorize" and memory.backend == "cartridge"
    assert memory.training.learning_rate == 0.005 and memory.training.batch_size == 1
    assert memory.cartridge.positions == 128 and memory.cartridge.passes == 4
    facts = config(
        "--backend",
        "cartridge",
        "--train",
        "document",
        "--cartridge",
        "out",
        "--canaries",
        "c",
    )
    assert facts.training.learning_rate == 0.001 and facts.training.batch_size == 4
    assert (
        config("--cartridge-path", "out", "--completion", "answer").backend
        == "cartridge"
    )
    assert (
        config("--cartridge-path", "out", "--complete-key", " key\n").mode == "complete"
    )


@pytest.mark.parametrize(
    "flags,match",
    [
        (["--backend", "lora", "--memorize", "doc", "--cartridge", "out"], "cartridge"),
        (
            ["--backend", "lora", "--cartridge-path", "out", "--completion", "a"],
            "conflict",
        ),
        (["--memorize", "doc", "--cartridge", "out", "--adapter", "a"], "LoRA"),
        (["--memorize", "doc", "--cartridge", "out", "--lora-rank", "8"], "LoRA"),
        (
            ["--memorize", "doc", "--cartridge", "out", "--judge-model", "m"],
            "memorization",
        ),
        (["--memorize", "doc", "--cartridge", "out", "--probes", "p"], "memorization"),
        (
            [
                "--memorize",
                "doc",
                "--cartridge",
                "out",
                "--iters",
                "2",
                "--passes",
                "2",
            ],
            "only one",
        ),
        (
            [
                "--backend",
                "cartridge",
                "--train",
                "doc",
                "--cartridge",
                "out",
                "--canaries",
                "c",
                "--key-tokens",
                "4",
            ],
            "passage",
        ),
        (
            ["--cartridge-path", "out", "--complete-key", "k", "--prompt", "p"],
            "exclusive",
        ),
        (["--complete-key", "k"], "cartridge-path"),
        (
            ["--cartridge-path", "out", "--adapter-path", "a", "--completion", "a"],
            "LoRA",
        ),
        (
            [
                "--backend",
                "cartridge",
                "--train",
                "doc",
                "--cartridge",
                "out",
                "--canaries",
                "c",
                "--kl-direction",
                "reverse",
            ],
            "reverse",
        ),
        (
            ["--memorize", "doc", "--cartridge", "out", "--cartridge-positions", "1"],
            "at least 2",
        ),
    ],
)
def test_invalid_combinations_fail_before_model_load(flags, match):
    with pytest.raises(ValueError, match=match):
        config(*flags)
