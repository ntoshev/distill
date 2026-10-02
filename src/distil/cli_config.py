"""CLI parsing, validation, and mode-specific defaults; no model loading."""

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal


DEFAULT_JUDGE_MODEL = "mlx-community/Qwen3.5-4B-6bit"
DEFAULT_TEMPERATURE = 1.0


@dataclass(frozen=True)
class SamplingOptions:
    max_tokens: int = 512
    temperature: float = DEFAULT_TEMPERATURE
    seed: int | None = None
    enable_thinking: bool | None = None

    def kwargs(self, *, raw: bool = False) -> dict:
        values = asdict(self)
        if raw:
            values.pop("enable_thinking")
        return values


@dataclass(frozen=True)
class TrainingOptions:
    eval_every: int = 5
    batch_size: int = 4
    learned_threshold: float = 0.5
    learning_rate: float = 1e-6
    lora_rank: int = 16
    canary_max_tokens: int = 32
    speculative_top_k: int | None = 3
    kl_direction: str = "forward"
    speculative_top_p: float | None = None


def resolve_speculative_policy(
    top_k: int | None, top_p: float | None, *, default_top_k: int | None = None,
) -> tuple[int | None, float | None]:
    """Validate alternative acceptance policies and apply an implicit default."""
    if top_k is not None and top_p is not None:
        raise ValueError("use only one of --speculative-top-k or --speculative-top-p")
    if top_k is not None and top_k <= 0:
        raise ValueError("--speculative-top-k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("--speculative-top-p must be in (0, 1]")
    if top_k is None and top_p is None:
        top_k = default_top_k
    return top_k, top_p


@dataclass(frozen=True)
class CommandConfig:
    mode: Literal["train", "progress", "sample", "score"]
    sampling: SamplingOptions
    training: TrainingOptions | None
    teacher_mode: bool
    sampled_progress: bool = False
    auto_teacher_system: str | None = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="distil",
        description="Visualize per-token likelihood or train a LoRA by on-policy context "
        "distillation. Values support @file and '-' for stdin.",
    )
    p.add_argument("--model", required=True, help="mlx model path or HF repo id (the student)")
    src = p.add_argument_group(
        "input (prompt/completion, rollout, sample, training, or checkpoint progress)"
    )
    src.add_argument("--system", help="optional system prompt")
    src.add_argument("--prompt", help="prompt / context (not scored)")
    src.add_argument("--completion", help="text to score")
    src.add_argument(
        "--raw", action="store_true", help="skip the chat template when scoring or sampling"
    )
    src.add_argument(
        "--no-thinking",
        action="store_true",
        help="pass enable_thinking=False to chat templates that support it",
    )
    src.add_argument(
        "--rollout",
        metavar="PATH",
        help=(
            "verifiers/prime-rl results.jsonl; with --progress, replay its saved prompt "
            "at every checkpoint"
        ),
    )
    src.add_argument("--index", type=int, help="rollout row (default 0)")
    src.add_argument("--train", metavar="CONTEXT", help="distill trusted inline, @file, or stdin context")
    src.add_argument(
        "--progress",
        metavar="PATH",
        help="visualize a completion or probe set over PATH/checkpoints/step-*",
    )
    src.add_argument(
        "--progress-prompt",
        choices=["test", "train"],
        default="test",
        help="which probe prompts to use with --progress --probes (default test)",
    )
    src.add_argument("--probes", metavar="PATH", help="JSONL probes for --progress or manual training")
    tch = p.add_argument_group("teacher (two-pass KL scoring; any flag enables it)")
    tch.add_argument("--teacher-model", help="teacher weights (default: same as --model)")
    tch.add_argument("--teacher-system", help="teacher's system prompt, replacing the student's")
    tch.add_argument("--teacher-prompt", help="replace the first user message for the teacher")
    smp = p.add_argument_group("sampling (generate the completion from the student)")
    smp.add_argument(
        "--sample",
        action="store_true",
        help="sample a completion from --model; with --progress --probes, sample every probe",
    )
    smp.add_argument("--max-tokens", type=int, help="sampling length limit (default 512)")
    smp.add_argument(
        "--temperature",
        type=float,
        help=f"sampling temperature (default {DEFAULT_TEMPERATURE:g})",
    )
    smp.add_argument("--seed", type=int, help="random seed for sampling")
    speculative = smp.add_mutually_exclusive_group()
    speculative.add_argument(
        "--speculative-top-k",
        type=int,
        help="use an interleaved student/teacher policy; replace proposals outside teacher top K",
    )
    speculative.add_argument(
        "--speculative-top-p",
        type=float,
        metavar="P",
        help=(
            "replace proposals outside the teacher's smallest nucleus covering P "
            "probability mass (0 < P <= 1; before temperature)"
        ),
    )
    smp.add_argument(
        "--save-rollout",
        metavar="PATH",
        help="append sampled, supplied, or training completions as results.jsonl",
    )
    trn = p.add_argument_group("LoRA adapter / training")
    trn.add_argument("--adapter", metavar="PATH", help="output adapter/run directory for --train")
    trn.add_argument(
        "--adapter-path",
        metavar="PATH",
        help="save a trained adapter, or load one for scoring/sampling",
    )
    trn.add_argument("--iters", type=int, help="successful optimizer updates (default adaptive)")
    trn.add_argument(
        "--batch-size",
        type=int,
        help="distinct facts per logical optimizer batch (default 4)",
    )
    trn.add_argument("--learning-rate", type=float, help=f"training learning rate (default {TrainingOptions.learning_rate:g})")
    trn.add_argument("--lora-rank", type=int, help="LoRA rank (default 16)")
    trn.add_argument("--from-adapter", metavar="PATH", help="continue an existing LoRA adapter")
    trn.add_argument("--generator-model", help="model used to enumerate facts and questions")
    trn.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"frozen model used for mastery judgments (default: {DEFAULT_JUDGE_MODEL})",
    )
    trn.add_argument(
        "--questions-per-fact",
        type=int,
        help="generated questions per fact; structured generation currently requires 2",
    )
    trn.add_argument("--generate-only", action="store_true", help="compile probes without training")
    trn.add_argument("--eval-every", type=int, help="evaluate and checkpoint every N updates (default 5)")
    trn.add_argument(
        "--learned-threshold",
        type=float,
        help="diagnostic geometric-mean target token probability threshold (default 0.5)",
    )
    trn.add_argument(
        "--checkpoint-every",
        type=int,
        help="save a numbered adapter checkpoint every N updates (default 1)",
    )
    trn.add_argument(
        "--canaries",
        metavar="PATH",
        help="JSONL retention probes checked at every evaluation",
    )
    trn.add_argument(
        "--canary-max-tokens",
        type=int,
        help="maximum generated tokens for each contamination canary (default 32)",
    )
    trn.add_argument(
        "--contamination-regex",
        help="case-insensitive regex marking a canary output as contaminated",
    )
    trn.add_argument(
        "--kl-direction",
        choices=["forward", "reverse"],
        help="KL objective direction (default: forward, teacher || student)",
    )
    out = p.add_argument_group("output")
    out.add_argument(
        "--metric",
        choices=["logprob", "prob", "kl"],
        help="color scale; kl needs a teacher and is its default",
    )
    out.add_argument("--stats", action="store_true", help="show color legend and summary stats")
    out.add_argument("--worst", type=int, default=0, help="table of N most surprising tokens")
    out.add_argument(
        "--top-k",
        type=int,
        help="alternatives per surprising token (default 3); progress alternatives require --json",
    )
    out.add_argument("--hide-context", action="store_true", help="do not echo unscored context")
    out.add_argument("--json", action="store_true", help="print JSON instead of colored text")
    out.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    out.add_argument("--width", type=int, help="force console width")
    return p


def resolve_config(args, argv: list[str]) -> CommandConfig:
    """Validate explicit flags before resolving defaults for the selected mode."""
    explicit = {value.partition("=")[0] for value in argv}
    progress_prompt_explicit = "--progress-prompt" in explicit
    judge_model_explicit = "--judge-model" in explicit
    teacher_mode = bool(
        args.teacher_model or args.teacher_system is not None or args.teacher_prompt is not None
    )
    training = args.train is not None
    progress = args.progress is not None
    rollout_file = args.rollout is not None
    rollout_progress = progress and rollout_file
    standalone_rollout = rollout_file and not progress
    standalone_completion = args.completion is not None and not progress
    sampled_progress_mode = (args.sample or rollout_progress) and progress
    if sum(
        [standalone_rollout, standalone_completion, args.sample and not progress, training, progress]
    ) != 1:
        raise ValueError(
            "provide exactly one of --completion, --rollout, --sample, --train or --progress"
        )
    if judge_model_explicit and not training:
        raise ValueError("--judge-model only combines with --train")
    if args.index is not None and not rollout_file:
        raise ValueError("--index only combines with --rollout")
    if args.top_k is not None and args.top_k < 0:
        raise ValueError("--top-k must be nonnegative")
    if (
        args.sample
        and args.probes is None
        and args.prompt is None
        and (args.raw or args.system is None)
    ):
        raise ValueError("--sample requires --prompt, or --system in chat mode")
    if args.speculative_top_k is not None or args.speculative_top_p is not None:
        flag = "--speculative-top-p" if args.speculative_top_p is not None else "--speculative-top-k"
        if not (args.sample or training):
            raise ValueError(f"{flag} only combines with --sample or --train")
        if sampled_progress_mode:
            raise ValueError(f"{flag} does not combine with --progress")
        resolve_speculative_policy(args.speculative_top_k, args.speculative_top_p)
        if args.sample and not teacher_mode:
            raise ValueError(f"--sample with {flag} requires a --teacher-* flag")
    for value, name in (
        (args.max_tokens, "--max-tokens"),
        (args.temperature, "--temperature"),
        (args.seed, "--seed"),
    ):
        if value is not None and not (args.sample or rollout_progress or training):
            raise ValueError(f"{name} only combines with --sample, --progress --rollout or --train")
    if args.save_rollout is not None and not (args.sample or training or standalone_completion):
        raise ValueError("--save-rollout only combines with --sample, --completion or --train")
    for name in (
        "iters", "batch_size", "learning_rate", "lora_rank", "checkpoint_every",
        "canaries", "canary_max_tokens", "contamination_regex", "kl_direction",
        "from_adapter", "generator_model", "questions_per_fact", "generate_only",
        "eval_every", "learned_threshold", "adapter",
    ):
        value = getattr(args, name)
        if value is not None and (name != "generate_only" or value) and not training:
            raise ValueError(f"--{name.replace('_', '-')} only combines with --train")
    if args.probes is not None and not (progress or training):
        raise ValueError("--probes only combines with --progress or --train")
    if progress_prompt_explicit and not progress:
        raise ValueError("--progress-prompt only combines with --progress")
    if training:
        _validate_training(args)
    if progress:
        _validate_progress(args)
    if args.raw and (args.system or rollout_file or training or args.no_thinking):
        raise ValueError(
            "--raw does not combine with system, rollout, training or chat-template options"
        )
    if args.raw and args.teacher_system is not None:
        raise ValueError("--teacher-system does not combine with --raw; use --teacher-prompt")
    if args.raw and args.save_rollout is not None:
        raise ValueError("--save-rollout does not combine with --raw")
    mode = "train" if training else "progress" if progress else "sample" if args.sample else "score"
    sampling = SamplingOptions(
        max_tokens=args.max_tokens if args.max_tokens is not None else (64 if training else 512),
        temperature=args.temperature if args.temperature is not None else DEFAULT_TEMPERATURE,
        seed=args.seed if args.seed is not None else (0 if training else None),
        enable_thinking=False if args.no_thinking else None,
    )
    training_options = None
    if training:
        values = {
            field.name: getattr(args, field.name) if getattr(args, field.name) is not None else field.default
            for field in fields(TrainingOptions)
        }
        values["speculative_top_k"], values["speculative_top_p"] = resolve_speculative_policy(
            args.speculative_top_k, args.speculative_top_p,
            default_top_k=TrainingOptions.speculative_top_k,
        )
        training_options = TrainingOptions(**values)
    return CommandConfig(mode, sampling, training_options, teacher_mode, sampled_progress_mode)


def _validate_training(args) -> None:
    output = args.adapter or args.adapter_path
    if not output:
        raise ValueError("--train requires --adapter")
    if args.adapter and args.adapter_path and args.adapter != args.adapter_path:
        raise ValueError("use only one of --adapter or --adapter-path for training output")
    if args.teacher_system is not None:
        raise ValueError("--teacher-system does not combine with --train; --train is the context")
    if args.teacher_prompt is not None:
        raise ValueError("--teacher-prompt does not combine with --train")
    if args.prompt is not None:
        raise ValueError("--prompt does not combine with --train; put prompts in the training file")
    if args.raw:
        raise ValueError("--raw does not combine with --train")
    if args.metric or args.stats or args.worst or args.hide_context:
        raise ValueError("scoring output flags do not combine with --train")
    if not args.generate_only and args.canaries is None:
        raise ValueError("--train requires --canaries")
    if args.generate_only and args.probes is not None:
        raise ValueError("--generate-only does not combine with --probes")
    if args.probes is not None and (
        args.generator_model is not None or args.questions_per_fact is not None
    ):
        raise ValueError("--generator-model and --questions-per-fact do not combine with --probes")
    if args.from_adapter is not None and args.lora_rank is not None:
        raise ValueError("--lora-rank does not combine with --from-adapter")
    if args.from_adapter is not None:
        try:
            if Path(args.from_adapter).resolve() == Path(output).resolve():
                raise ValueError("--from-adapter must differ from --adapter")
        except OSError:
            pass
    for name in ("iters", "batch_size", "learning_rate", "lora_rank",
                 "checkpoint_every", "canary_max_tokens"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.questions_per_fact is not None and args.questions_per_fact != 2:
        raise ValueError("--questions-per-fact must be exactly 2")
    if args.eval_every is not None and args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.learned_threshold is not None and not 0 < args.learned_threshold <= 1:
        raise ValueError("--learned-threshold must be in (0, 1]")
    if args.checkpoint_every is not None:
        raise ValueError("--checkpoint-every was replaced by --eval-every")
    if args.contamination_regex is not None:
        raise ValueError("--contamination-regex is not used in continual mode")


def _validate_progress(args) -> None:
    rollout_progress = args.rollout is not None
    target_progress = args.completion is not None
    probe_progress = args.probes is not None
    sampled_progress = args.sample or rollout_progress
    if args.sample and rollout_progress:
        raise ValueError("--sample does not combine with --progress --rollout")
    # Sampling modifies a probe progress run when --probes is supplied;
    # otherwise it is the source of the one ad-hoc progress prompt.
    progress_inputs = sum([
        target_progress,
        probe_progress,
        args.sample and not probe_progress,
        rollout_progress,
    ])
    if progress_inputs != 1:
        raise ValueError(
            "--progress requires exactly one of --completion, --probes, --sample or --rollout"
        )
    if args.adapter_path:
        raise ValueError("--adapter-path does not combine with --progress")
    if args.raw:
        raise ValueError("--raw does not combine with --progress")
    if args.stats or args.worst or args.hide_context:
        raise ValueError("--stats, --worst and --hide-context do not combine with --progress")
    if args.top_k is not None and not args.json:
        raise ValueError("--top-k with --progress requires --json")
    if args.probes is not None:
        if args.prompt is not None:
            raise ValueError("--progress with --probes takes prompts from each probe")
    if rollout_progress and (args.prompt is not None or args.system is not None):
        raise ValueError("--progress --rollout replays the saved prompt; do not pass --prompt or --system")
    if args.progress_prompt == "train" and args.probes is None:
        raise ValueError("--progress-prompt train requires --probes")
    if sampled_progress and args.completion is not None:
        raise ValueError("--completion does not combine with a sampled --progress run")
