"""CLI parsing, validation, and mode-specific defaults; no model loading."""

import argparse
from dataclasses import asdict, dataclass, fields, replace
import copy
import json
import math
from pathlib import Path
from typing import Literal


DEFAULT_JUDGE_MODEL = "mlx-community/Qwen3.5-4B-4bit"
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
    mastery_patience: int = 4
    learning_rate: float = 1e-6
    lora_rank: int = 16
    canary_max_tokens: int = 32
    speculative_top_k: int | None = 3
    kl_direction: str = "forward"
    speculative_top_p: float | None = None


def resolve_speculative_policy(
    top_k: int | None,
    top_p: float | None,
    *,
    default_top_k: int | None = None,
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
class CartridgeOptions:
    positions: int = 128
    passes: int | None = 4
    key_tokens: int = 32
    target_tokens: int = 128
    stride: int = 32
    objective: str = "ntp"


@dataclass(frozen=True)
class CommandConfig:
    mode: Literal[
        "train", "progress", "sample", "score", "memorize", "complete", "resume"
    ]
    sampling: SamplingOptions
    training: TrainingOptions | None
    teacher_mode: bool
    sampled_progress: bool = False
    auto_teacher_system: str | None = None
    backend: str = "lora"
    cartridge: CartridgeOptions | None = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="distill",
        description="Visualize per-token likelihood or train a LoRA by on-policy context "
        "distillation. Values support @file and '-' for stdin.",
    )
    p.add_argument(
        "--model", required=True, help="mlx model path or HF repo id (the student)"
    )
    src = p.add_argument_group(
        "input (prompt/completion, rollout, sample, training, or checkpoint progress)"
    )
    src.add_argument("--system", help="optional system prompt")
    src.add_argument("--prompt", help="prompt / context (not scored)")
    src.add_argument("--completion", help="text to score")
    src.add_argument(
        "--eval-prompt", help="held-out question for inline single-fact training"
    )
    src.add_argument(
        "--canary",
        action="append",
        help="inline retention question; repeat for each canary",
    )
    src.add_argument(
        "--raw",
        action="store_true",
        help="skip the chat template when scoring or sampling",
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
    src.add_argument(
        "--train",
        metavar="CONTEXT",
        help="distill trusted inline, @file, or stdin context",
    )
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
    src.add_argument(
        "--probes",
        metavar="PATH",
        help="JSONL probes for --progress or manual training",
    )
    tch = p.add_argument_group("teacher (two-pass KL scoring; any flag enables it)")
    tch.add_argument(
        "--teacher-model", help="teacher weights (default: same as --model)"
    )
    tch.add_argument(
        "--teacher-system", help="teacher's system prompt, replacing the student's"
    )
    tch.add_argument(
        "--teacher-prompt", help="replace the first user message for the teacher"
    )
    smp = p.add_argument_group("sampling (generate the completion from the student)")
    smp.add_argument(
        "--sample",
        action="store_true",
        help="sample a completion from --model; with --progress --probes, sample every probe",
    )
    smp.add_argument(
        "--max-tokens", type=int, help="sampling length limit (default 512)"
    )
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
    trn.add_argument(
        "--adapter", metavar="PATH", help="output adapter/run directory for --train"
    )
    trn.add_argument(
        "--adapter-path",
        metavar="PATH",
        help="save a trained adapter, or load one for scoring/sampling",
    )
    trn.add_argument(
        "--iters", type=int, help="successful optimizer updates (default adaptive)"
    )
    trn.add_argument(
        "--batch-size",
        type=int,
        help="distinct facts per logical optimizer batch (default 4)",
    )
    trn.add_argument(
        "--learning-rate",
        type=float,
        help=f"training learning rate (default {TrainingOptions.learning_rate:g})",
    )
    trn.add_argument("--lora-rank", type=int, help="LoRA rank (default 16)")
    trn.add_argument(
        "--from-adapter", metavar="PATH", help="continue an existing LoRA adapter"
    )
    trn.add_argument(
        "--generator-model", help="model used to enumerate facts and questions"
    )
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
    trn.add_argument(
        "--generate-only", action="store_true", help="compile probes without training"
    )
    trn.add_argument(
        "--eval-every",
        type=int,
        help="evaluate and checkpoint every N updates (default 5)",
    )
    trn.add_argument(
        "--mastery-confirmations",
        dest="mastery_patience",
        type=int,
        help=f"consecutive passing evaluations required to declare mastery (default {TrainingOptions.mastery_patience})",
    )
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
        "--show-special-tokens",
        action="store_true",
        help="show end-token spellings in checkpoint grids",
    )
    out.add_argument(
        "--metric",
        choices=["logprob", "prob", "kl"],
        help="color scale; kl needs a teacher and is its default",
    )
    out.add_argument(
        "--stats", action="store_true", help="show color legend and summary stats"
    )
    out.add_argument(
        "--worst", type=int, default=0, help="table of N most surprising tokens"
    )
    out.add_argument(
        "--top-k",
        type=int,
        help="alternatives per surprising token (default 3); progress alternatives require --json",
    )
    out.add_argument(
        "--hide-context", action="store_true", help="do not echo unscored context"
    )
    out.add_argument(
        "--json", action="store_true", help="print JSON instead of colored text"
    )
    out.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    out.add_argument("--width", type=int, help="force console width")
    cart = p.add_argument_group("cartridge backend / passage memorization")
    cart.add_argument(
        "--backend",
        choices=["lora", "cartridge"],
        help="learning backend (default lora; memorize defaults cartridge)",
    )
    cart.add_argument(
        "--memorize",
        metavar="CONTEXT",
        help="learn verbatim passage continuations from inline, @file, or stdin text",
    )
    cart.add_argument(
        "--cartridge", metavar="PATH", help="new cartridge run output directory"
    )
    cart.add_argument(
        "--cartridge-path", metavar="PATH", help="cartridge inference artifact"
    )
    cart.add_argument(
        "--from-cartridge",
        metavar="PATH",
        help="refine a cartridge with a fresh optimizer in a new run",
    )
    cart.add_argument(
        "--resume",
        metavar="PATH",
        help="resume a committed cartridge run with an increased total budget",
    )
    cart.add_argument(
        "--cartridge-positions",
        type=int,
        help="prefix capacity including frozen first position (default 128)",
    )
    cart.add_argument(
        "--passes",
        type=int,
        help="total source-window passes (default 4); alternative to --iters",
    )
    cart.add_argument(
        "--objective", choices=["ntp", "kd"], help="source objective (default ntp)"
    )
    cart.add_argument("--key-tokens", type=int, help="source key length (default 32)")
    cart.add_argument(
        "--target-tokens", type=int, help="source continuation length (default 128)"
    )
    cart.add_argument("--stride", type=int, help="source window stride (default 32)")
    cart.add_argument(
        "--complete-key",
        metavar="TEXT",
        help="generate a raw continuation using an artifact's recorded recipe",
    )
    cart.add_argument(
        "--memory-keys",
        metavar="PATH",
        help="recipe-aware source-window JSONL for --progress",
    )
    return p


def _resolve_lora_config(args, argv: list[str]) -> CommandConfig:
    """Validate explicit flags before resolving defaults for the selected mode."""
    explicit = {value.partition("=")[0] for value in argv}
    progress_prompt_explicit = "--progress-prompt" in explicit
    judge_model_explicit = "--judge-model" in explicit
    teacher_mode = bool(
        args.teacher_model
        or args.teacher_system is not None
        or args.teacher_prompt is not None
    )
    training = args.train is not None
    progress = args.progress is not None
    rollout_file = args.rollout is not None
    rollout_progress = progress and rollout_file
    standalone_rollout = rollout_file and not progress
    standalone_completion = args.completion is not None and not (progress or training)
    sampled_progress_mode = (args.sample or rollout_progress) and progress
    if (
        sum(
            [
                standalone_rollout,
                standalone_completion,
                args.sample and not progress,
                training,
                progress,
            ]
        )
        != 1
    ):
        raise ValueError(
            "provide exactly one of --completion, --rollout, --sample, --train or --progress"
        )
    if judge_model_explicit and not training:
        raise ValueError("--judge-model only combines with --train")
    if args.eval_prompt is not None and not training:
        raise ValueError("--eval-prompt only combines with --train")
    if args.canary and not (training or progress):
        raise ValueError("--canary only combines with --train or --progress")
    if args.canary and args.canaries:
        raise ValueError("use only one of --canary or --canaries")
    if args.canary and any(not value.strip() for value in args.canary):
        raise ValueError("--canary must not be empty")
    if args.canary and len(set(args.canary)) != len(args.canary):
        raise ValueError("duplicate --canary questions")
    if args.show_special_tokens and not (training or progress):
        raise ValueError(
            "--show-special-tokens only combines with --train or --progress"
        )
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
        flag = (
            "--speculative-top-p"
            if args.speculative_top_p is not None
            else "--speculative-top-k"
        )
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
            raise ValueError(
                f"{name} only combines with --sample, --progress --rollout or --train"
            )
    if args.save_rollout is not None and not (
        args.sample or training or standalone_completion
    ):
        raise ValueError(
            "--save-rollout only combines with --sample, --completion or --train"
        )
    for name in (
        "iters",
        "batch_size",
        "learning_rate",
        "lora_rank",
        "checkpoint_every",
        "canaries",
        "canary_max_tokens",
        "contamination_regex",
        "kl_direction",
        "from_adapter",
        "generator_model",
        "questions_per_fact",
        "generate_only",
        "eval_every",
        "learned_threshold",
        "mastery_patience",
        "adapter",
    ):
        value = getattr(args, name)
        if value is not None and (name != "generate_only" or value) and not training:
            flag = "mastery-confirmations" if name == "mastery_patience" else name.replace("_", "-")
            raise ValueError(f"--{flag} only combines with --train")
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
        raise ValueError(
            "--teacher-system does not combine with --raw; use --teacher-prompt"
        )
    if args.raw and args.save_rollout is not None:
        raise ValueError("--save-rollout does not combine with --raw")
    mode = (
        "train"
        if training
        else "progress"
        if progress
        else "sample"
        if args.sample
        else "score"
    )
    sampling = SamplingOptions(
        max_tokens=args.max_tokens
        if args.max_tokens is not None
        else (64 if training else 512),
        temperature=args.temperature
        if args.temperature is not None
        else DEFAULT_TEMPERATURE,
        seed=args.seed if args.seed is not None else (0 if training else None),
        enable_thinking=False if args.no_thinking else None,
    )
    training_options = None
    if training:
        values = {
            field.name: getattr(args, field.name)
            if getattr(args, field.name) is not None
            else field.default
            for field in fields(TrainingOptions)
        }
        values["speculative_top_k"], values["speculative_top_p"] = (
            resolve_speculative_policy(
                args.speculative_top_k,
                args.speculative_top_p,
                default_top_k=TrainingOptions.speculative_top_k,
            )
        )
        training_options = TrainingOptions(**values)
    return CommandConfig(
        mode, sampling, training_options, teacher_mode, sampled_progress_mode
    )


def _validate_training(args) -> None:
    output = args.adapter or args.adapter_path
    if not output:
        raise ValueError("--train requires --adapter")
    if args.adapter and args.adapter_path and args.adapter != args.adapter_path:
        raise ValueError(
            "use only one of --adapter or --adapter-path for training output"
        )
    if args.teacher_system is not None:
        raise ValueError(
            "--teacher-system does not combine with --train; --train is the context"
        )
    if args.teacher_prompt is not None:
        raise ValueError("--teacher-prompt does not combine with --train")
    inline = any(
        value is not None for value in (args.prompt, args.eval_prompt, args.completion)
    )
    if inline:
        if not all(
            value and value.strip()
            for value in (args.prompt, args.eval_prompt, args.completion)
        ):
            raise ValueError(
                "inline --train requires --prompt, --eval-prompt and --completion"
            )
        if args.prompt.strip() == args.eval_prompt.strip():
            raise ValueError("--eval-prompt must differ from the training --prompt")
        if (
            args.probes
            or args.generator_model
            or args.questions_per_fact
            or args.generate_only
        ):
            raise ValueError(
                "inline --train cannot combine with --probes or generation options"
            )
    if args.raw:
        raise ValueError("--raw does not combine with --train")
    if args.metric or args.stats or args.worst or args.hide_context:
        raise ValueError("scoring output flags do not combine with --train")
    if not args.generate_only and args.canaries is None and not args.canary:
        raise ValueError("--train requires --canaries or --canary")
    if args.generate_only and args.probes is not None:
        raise ValueError("--generate-only does not combine with --probes")
    if args.probes is not None and (
        args.generator_model is not None or args.questions_per_fact is not None
    ):
        raise ValueError(
            "--generator-model and --questions-per-fact do not combine with --probes"
        )
    if args.from_adapter is not None and args.lora_rank is not None:
        raise ValueError("--lora-rank does not combine with --from-adapter")
    if args.from_adapter is not None:
        try:
            if Path(args.from_adapter).resolve() == Path(output).resolve():
                raise ValueError("--from-adapter must differ from --adapter")
        except OSError:
            pass
    for name in (
        "iters",
        "batch_size",
        "learning_rate",
        "lora_rank",
        "checkpoint_every",
        "canary_max_tokens",
        "mastery_patience",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            flag = "mastery-confirmations" if name == "mastery_patience" else name.replace("_", "-")
            raise ValueError(f"--{flag} must be positive")
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
    progress_inputs = sum(
        [
            target_progress,
            probe_progress,
            args.sample and not probe_progress,
            rollout_progress,
        ]
    )
    recorded = (
        progress_inputs == 0 and (Path(args.progress) / "eval_metrics.jsonl").is_file()
    )
    if recorded and (args.prompt is not None or args.canary or args.system is not None):
        raise ValueError(
            "recorded --progress replays saved inputs; supply --completion or --sample to override"
        )
    if progress_inputs != 1 and not recorded:
        raise ValueError(
            "--progress requires exactly one of --completion, --probes, --sample or --rollout"
        )
    if args.adapter_path:
        raise ValueError("--adapter-path does not combine with --progress")
    if args.raw:
        raise ValueError("--raw does not combine with --progress")
    if args.stats or args.worst or args.hide_context:
        raise ValueError(
            "--stats, --worst and --hide-context do not combine with --progress"
        )
    if args.top_k is not None and not args.json:
        raise ValueError("--top-k with --progress requires --json")
    if args.probes is not None:
        if args.prompt is not None:
            raise ValueError("--progress with --probes takes prompts from each probe")
    if rollout_progress and (args.prompt is not None or args.system is not None):
        raise ValueError(
            "--progress --rollout replays the saved prompt; do not pass --prompt or --system"
        )
    if args.progress_prompt == "train" and args.probes is None:
        raise ValueError("--progress-prompt train requires --probes")
    if sampled_progress and args.completion is not None:
        raise ValueError("--completion does not combine with a sampled --progress run")


def resolve_config(args, argv: list[str]) -> CommandConfig:
    """Resolve recipe/backend first; retain the original LoRA validator unchanged."""
    explicit = {v.partition("=")[0] for v in argv if v.startswith("--")}
    if sum(v.partition("=")[0] == "--cartridge-path" for v in argv) > 1:
        raise ValueError("multiple attached cartridges are unsupported")
    memorizing = args.memorize is not None
    complete = args.complete_key is not None
    backend = args.backend or (
        "cartridge"
        if memorizing
        or complete
        or args.cartridge_path
        or args.resume
        or args.from_cartridge
        else "lora"
    )
    if args.cartridge_path and args.backend == "lora":
        raise ValueError("--backend lora conflicts with --cartridge-path")
    if memorizing and backend != "cartridge":
        raise ValueError("--memorize requires the cartridge backend")
    if args.progress:
        manifests = list(
            Path(args.progress).glob("checkpoints/step-*/cartridge.json")
        ) + list(Path(args.progress).glob("step-*/cartridge.json"))
        lora_artifacts = list(
            Path(args.progress).glob("checkpoints/step-*/adapters.safetensors")
        ) + list(Path(args.progress).glob("step-*/adapters.safetensors"))
        if manifests and lora_artifacts:
            raise ValueError("mixed cartridge and LoRA checkpoints are unsupported")
        if lora_artifacts and args.backend == "cartridge":
            raise ValueError("explicit backend conflicts with LoRA checkpoints")
        if manifests:
            if args.backend == "lora":
                raise ValueError(
                    "explicit backend conflicts with cartridge checkpoint manifests"
                )
            backend = "cartridge"
    cart_flags = (
        "cartridge",
        "cartridge_path",
        "from_cartridge",
        "resume",
        "cartridge_positions",
        "passes",
        "objective",
        "key_tokens",
        "target_tokens",
        "stride",
        "complete_key",
        "memory_keys",
    )
    if backend == "lora":
        if any(getattr(args, name) is not None for name in cart_flags):
            raise ValueError("cartridge flags require --backend cartridge")
        return _resolve_lora_config(args, argv)
    if any(
        getattr(args, name) is not None
        for name in ("adapter", "adapter_path", "from_adapter", "lora_rank")
    ):
        raise ValueError("LoRA rank and adapter flags do not combine with cartridges")
    if args.top_k is not None and args.top_k < 0:
        raise ValueError("--top-k must be nonnegative")
    if args.seed is not None and not 0 <= args.seed < 2**64:
        raise ValueError("--seed must be an unsigned 64-bit integer")
    if args.kl_direction == "reverse":
        raise ValueError("reverse KL is unsupported for cartridges")
    if (
        args.train is not None
        and args.teacher_model is not None
        and args.teacher_model != args.model
    ):
        raise ValueError(
            "cartridge distillation requires the same frozen base teacher identity"
        )
    if args.cartridge_positions is not None and args.cartridge_positions < 2:
        raise ValueError("--cartridge-positions must be at least 2")
    for name in (
        "passes",
        "iters",
        "key_tokens",
        "target_tokens",
        "stride",
        "batch_size",
        "eval_every",
        "max_tokens",
        "canary_max_tokens",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_rate", "temperature"):
        value = getattr(args, name)
        if value is not None and (
            not math.isfinite(value)
            or value < 0
            or (name == "learning_rate" and value == 0)
        ):
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and {'positive' if name == 'learning_rate' else 'nonnegative'}"
            )
    if args.passes is not None and args.iters is not None:
        raise ValueError("use only one of --passes or --iters")
    if args.resume:
        allowed = {
            "--model",
            "--resume",
            "--passes",
            "--iters",
            "--json",
            "--color",
            "--width",
        }
        invalid = explicit - allowed
        if invalid:
            raise ValueError(
                "resume only permits an increased total budget and presentation flags; incompatible: "
                + ", ".join(sorted(invalid))
            )
        config = json.loads((Path(args.resume) / "run_config.json").read_text())
        if config.get("backend") != "cartridge" or config.get("schema_version") != 2:
            raise ValueError(
                "resume requires an integrated cartridge run; inference imports cannot resume"
            )
        if config.get("model") != args.model:
            raise ValueError("resume model identity mismatch")
        field = (
            "iters"
            if config.get("iters") is not None or config.get("recipe") == "facts"
            else "passes"
        )
        value = getattr(args, field)
        previous_budget = config.get(field) or config.get("resolved_iters", 0)
        if (
            value is None
            or value <= previous_budget
            or getattr(args, "passes" if field == "iters" else "iters") is not None
        ):
            raise ValueError(
                "resume requires an increased total budget in the original units"
            )
        selection_path = Path(args.resume) / "selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection.get("reason") in (
                "locality",
                "mastered",
                "already_mastered",
            ) or selection.get("stopped") in ("success", "already_mastered"):
                raise ValueError(
                    "cannot resume a fact run stopped for success or contamination"
                )
        return CommandConfig("resume", SamplingOptions(), None, False, backend=backend)
    if memorizing:
        if any(
            (
                args.train is not None,
                args.progress is not None,
                args.sample,
                args.rollout is not None,
                args.completion is not None,
                complete,
            )
        ):
            raise ValueError("--memorize is exclusive with other inputs")
        if not args.cartridge:
            raise ValueError("--memorize requires --cartridge")
        allowed = {
            "--model",
            "--backend",
            "--memorize",
            "--cartridge",
            "--from-cartridge",
            "--cartridge-positions",
            "--passes",
            "--iters",
            "--objective",
            "--key-tokens",
            "--target-tokens",
            "--stride",
            "--learning-rate",
            "--batch-size",
            "--eval-every",
            "--seed",
            "--canaries",
            "--canary-max-tokens",
            "--json",
            "--color",
            "--width",
        }
        invalid = explicit - allowed
        if invalid:
            raise ValueError(
                ", ".join(sorted(invalid)) + " does not combine with memorization"
            )
        if (
            args.from_cartridge
            and Path(args.from_cartridge).resolve() == Path(args.cartridge).resolve()
        ):
            raise ValueError("--from-cartridge must differ from --cartridge")
        training = TrainingOptions(
            learning_rate=args.learning_rate
            if args.learning_rate is not None
            else 0.005,
            batch_size=args.batch_size if args.batch_size is not None else 1,
            eval_every=args.eval_every if args.eval_every is not None else 5,
            canary_max_tokens=args.canary_max_tokens
            if args.canary_max_tokens is not None
            else 32,
        )
        options = CartridgeOptions(
            positions=args.cartridge_positions or 128,
            passes=args.passes
            if args.passes is not None
            else (None if args.iters is not None else 4),
            key_tokens=args.key_tokens or 32,
            target_tokens=args.target_tokens or 128,
            stride=args.stride or 32,
            objective=args.objective or "ntp",
        )
        return CommandConfig(
            "memorize",
            SamplingOptions(seed=args.seed if args.seed is not None else 0),
            training,
            False,
            backend=backend,
            cartridge=options,
        )
    for name in ("passes", "objective", "key_tokens", "target_tokens", "stride"):
        if getattr(args, name) is not None:
            raise ValueError("passage-window flags require --memorize")
    if complete:
        if not args.cartridge_path:
            raise ValueError("--complete-key requires --cartridge-path")
        if any(
            (
                args.prompt is not None,
                args.completion is not None,
                args.rollout is not None,
                args.sample,
                args.train is not None,
                args.progress is not None,
                args.system is not None,
                args.raw,
                args.no_thinking,
            )
        ):
            raise ValueError(
                "--complete-key is exclusive with prompt, completion, rollout, sample and chat options"
            )
        allowed = {
            "--model",
            "--cartridge-path",
            "--complete-key",
            "--backend",
            "--max-tokens",
            "--temperature",
            "--seed",
            "--metric",
            "--stats",
            "--worst",
            "--top-k",
            "--hide-context",
            "--json",
            "--color",
            "--width",
        }
        if explicit - allowed:
            raise ValueError(
                "unsupported complete-key options: "
                + ", ".join(sorted(explicit - allowed))
            )
        return CommandConfig(
            "complete",
            SamplingOptions(
                max_tokens=args.max_tokens or 512,
                temperature=args.temperature
                if args.temperature is not None
                else DEFAULT_TEMPERATURE,
                seed=args.seed,
            ),
            None,
            False,
            backend=backend,
        )
    clone = copy.copy(args)
    if args.train is not None:
        if not args.cartridge:
            raise ValueError("--train with cartridge backend requires --cartridge")
        if args.cartridge_path or args.memory_keys:
            raise ValueError(
                "inference artifacts and memory keys do not combine with --train"
            )
        if (
            args.from_cartridge
            and Path(args.from_cartridge).resolve() == Path(args.cartridge).resolve()
        ):
            raise ValueError("--from-cartridge must differ from --cartridge")
        clone.adapter = args.cartridge
    else:
        if (
            args.cartridge
            or args.from_cartridge
            or args.cartridge_positions is not None
        ):
            raise ValueError(
                "cartridge output, capacity and refinement flags require training"
            )
        if args.memory_keys is not None:
            if any(
                (
                    args.teacher_model,
                    args.teacher_system is not None,
                    args.teacher_prompt is not None,
                    args.no_thinking,
                )
            ):
                raise ValueError(
                    "teacher and chat flags do not combine with raw memory progress"
                )
            if not args.progress or any(
                (
                    args.probes,
                    args.completion,
                    args.sample,
                    args.rollout,
                    args.prompt,
                    args.system,
                )
            ):
                raise ValueError(
                    "--memory-keys requires --progress and is exclusive with existing progress inputs"
                )
            clone.completion = "recipe-aware memory progress"
        elif not args.progress and not args.cartridge_path:
            raise ValueError("cartridge inference requires --cartridge-path")
        if args.progress and args.cartridge_path:
            raise ValueError("--cartridge-path does not combine with --progress")
    config = _resolve_lora_config(clone, argv)
    if config.training:
        config = replace(
            config,
            training=replace(
                config.training,
                learning_rate=args.learning_rate
                if args.learning_rate is not None
                else 0.001,
            ),
        )
    return replace(
        config,
        backend=backend,
        cartridge=CartridgeOptions(
            positions=args.cartridge_positions or 128, passes=None
        ),
    )
