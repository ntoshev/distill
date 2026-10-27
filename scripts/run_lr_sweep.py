#!/usr/bin/env python3
"""Run the Project Lumen learning-rate sweep with per-step checkpoints."""

import argparse
import json
from pathlib import Path
from time import perf_counter

# The useful transition window for this task is below the original sweep.
# Keep approximately constant cumulative LR while resolving the low-rate range.
DEFAULT_SPECS = "2e-6:100,1e-6:200,5e-7:400"


def parse_specs(value: str) -> list[tuple[str, float, int]]:
    specs = []
    labels = set()
    for item in value.split(","):
        try:
            label, steps = item.strip().split(":", 1)
            learning_rate = float(label)
            iterations = int(steps)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "specs must be comma-separated LEARNING_RATE:STEPS pairs"
            ) from error
        if learning_rate <= 0 or iterations <= 0:
            raise ValueError("learning rates and step counts must be positive")
        if label in labels:
            raise ValueError(f"duplicate learning-rate label: {label}")
        labels.add(label)
        specs.append((label, learning_rate, iterations))
    if not specs:
        raise ValueError("at least one learning-rate spec is required")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-4bit")
    parser.add_argument("--specs", default=DEFAULT_SPECS)
    prompts_group = parser.add_mutually_exclusive_group()
    prompts_group.add_argument("--prompts", type=Path)
    prompts_group.add_argument("--prompt", action="append", help="inline training question; repeatable")
    teacher_group = parser.add_mutually_exclusive_group()
    teacher_group.add_argument("--teacher", type=Path)
    teacher_group.add_argument("--teacher-text", help="inline privileged teacher context")
    canaries_group = parser.add_mutually_exclusive_group()
    canaries_group.add_argument(
        "--canaries",
        type=Path,
    )
    canaries_group.add_argument("--canary", action="append", help="inline retention question; repeatable")
    parser.add_argument(
        "--speculative-top-k",
        type=int,
        help="run the sweep with interleaved Speculative KD at this K",
    )
    parser.add_argument(
        "--kl-direction",
        choices=["forward", "reverse"],
        default="forward",
        help="KL objective direction (default: forward)",
    )
    args = parser.parse_args()

    try:
        specs = parse_specs(args.specs)
    except ValueError as error:
        parser.error(str(error))
    if args.speculative_top_k is not None and args.speculative_top_k <= 0:
        parser.error("--speculative-top-k must be positive")

    from mlx_lm import load
    from mlx_lm.tuner.utils import remove_lora_layers

    from distill.training import load_prompts, train_lora

    try:
        prompts = args.prompt or load_prompts(args.prompts or Path("examples/distillation/lumen-questions.txt"))
        canaries = args.canary or load_prompts(args.canaries or Path("examples/distillation/contamination-canaries.txt"))
        teacher_system = args.teacher_text if args.teacher_text is not None else (
            args.teacher or Path("examples/distillation/lumen-teacher.txt")
        ).read_text()
        if not all(text.strip() for text in [*prompts, *canaries, teacher_system]):
            raise ValueError("prompts, canaries, and teacher context must not be empty")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.output.exists():
        if not args.output.is_dir():
            parser.error(f"output path is not a directory: {args.output}")
        if any(args.output.iterdir()):
            parser.error(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(args.model)

    manifest = {
        "model": args.model,
        "specs": [
            {"label": label, "learning_rate": learning_rate, "steps": steps}
            for label, learning_rate, steps in specs
        ],
        "prompts": prompts,
        "teacher": teacher_system,
        "canaries": canaries,
        "kl_direction": args.kl_direction,
        "sampling": {
            "policy": "speculative" if args.speculative_top_k is not None else "student",
            "top_k": args.speculative_top_k,
        },
        "runs": [],
    }

    for label, learning_rate, iterations in specs:
        run_path = args.output / f"lr-{label}"
        run_started = perf_counter()

        def report(step):
            canary_results = step.canaries or []
            changed = sum(result.changed for result in canary_results)
            contaminated = sum(result.contaminated for result in canary_results)
            print(
                f"lr={label} step={step.iteration}/{iterations} "
                f"loss={step.loss:.4f} train={step.train_seconds:.3f}s "
                f"checkpoint={step.checkpoint_seconds:.3f}s "
                f"canaries={step.canary_seconds:.3f}s "
                f"changed={changed}/{len(canary_results)} "
                f"contaminated={contaminated}/{len(canary_results)}"
                + (
                    f" accepted={step.accepted_tokens}/{step.num_tokens}"
                    if step.speculative_top_k is not None
                    else ""
                ),
                flush=True,
            )

        try:
            result = train_lora(
                model,
                tokenizer,
                prompts,
                teacher_system,
                run_path,
                model_name=args.model,
                iters=iterations,
                learning_rate=learning_rate,
                lora_rank=4,
                max_tokens=16,
                temperature=0.0,
                seed=0,
                enable_thinking=False,
                checkpoint_every=1,
                canary_prompts=canaries,
                canary_max_tokens=16,
                contamination_regex=r"^\s*VIO",
                speculative_top_k=args.speculative_top_k,
                kl_direction=args.kl_direction,
                report=report,
            )
            manifest["runs"].append(
                {
                    "label": label,
                    "learning_rate": learning_rate,
                    "steps": iterations,
                    "wall_seconds": perf_counter() - run_started,
                    "canary_baseline_seconds": result.canary_baseline_seconds,
                    "adapter_file": str(result.adapter_file),
                    "metrics_file": str(result.metrics_file),
                }
            )
        finally:
            remove_lora_layers(model)
            model.eval()
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
