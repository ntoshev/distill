"""Command-line interface."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from distil.render import RenderOptions, render, to_json_dict
from distil.rollout import load_rollout, save_rollout
from distil.sample import sample_completion
from distil.scoring import score_text, score_text_pair
from distil.templating import apply_teacher_context, build_raw, build_rollout


def _read_input(value: str | None) -> str | None:
    """Inline string, @path to read a file, or "-" for stdin."""
    if value is None:
        return None
    if value == "-":
        return sys.stdin.read()
    if value.startswith("@"):
        with open(value[1:]) as f:
            return f.read()
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="distil",
        description="Visualize per-token likelihood or train a LoRA by on-policy context "
        "distillation. Values support @file and '-' for stdin.",
    )
    p.add_argument("--model", required=True, help="mlx model path or HF repo id (the student)")
    src = p.add_argument_group("input (prompt/completion, rollout, sample, or training)")
    src.add_argument("--system", help="optional system prompt")
    src.add_argument("--prompt", help="prompt / context (not scored)")
    src.add_argument("--completion", help="text to score")
    src.add_argument("--raw", action="store_true", help="skip the chat template")
    src.add_argument(
        "--no-thinking",
        action="store_true",
        help="pass enable_thinking=False to chat templates that support it",
    )
    src.add_argument("--rollout", help="verifiers/prime-rl results.jsonl")
    src.add_argument("--index", type=int, default=0, help="rollout row (default 0)")
    src.add_argument("--train", metavar="PATH", help="train LoRA from one prompt per line")
    tch = p.add_argument_group("teacher (two-pass KL scoring; any flag enables it)")
    tch.add_argument("--teacher-model", help="teacher weights (default: same as --model)")
    tch.add_argument("--teacher-system", help="teacher's system prompt, replacing the student's")
    tch.add_argument("--teacher-prompt", help="replace the first user message for the teacher")
    smp = p.add_argument_group("sampling (generate the completion from the student)")
    smp.add_argument("--sample", action="store_true", help="sample a completion from --model")
    smp.add_argument("--max-tokens", type=int, help="sampling length limit (default 512)")
    smp.add_argument("--temperature", type=float, help="sampling temperature (default 1.0)")
    smp.add_argument("--seed", type=int, help="random seed for sampling")
    smp.add_argument(
        "--save-rollout",
        metavar="PATH",
        help="append sampled or training rollouts to PATH as results.jsonl",
    )
    trn = p.add_argument_group("LoRA adapter / training")
    trn.add_argument(
        "--adapter-path",
        metavar="PATH",
        help="save a trained adapter, or load one for scoring/sampling",
    )
    trn.add_argument("--iters", type=int, help="successful training updates (default 100)")
    trn.add_argument("--learning-rate", type=float, help="training learning rate (default 1e-5)")
    trn.add_argument("--lora-rank", type=int, help="LoRA rank (default 16)")
    out = p.add_argument_group("output")
    out.add_argument(
        "--metric",
        choices=["logprob", "prob", "kl"],
        help="color scale; kl needs a teacher and is its default",
    )
    out.add_argument("--stats", action="store_true", help="show color legend and summary stats")
    out.add_argument("--worst", type=int, default=0, help="table of N most surprising tokens")
    out.add_argument("--top-k", type=int, default=3, help="alternatives per surprising token")
    out.add_argument("--hide-context", action="store_true", help="do not echo unscored context")
    out.add_argument("--json", action="store_true", help="print JSON instead of colored text")
    out.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    out.add_argument("--width", type=int, help="force console width")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    force = True if args.color == "always" else None
    console = Console(
        force_terminal=force,
        no_color=args.color == "never",
        color_system=None if args.color == "never" else "truecolor",
        width=args.width,
    )
    errcon = Console(stderr=True)

    def fail(message: str) -> int:
        errcon.print(f"[red]error:[/red] {message}")
        return 2

    teacher_mode = bool(
        args.teacher_model or args.teacher_system is not None or args.teacher_prompt is not None
    )
    training = args.train is not None
    if sum([bool(args.rollout), args.completion is not None, args.sample, training]) != 1:
        return fail("provide exactly one of --completion, --rollout, --sample or --train")
    if args.sample and not args.prompt:
        return fail("--sample requires --prompt")
    for value, name in (
        (args.max_tokens, "--max-tokens"),
        (args.temperature, "--temperature"),
        (args.seed, "--seed"),
        (args.save_rollout, "--save-rollout"),
    ):
        if value is not None and not (args.sample or training):
            return fail(f"{name} only combines with --sample or --train")
    for value, name in (
        (args.iters, "--iters"),
        (args.learning_rate, "--learning-rate"),
        (args.lora_rank, "--lora-rank"),
    ):
        if value is not None and not training:
            return fail(f"{name} only combines with --train")
    if training:
        if args.teacher_system is None:
            return fail("--train requires --teacher-system with privileged context")
        if not args.adapter_path:
            return fail("--train requires --adapter-path")
        if args.teacher_prompt is not None:
            return fail("--teacher-prompt does not combine with --train")
        if args.prompt is not None:
            return fail("--prompt does not combine with --train; put prompts in the training file")
        if args.raw:
            return fail("--raw does not combine with --train")
        if args.metric or args.stats or args.worst or args.hide_context or args.json:
            return fail("scoring output flags do not combine with --train")
        if args.iters is not None and args.iters <= 0:
            return fail("--iters must be positive")
        if args.learning_rate is not None and args.learning_rate <= 0:
            return fail("--learning-rate must be positive")
        if args.lora_rank is not None and args.lora_rank <= 0:
            return fail("--lora-rank must be positive")
    if args.raw and (
        args.system or args.rollout or args.sample or training or teacher_mode or args.no_thinking
    ):
        return fail(
            "--raw only combines with --prompt/--completion "
            "(no system, teacher, sampling or chat-template options)"
        )
    if args.metric == "kl" and not teacher_mode:
        return fail("--metric kl needs teacher-mode scoring (a --teacher-* flag)")
    metric = args.metric or ("kl" if teacher_mode else "logprob")

    meta = {}
    prompts = None
    if training:
        from distil.training import load_prompts

        try:
            prompts = load_prompts(args.train)
        except (FileNotFoundError, ValueError) as e:
            return fail(str(e))
        completion = None
        prompt = ""
        system = _read_input(args.system)
    elif args.rollout:
        try:
            rollout = load_rollout(args.rollout, args.index)
        except (FileNotFoundError, IndexError, ValueError) as e:
            return fail(str(e))
        meta = rollout.meta
    else:
        completion = _read_input(args.completion)
        prompt = _read_input(args.prompt) or ""
        system = _read_input(args.system)
    teacher_system = _read_input(args.teacher_system)
    teacher_prompt = _read_input(args.teacher_prompt)

    from mlx_lm import load  # deferred: slow import

    if args.adapter_path and not training and not Path(args.adapter_path).exists():
        return fail(f"adapter path not found: {args.adapter_path}")
    model, tokenizer = load(
        args.model,
        adapter_path=None if training else args.adapter_path,
    )
    teacher_name = args.teacher_model or args.model
    if teacher_mode and teacher_name != args.model:
        teacher_model, teacher_tokenizer = load(teacher_name)
    else:
        teacher_model, teacher_tokenizer = model, tokenizer

    max_tokens = args.max_tokens if args.max_tokens is not None else 512
    temperature = args.temperature if args.temperature is not None else 1.0
    enable_thinking = False if args.no_thinking else None

    if training:
        from distil.training import train_lora

        def report(step):
            errcon.print(
                f"iter {step.iteration}: loss={step.loss:.4f}  "
                f"tokens={step.num_tokens}  finish={step.finish_reason}",
                style="grey50",
            )

        try:
            trained = train_lora(
                model,
                tokenizer,
                prompts,
                teacher_system,
                args.adapter_path,
                model_name=args.model,
                student_system=system,
                teacher_model=teacher_model,
                teacher_tokenizer=teacher_tokenizer,
                iters=args.iters if args.iters is not None else 100,
                learning_rate=(
                    args.learning_rate if args.learning_rate is not None else 1e-5
                ),
                lora_rank=args.lora_rank if args.lora_rank is not None else 16,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=args.seed if args.seed is not None else 0,
                enable_thinking=enable_thinking,
                save_rollout_path=args.save_rollout,
                report=report,
                warn=lambda message: errcon.print(
                    f"[yellow]warning:[/yellow] {message}"
                ),
            )
        except (OSError, RuntimeError, ValueError) as e:
            return fail(str(e))
        errcon.print(f"saved adapter to {trained.adapter_file}", style="grey50")
        return 0

    sample = None
    if args.rollout:
        messages = rollout.messages
        built = build_rollout(tokenizer, messages, enable_thinking=enable_thinking)
    elif args.sample:
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": prompt})
        sample = sample_completion(
            model, tokenizer, messages,
            max_tokens=max_tokens, temperature=temperature, seed=args.seed,
            enable_thinking=enable_thinking,
        )
        errcon.print(
            f"sampled {sample.num_tokens} tokens (finish: {sample.finish_reason})", style="grey50"
        )
        if args.save_rollout:
            # saved before scoring, so a scoring failure never loses the sample
            index = save_rollout(
                args.save_rollout,
                messages,
                [{"role": "assistant", "content": sample.content}],
                extra={
                    "task": "distil-sample",
                    "model": args.model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **({"seed": args.seed} if args.seed is not None else {}),
                    "is_truncated": sample.finish_reason != "stop",
                },
            )
            errcon.print(f"saved sample to {args.save_rollout} as index {index}", style="grey50")
        # the rendered generation prompt plus the generated text is already the
        # exact scored string; no re-templating round-trip
        built = build_raw(sample.prompt_text, sample.completion_text)
    elif args.raw:
        built = build_raw(prompt, completion)
    else:
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": completion})
        built = build_rollout(tokenizer, messages, enable_thinking=enable_thinking)

    if teacher_mode:
        try:
            teacher_messages = apply_teacher_context(
                messages, system=teacher_system, prompt=teacher_prompt
            )
        except ValueError as e:
            return fail(str(e))
        if args.sample:
            template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
            teacher_prompt_text = teacher_tokenizer.apply_chat_template(
                teacher_messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            built_teacher = build_raw(teacher_prompt_text, sample.completion_text)
        else:
            built_teacher = build_rollout(
                teacher_tokenizer, teacher_messages, enable_thinking=enable_thinking
            )
        try:
            result = score_text_pair(
                model, tokenizer, built,
                teacher_model, teacher_tokenizer, built_teacher,
                model_name=args.model, teacher_name=teacher_name, top_k=args.top_k,
                teacher_without_adapter=bool(args.adapter_path and teacher_model is model),
            )
        except ValueError as e:
            errcon.print(f"[red]error:[/red] {e}")
            return 1
    else:
        result = score_text(
            model,
            tokenizer,
            built.text,
            built.spans,
            model_name=args.model,
            top_k=args.top_k,
            warnings=built.warnings,
        )

    if args.json:
        payload = to_json_dict(result)
        if meta:
            payload["rollout"] = meta
        if sample:
            payload["sample"] = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": args.seed,
                "finish_reason": sample.finish_reason,
                "num_tokens": sample.num_tokens,
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for w in result.warnings:
        errcon.print(f"[yellow]warning:[/yellow] {w}")
    if meta:
        errcon.print("rollout: " + "  ".join(f"{k}={v}" for k, v in meta.items()), style="grey50")
    if teacher_mode:
        overrides = [
            name for name, v in (("system", teacher_system), ("prompt", teacher_prompt))
            if v is not None
        ]
        line = f"teacher: {teacher_name}"
        if overrides:
            line += "  overrides: " + ", ".join(overrides)
        errcon.print(line, style="grey50")
    render(
        result,
        console,
        RenderOptions(
            metric=metric,
            hide_context=args.hide_context,
            stats=args.stats,
            worst=args.worst,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
