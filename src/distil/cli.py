"""Command-line interface."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from distil.render import (
    RenderOptions,
    render,
    render_progress,
    render_progress_tokens,
    to_json_dict,
)
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
    src = p.add_argument_group(
        "input (prompt/completion, rollout, sample, training, or checkpoint progress)"
    )
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
    src.add_argument("--train", metavar="CONTEXT", help="distill trusted inline, @file, or stdin context")
    src.add_argument(
        "--progress",
        metavar="PATH",
        help="visualize a completion or probe set over PATH/checkpoints/step-*",
    )
    src.add_argument("--probes", metavar="PATH", help="JSONL probes for --progress or manual training")
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
    trn.add_argument("--adapter", metavar="PATH", help="output adapter/run directory for --train")
    trn.add_argument(
        "--adapter-path",
        metavar="PATH",
        help="save a trained adapter, or load one for scoring/sampling",
    )
    trn.add_argument("--iters", type=int, help="successful training updates (default 100)")
    trn.add_argument("--learning-rate", type=float, help="training learning rate (default 2e-6)")
    trn.add_argument("--lora-rank", type=int, help="LoRA rank (default 16)")
    trn.add_argument("--from-adapter", metavar="PATH", help="continue an existing LoRA adapter")
    trn.add_argument("--generator-model", help="model used to enumerate facts and questions")
    trn.add_argument("--questions-per-fact", type=int, help="generated questions per fact (default 2)")
    trn.add_argument("--generate-only", action="store_true", help="compile probes without training")
    trn.add_argument("--eval-every", type=int, help="evaluate and checkpoint every N updates (default 5)")
    trn.add_argument(
        "--learned-threshold",
        type=float,
        help="geometric-mean target token probability required for mastery (default 0.5)",
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
        "--speculative-top-k",
        type=int,
        help="use interleaved Speculative KD; replace proposals outside teacher top K",
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
    out.add_argument("--top-k", type=int, default=3, help="alternatives per surprising token")
    out.add_argument("--hide-context", action="store_true", help="do not echo unscored context")
    out.add_argument("--json", action="store_true", help="print JSON instead of colored text")
    out.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    out.add_argument("--width", type=int, help="force console width")
    return p


def _continual_cli(args, errcon, fail) -> int:
    """Compile the document, load the starting policies, and run continual training."""
    from distil.generation import (
        answer_facts,
        ensure_fits,
        generate_facts,
        resolve_context,
        teacher_system as document_teacher_system,
        write_generation,
    )
    from distil.progress import load_probes
    from distil.sample import sample_completion
    from distil.stopping import load_learning_items

    output = args.adapter or args.adapter_path
    selection = Path(output) / "selection.json"
    if selection.is_file():
        try:
            if json.loads(selection.read_text()).get("stopped") in {
                "success",
                "already_mastered",
            }:
                return fail(f"refusing to overwrite successful adapter run: {output}")
        except (OSError, json.JSONDecodeError):
            return fail(f"invalid existing selection file: {selection}")
    try:
        document, source = resolve_context(args.train, stdin=sys.stdin.read)
        canaries = load_probes(args.canaries) if not args.generate_only else []
        items = load_learning_items(args.probes) if args.probes else None
    except (FileNotFoundError, OSError, ValueError) as error:
        return fail(str(error))

    from mlx_lm import load

    enable_thinking = False if args.no_thinking else None
    student_system = _read_input(args.system)
    seed = args.seed if args.seed is not None else 0
    max_tokens = args.max_tokens if args.max_tokens is not None else 64
    generator_name = args.generator_model or args.model
    student = student_tokenizer = None

    try:
        if items is None:
            generator, generator_tokenizer = load(generator_name)

            def generate(prompt: str) -> str:
                messages = [{"role": "user", "content": prompt}]
                ensure_fits(
                    generator,
                    generator_tokenizer,
                    messages,
                    max(256, max_tokens),
                    "generator prompt",
                    enable_thinking,
                )
                return sample_completion(
                    generator,
                    generator_tokenizer,
                    messages,
                    max_tokens=max(256, max_tokens),
                    temperature=0.7,
                    seed=seed,
                    enable_thinking=enable_thinking,
                ).content

            facts = generate_facts(
                document,
                args.questions_per_fact if args.questions_per_fact is not None else 2,
                generate,
            )
            if generator_name == args.model and args.from_adapter is None:
                student, student_tokenizer = generator, generator_tokenizer
            else:
                del generator, generator_tokenizer
                import mlx.core as mx

                mx.clear_cache()

        if student is None:
            student, student_tokenizer = load(
                args.model, adapter_path=args.from_adapter
            )

        if args.teacher_model is not None:
            teacher, teacher_tokenizer = load(args.teacher_model)
        elif args.from_adapter is not None:
            teacher, teacher_tokenizer = load(
                args.model, adapter_path=args.from_adapter
            )
        else:
            teacher, teacher_tokenizer = student, student_tokenizer

        teacher_context = document_teacher_system(document)
        if items is None:
            def answer(prompt: str) -> str:
                messages = [
                    {"role": "system", "content": teacher_context},
                    {"role": "user", "content": prompt},
                ]
                ensure_fits(
                    teacher,
                    teacher_tokenizer,
                    messages,
                    max_tokens,
                    "teacher document prompt",
                    enable_thinking,
                )
                return sample_completion(
                    teacher,
                    teacher_tokenizer,
                    messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=0,
                    enable_thinking=enable_thinking,
                ).content

            answer_facts(facts, answer)
            probes_path = write_generation(
                Path(output) / "generated",
                document,
                source,
                facts,
                generator_model=generator_name,
                questions_per_fact=(
                    args.questions_per_fact if args.questions_per_fact is not None else 2
                ),
                seed=seed,
            )
            items = load_learning_items(probes_path)
        else:
            probes_path = Path(args.probes)

        if args.generate_only:
            errcon.print(f"generated probes at {probes_path}", style="grey50")
            return 0

        for item in items:
            for prompt in (*item.train_prompts, item.prompt):
                ensure_fits(
                    teacher,
                    teacher_tokenizer,
                    [
                        {"role": "system", "content": teacher_context},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens,
                    "teacher document prompt",
                    enable_thinking,
                )

        from distil.distill import run_continual

        def report(row):
            errcon.print(
                f"eval {row['iteration']}: mastered={row['mastered']}/{row['total']}  "
                f"active={len(row['active_item_ids'])}  changed={row['canary_changed']}",
                style="grey50",
            )

        result = run_continual(
            student,
            student_tokenizer,
            items,
            canaries,
            teacher_context,
            output,
            model_name=args.model,
            generator_model=generator_name,
            student_system=student_system,
            teacher_model=teacher,
            teacher_tokenizer=teacher_tokenizer,
            from_adapter=args.from_adapter,
            iters=args.iters,
            eval_every=args.eval_every if args.eval_every is not None else 5,
            learned_threshold=(
                args.learned_threshold if args.learned_threshold is not None else 0.5
            ),
            learning_rate=(
                args.learning_rate if args.learning_rate is not None else 1e-6
            ),
            lora_rank=args.lora_rank if args.lora_rank is not None else 16,
            max_tokens=max_tokens,
            temperature=args.temperature if args.temperature is not None else 1.0,
            seed=seed,
            enable_thinking=enable_thinking,
            speculative_top_k=(
                args.speculative_top_k if args.speculative_top_k is not None else 5
            ),
            kl_direction=args.kl_direction or "forward",
            source=source,
            probes_path=probes_path,
            canaries_path=args.canaries,
            report=report,
            warn=lambda message: errcon.print(f"[yellow]warning:[/yellow] {message}"),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        return fail(str(error))

    errcon.print(
        f"selected step {result.selected_step} ({result.stop_reason}) at {result.adapter_file}",
        style="grey50",
    )
    return 0 if result.success else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    force = True if args.color == "always" else None
    console = Console(
        force_terminal=force,
        no_color=args.color == "never",
        color_system=None if args.color == "never" else "truecolor",
        width=args.width,
        height=25 if args.width is not None else None,
    )
    errcon = Console(stderr=True)

    def fail(message: str) -> int:
        errcon.print(f"[red]error:[/red] {message}")
        return 2

    teacher_mode = bool(
        args.teacher_model or args.teacher_system is not None or args.teacher_prompt is not None
    )
    training = args.train is not None
    progress = args.progress is not None
    standalone_completion = args.completion is not None and not progress
    if sum([bool(args.rollout), standalone_completion, args.sample, training, progress]) != 1:
        return fail(
            "provide exactly one of --completion, --rollout, --sample, --train or --progress"
        )
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
        (args.checkpoint_every, "--checkpoint-every"),
        (args.canaries, "--canaries"),
        (args.canary_max_tokens, "--canary-max-tokens"),
        (args.contamination_regex, "--contamination-regex"),
        (args.speculative_top_k, "--speculative-top-k"),
        (args.kl_direction, "--kl-direction"),
        (args.from_adapter, "--from-adapter"),
        (args.generator_model, "--generator-model"),
        (args.questions_per_fact, "--questions-per-fact"),
        (True if args.generate_only else None, "--generate-only"),
        (args.eval_every, "--eval-every"),
        (args.learned_threshold, "--learned-threshold"),
        (args.adapter, "--adapter"),
    ):
        if value is not None and not training:
            return fail(f"{name} only combines with --train")
    if args.probes is not None and not (progress or training):
        return fail("--probes only combines with --progress or --train")
    if training:
        output = args.adapter or args.adapter_path
        if not output:
            return fail("--train requires --adapter")
        if args.adapter and args.adapter_path and args.adapter != args.adapter_path:
            return fail("use only one of --adapter or --adapter-path for training output")
        if args.teacher_system is not None:
            return fail("--teacher-system does not combine with --train; --train is the context")
        if args.teacher_prompt is not None:
            return fail("--teacher-prompt does not combine with --train")
        if args.prompt is not None:
            return fail("--prompt does not combine with --train; put prompts in the training file")
        if args.raw:
            return fail("--raw does not combine with --train")
        if args.metric or args.stats or args.worst or args.hide_context or args.json:
            return fail("scoring output flags do not combine with --train")
        if args.save_rollout is not None:
            return fail("--save-rollout does not combine with continual --train")
        if not args.generate_only and args.canaries is None:
            return fail("--train requires --canaries")
        if args.generate_only and args.probes is not None:
            return fail("--generate-only does not combine with --probes")
        if args.probes is not None and (
            args.generator_model is not None or args.questions_per_fact is not None
        ):
            return fail("--generator-model and --questions-per-fact do not combine with --probes")
        if args.from_adapter is not None and args.lora_rank is not None:
            return fail("--lora-rank does not combine with --from-adapter")
        if args.from_adapter is not None:
            try:
                if Path(args.from_adapter).resolve() == Path(output).resolve():
                    return fail("--from-adapter must differ from --adapter")
            except OSError:
                pass
        if args.iters is not None and args.iters <= 0:
            return fail("--iters must be positive")
        if args.learning_rate is not None and args.learning_rate <= 0:
            return fail("--learning-rate must be positive")
        if args.lora_rank is not None and args.lora_rank <= 0:
            return fail("--lora-rank must be positive")
        if args.checkpoint_every is not None and args.checkpoint_every <= 0:
            return fail("--checkpoint-every must be positive")
        if args.canary_max_tokens is not None and args.canary_max_tokens <= 0:
            return fail("--canary-max-tokens must be positive")
        if args.speculative_top_k is not None and args.speculative_top_k <= 0:
            return fail("--speculative-top-k must be positive")
        if args.questions_per_fact is not None and args.questions_per_fact < 2:
            return fail("--questions-per-fact must be at least 2")
        if args.eval_every is not None and args.eval_every <= 0:
            return fail("--eval-every must be positive")
        if args.learned_threshold is not None and not 0 < args.learned_threshold <= 1:
            return fail("--learned-threshold must be in (0, 1]")
        if args.checkpoint_every is not None:
            return fail("--checkpoint-every was replaced by --eval-every")
        if args.canary_max_tokens is not None or args.contamination_regex is not None:
            return fail("--canary-max-tokens and --contamination-regex are not used in continual mode")
    if progress:
        target_progress = args.completion is not None
        if target_progress == (args.probes is not None):
            return fail("--progress requires exactly one of --completion or --probes")
        if args.adapter_path:
            return fail("--adapter-path does not combine with --progress")
        if teacher_mode or args.raw:
            return fail("teacher and raw modes do not combine with --progress")
        if args.stats or args.worst or args.hide_context:
            return fail("--stats, --worst and --hide-context do not combine with --progress")
        if args.probes is not None:
            if args.system or args.prompt is not None:
                return fail("--progress with --probes takes context from each probe")
            if args.metric:
                return fail("--metric only combines with single-completion --progress")
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

    if training:
        return _continual_cli(args, errcon, fail)

    meta = {}
    if progress:
        from distil.progress import Probe, load_probes

        if target_progress:
            completion = _read_input(args.completion)
            prompt = _read_input(args.prompt) or ""
            system = _read_input(args.system)
            if not completion:
                return fail("--completion must not be empty")
            probes = [Probe("target", prompt, completion, system)]
        else:
            try:
                probes = load_probes(args.probes)
            except (FileNotFoundError, ValueError) as e:
                return fail(str(e))
            completion = None
            prompt = ""
            system = None
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

    if progress:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()

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

    if progress:
        from distil.progress import progress_to_json, score_progress

        try:
            progress_result = score_progress(
                model,
                tokenizer,
                args.progress,
                probes,
                enable_thinking=enable_thinking,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
            return fail(str(e))
        if args.json:
            print(json.dumps(progress_to_json(progress_result), ensure_ascii=False, indent=2))
        elif target_progress:
            render_progress_tokens(progress_result, console, metric=args.metric or "prob")
        else:
            render_progress(progress_result, console)
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
            template_kwargs = (
                {} if enable_thinking is None else {"enable_thinking": enable_thinking}
            )
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
