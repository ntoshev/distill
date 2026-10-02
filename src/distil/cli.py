"""Command-line interface."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from rich.console import Console

from distil.cli_config import (
    DEFAULT_JUDGE_MODEL, DEFAULT_TEMPERATURE, build_parser, resolve_config,
)
from distil.render import (
    RenderOptions,
    render,
    render_progress_row_point,
    render_progress_rows_header,
    render_skd_rollout,
    to_json_dict,
)
from distil.rollout import load_rollout, save_rollout
from distil.sample import (
    sample_completion,
    sample_speculative_completion,
    sample_speculative_text,
    sample_text,
)
from distil.scoring import score_text, score_text_pair
from distil.templating import (
    apply_teacher_context, build_raw, build_rollout, prompt_messages,
    render_generation_prompt,
)

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


def _rollout_progress_probe(rollout, index: int):
    """Replay the prompt recorded in a saved training rollout exactly."""
    from distil.progress import Probe

    messages = tuple(dict(message) for message in rollout.prompt)
    prompt = next(
        (
            message["content"]
            for message in reversed(messages)
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ),
        None,
    )
    if not prompt:
        raise ValueError(f"rollout row {index} has no user prompt to replay")
    completion = next(
        (
            message["content"]
            for message in reversed(rollout.completion)
            if message.get("role") == "assistant" and isinstance(message.get("content"), str)
        ),
        "",
    )
    return Probe(f"rollout-{index}", prompt, completion, messages=messages)


def _teacher_system_from_progress_run(path: str | Path) -> str | None:
    """Rebuild the training teacher context from a progress run directory."""
    root = Path(path)
    config_path = root / "run_config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid progress run config: {config_path}: {error}") from error
    source = config.get("source")
    if not isinstance(source, dict) or source.get("source") != "file":
        return None
    source_path = source.get("path")
    if not isinstance(source_path, str) or not source_path:
        return None
    document_path = Path(source_path)
    if not document_path.is_absolute():
        document_path = Path.cwd() / document_path
    document = document_path.read_text()
    from distil.generation import teacher_system as document_teacher_system

    return document_teacher_system(document)


def _continual_cli(args, errcon, fail) -> int:
    """Compile the document, load the starting policies, and run continual training."""
    from distil.generation import (
        ensure_fits,
        generate_facts,
        resolve_context,
        teacher_system as document_teacher_system,
        write_generation,
    )
    from distil.progress import load_probes
    from distil.stopping import answer_only_prompt, load_learning_items

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

    sampling = args.config.sampling
    enable_thinking = sampling.enable_thinking
    student_system = _read_input(args.system)
    seed, max_tokens = sampling.seed, sampling.max_tokens
    questions_per_fact = args.questions_per_fact if args.questions_per_fact is not None else 2
    generator_name = args.generator_model or args.model
    student = student_tokenizer = None
    probes_path = Path(args.probes) if args.probes else None

    def generated() -> int:
        if args.json:
            print(json.dumps({"generated_probes": str(probes_path)}, indent=2))
        else:
            errcon.print(f"generated probes at {probes_path}", style="grey50")
        return 0

    try:
        if items is None:
            generator, generator_tokenizer = load(generator_name)

            def generator_fn(limit: int, thinking: bool | None):
                def generate(prompt: str) -> str:
                    messages = prompt_messages(prompt)
                    ensure_fits(
                        generator, generator_tokenizer, messages, limit,
                        "generator prompt", thinking,
                    )
                    return sample_completion(
                        generator, generator_tokenizer, messages,
                        max_tokens=limit, temperature=0.0, seed=seed,
                        enable_thinking=thinking,
                    ).content
                return generate

            facts = generate_facts(
                document,
                questions_per_fact,
                generator_fn(max(512, max_tokens), enable_thinking),
                generator_fn(max(256, max_tokens), False),
            )
            probes_path = write_generation(
                Path(output) / "generated",
                document,
                source,
                facts,
                generator_model=generator_name,
                questions_per_fact=questions_per_fact,
                seed=seed,
            )
            items = load_learning_items(probes_path)
            if args.generate_only:
                return generated()
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

        if args.generate_only:
            return generated()

        if args.judge_model.casefold() == args.model.casefold():
            judge, judge_tokenizer = student, student_tokenizer
        else:
            judge, judge_tokenizer = load(args.judge_model)
        teacher_context = document_teacher_system(document)

        for item in items:
            for prompt in (*item.train_prompts, answer_only_prompt(item.prompt)):
                ensure_fits(
                    teacher,
                    teacher_tokenizer,
                    prompt_messages(prompt, teacher_context),
                    max_tokens,
                    "teacher document prompt",
                    enable_thinking,
                )

        from distil.distill import run_continual

        def report(row):
            errcon.print(
                f"eval {row['iteration']}: mastered={row['mastered']}/{row['total']}  "
                f"probability={row['probability_mastered']}/{row['total']}  "
                f"active={len(row['active_item_ids'])}  changed={row['canary_changed']}  "
                f"time={row['evaluation_seconds']:.2f}s",
                style="grey50",
            )
            if row.get("training_rollout"):
                render_skd_rollout(
                    row["training_rollout"],
                    errcon,
                    label=f"skd@{row['iteration']}",
                )

        def train_report(event):
            if event["event"] == "question":
                errcon.print(
                    f"train batch size: {event['batch_size']}",
                    style="grey50",
                )
                errcon.print(
                    f"train 1 question: {event['prompt']}",
                    style="grey50",
                    markup=False,
                )
                return
            timing = event["timing"]
            errcon.print(
                f"train {event['iteration']}: loss={event['loss']:.4f}  "
                f"tokens={event['num_tokens']}  time={timing['total_seconds']:.2f}s  "
                f"(rollout={timing['rollout_seconds']:.2f}s  "
                f"distill={timing['distill_seconds']:.2f}s  "
                f"update={timing['optimizer_seconds']:.2f}s)",
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
            judge_model=judge,
            judge_tokenizer=judge_tokenizer,
            judge_model_name=args.judge_model,
            from_adapter=args.from_adapter,
            iters=args.iters,
            **asdict(args.config.training),
            **sampling.kwargs(),
            source=source,
            probes_path=probes_path,
            canaries_path=args.canaries,
            save_rollout_path=args.save_rollout,
            report=report,
            train_report=train_report,
            warn=lambda message: errcon.print(f"[yellow]warning:[/yellow] {message}"),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        return fail(str(error))

    payload = {
        "success": result.success,
        "stop_reason": result.stop_reason,
        "selected_step": result.selected_step,
        "adapter_file": str(result.adapter_file),
        "selection_file": str(result.selection_file),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        errcon.print(
            f"selected step {result.selected_step} ({result.stop_reason}) "
            f"at {result.adapter_file}",
            style="grey50",
        )
    return 0 if result.success else 1


def _load_models(args):
    """Load weights only after command and input validation have succeeded."""
    from mlx_lm import load

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    teacher_name = args.teacher_model or args.model
    if args.config.teacher_mode and teacher_name != args.model:
        teacher_model, teacher_tokenizer = load(teacher_name)
    else:
        teacher_model, teacher_tokenizer = model, tokenizer
    return model, tokenizer, teacher_model, teacher_tokenizer, teacher_name


def _teacher_inputs(args) -> tuple[str | None, str | None]:
    system = _read_input(args.teacher_system)
    if system is None:
        system = args.config.auto_teacher_system
    return system, _read_input(args.teacher_prompt)


def _progress_probes(args):
    from distil.progress import Probe, load_probes

    if args.probes is not None:
        return load_probes(args.probes, prompt_mode=args.progress_prompt), _read_input(args.system)
    if args.rollout is not None:
        index = args.index if args.index is not None else 0
        return [_rollout_progress_probe(load_rollout(args.rollout, index), index)], None
    completion = None if args.config.sampled_progress else _read_input(args.completion)
    prompt = _read_input(args.prompt) or ""
    system = _read_input(args.system)
    if args.config.sampled_progress:
        return [Probe("sample", prompt, "", system)], system
    if not completion:
        raise ValueError("--completion must not be empty")
    return [Probe("target", prompt, completion, system)], system


def _progress_cli(args, console, errcon, fail) -> int:
    from distil.progress import progress_to_json, sample_progress, score_progress

    try:
        probes, system = _progress_probes(args)
    except (FileNotFoundError, IndexError, ValueError) as error:
        return fail(str(error))
    teacher_system, teacher_prompt = _teacher_inputs(args)
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    model, tokenizer, teacher_model, teacher_tokenizer, teacher_name = _load_models(args)
    config = args.config
    metric = args.metric or ("kl" if config.teacher_mode else "prob")
    layout = {}

    def on_start(result):
        if not args.json:
            layout.update(render_progress_rows_header(
                result, console, metric, sampled=config.sampled_progress,
            ))

    def on_point(probe, point):
        if not args.json:
            render_progress_row_point(probe, point, console, metric, layout)

    options = dict(
        enable_thinking=config.sampling.enable_thinking,
        top_k=args.top_k if args.top_k is not None else 0,
        default_system=system,
        teacher_model=teacher_model if config.teacher_mode else None,
        teacher_tokenizer=teacher_tokenizer if config.teacher_mode else None,
        teacher_system=teacher_system, teacher_prompt=teacher_prompt,
        teacher_name=teacher_name if config.teacher_mode else None,
        on_start=on_start, on_point=on_point,
    )
    evaluate = score_progress
    if config.sampled_progress:
        evaluate = sample_progress
        options.update(config.sampling.kwargs())
    try:
        result = evaluate(model, tokenizer, args.progress, probes, **options)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        return fail(str(error))
    if args.json:
        print(json.dumps(progress_to_json(result), ensure_ascii=False, indent=2))
    return 0


def _speculative_metadata(sample) -> dict:
    if sample.speculative_top_k is None and sample.speculative_top_p is None:
        return {}
    return {
        "speculative_top_k": sample.speculative_top_k,
        **({"speculative_top_p": sample.speculative_top_p} if sample.speculative_top_p is not None else {}),
        "accepted_tokens": sample.accepted_tokens,
        "replaced_tokens": sample.replaced_tokens,
    }


def _score_cli(args, console, errcon, fail) -> int:
    config = args.config
    sampling = config.sampling
    teacher_mode = config.teacher_mode
    metric = args.metric or ("kl" if teacher_mode else "logprob")
    rollout_file = args.rollout is not None
    meta = {}
    if rollout_file:
        try:
            rollout = load_rollout(args.rollout, args.index if args.index is not None else 0)
        except (FileNotFoundError, IndexError, ValueError) as error:
            return fail(str(error))
        meta = rollout.meta
        system, prompt = _read_input(args.system), _read_input(args.prompt)
    else:
        completion = _read_input(args.completion)
        prompt, system = _read_input(args.prompt), _read_input(args.system)
    teacher_system, teacher_prompt = _teacher_inputs(args)
    if args.adapter_path and not Path(args.adapter_path).exists():
        return fail(f"adapter path not found: {args.adapter_path}")
    model, tokenizer, teacher_model, teacher_tokenizer, teacher_name = _load_models(args)
    max_tokens, temperature = sampling.max_tokens, sampling.temperature
    enable_thinking = sampling.enable_thinking
    top_k = args.top_k if args.top_k is not None else 3

    sample = None
    teacher_messages = None
    rollout_prompt = rollout_completion = None
    rollout_extra = {}
    if rollout_file:
        try:
            messages = apply_teacher_context(rollout.messages, system=system, prompt=prompt)
        except ValueError as e:
            return fail(str(e).replace("--teacher-prompt", "--prompt"))
        built = build_rollout(tokenizer, messages, enable_thinking=enable_thinking)
    elif args.sample:
        messages = None if args.raw else prompt_messages(
            prompt if prompt is not None else "", system,
        )
        prefix = prompt if args.raw else messages
        sampler = sample_text if args.raw else sample_completion
        sample_args = [model, tokenizer, prefix]
        sample_options = sampling.kwargs(raw=args.raw)
        if args.speculative_top_k is not None or args.speculative_top_p is not None:
            if args.raw:
                teacher_prefix = teacher_prompt if teacher_prompt is not None else prompt
                sampler = sample_speculative_text
            else:
                try:
                    teacher_messages = apply_teacher_context(
                        messages, system=teacher_system, prompt=teacher_prompt,
                    )
                except ValueError as error:
                    return fail(str(error))
                teacher_prefix = teacher_messages
                sampler = sample_speculative_completion
            sample_args.extend([teacher_model, teacher_tokenizer, teacher_prefix])
            if args.speculative_top_p is not None:
                sample_options["top_p"] = args.speculative_top_p
            else:
                sample_options["top_k"] = args.speculative_top_k
            sample_options.update(
                teacher_without_adapter=bool(args.adapter_path and teacher_model is model),
            )
        try:
            sample = sampler(*sample_args, **sample_options)
        except ValueError as error:
            return fail(str(error))
        detail = f"finish: {sample.finish_reason}"
        if sample.speculative_top_k is not None or sample.speculative_top_p is not None:
            detail = (
                f"{sample.accepted_tokens} accepted, {sample.replaced_tokens} replaced; "
                f"teacher tokens double-underlined; {detail}"
            )
        errcon.print(f"sampled {sample.num_tokens} tokens ({detail})", style="grey50")
        # the rendered generation prompt plus the generated text is already the
        # exact scored string; no re-templating round-trip
        built = build_raw(sample.prompt_text, sample.completion_text)
        if not args.raw:
            rollout_prompt = messages
            rollout_completion = [{"role": "assistant", "content": sample.content}]
            rollout_extra = {
                "task": "distil-sample",
                "model": args.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **({"seed": args.seed} if args.seed is not None else {}),
                **_speculative_metadata(sample),
                "is_truncated": sample.finish_reason != "stop",
            }
    elif args.raw:
        built = build_raw(prompt or "", completion)
    else:
        messages = prompt_messages(prompt if prompt is not None else "", system)
        messages.append({"role": "assistant", "content": completion})
        built = build_rollout(tokenizer, messages, enable_thinking=enable_thinking)
        rollout_prompt = messages[:-1]
        rollout_completion = messages[-1:]
        rollout_extra = {"task": "distil-completion", "model": args.model}

    if args.save_rollout is not None:
        # Save before scoring so a later scoring failure never loses the completion.
        index = save_rollout(
            args.save_rollout, rollout_prompt, rollout_completion, extra=rollout_extra
        )
        errcon.print(f"saved rollout to {args.save_rollout} as index {index}", style="grey50")

    if teacher_mode:
        if args.raw:
            built_teacher = build_raw(
                teacher_prompt if teacher_prompt is not None else (prompt or ""),
                sample.completion_text if sample is not None else completion,
            )
        else:
            if teacher_messages is None:
                try:
                    teacher_messages = apply_teacher_context(
                        messages, system=teacher_system, prompt=teacher_prompt
                    )
                except ValueError as e:
                    return fail(str(e))
            if args.sample:
                teacher_prompt_text = render_generation_prompt(
                    teacher_tokenizer, teacher_messages, enable_thinking,
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
                model_name=args.model, teacher_name=teacher_name, top_k=top_k,
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
            top_k=top_k,
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
                **_speculative_metadata(sample),
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for w in result.warnings:
        errcon.print(f"[yellow]warning:[/yellow] {w}")
    if meta:
        errcon.print("rollout: " + "  ".join(f"{k}={v}" for k, v in meta.items()), style="grey50")
    if rollout_file and (system is not None or prompt is not None):
        overrides = [
            name for name, value in (("system", system), ("prompt", prompt))
            if value is not None
        ]
        errcon.print("student overrides: " + ", ".join(overrides), style="grey50")
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
            teacher_sampled=sample.teacher_sampled if sample is not None else None,
        ),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    console = Console(
        force_terminal=True if args.color == "always" else None,
        no_color=args.color == "never",
        color_system=None if args.color == "never" else "truecolor",
        width=args.width,
        height=25 if args.width is not None else None,
    )
    errcon = Console(stderr=True)

    def fail(message: str) -> int:
        errcon.print(f"[red]error:[/red] {message}")
        return 2

    try:
        config = resolve_config(args, argv)
        if config.mode == "progress" and not config.teacher_mode:
            system = _teacher_system_from_progress_run(args.progress)
            config = replace(config, auto_teacher_system=system, teacher_mode=system is not None)
        if args.metric == "kl" and not config.teacher_mode:
            return fail("--metric kl needs teacher-mode scoring (a --teacher-* flag)")
    except (OSError, ValueError) as error:
        return fail(str(error))
    args.config = config
    if config.mode == "train":
        return _continual_cli(args, errcon, fail)
    if config.mode == "progress":
        return _progress_cli(args, console, errcon, fail)
    return _score_cli(args, console, errcon, fail)


if __name__ == "__main__":
    sys.exit(main())
