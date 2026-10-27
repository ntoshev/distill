"""Command-line interface."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from rich.console import Console

from distill.cli_config import (
    DEFAULT_JUDGE_MODEL,  # noqa: F401 - public compatibility export
    DEFAULT_TEMPERATURE,  # noqa: F401 - public compatibility export
    build_parser,
    resolve_config,
)
from distill.render import (
    RenderOptions,
    render,
    render_checkpoint_grid,
    CheckpointDisplay,
    to_json_dict,
)
from distill.rollout import load_rollout, save_rollout
from distill.sample import (
    sample_completion,
    sample_speculative_completion,
    sample_speculative_text,
    sample_text,
)
from distill.scoring import score_text, score_text_pair, score_generated
from distill.templating import (
    apply_teacher_context,
    build_raw,
    build_rollout,
    prompt_messages,
    render_generation_prompt,
)


def _read_input(value: str | None) -> str | None:
    """Inline string, @path to read a file, or "-" for stdin."""
    if value is None:
        return None
    if value == "-":
        return sys.stdin.read()
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8", newline="") as f:
            return f.read()
    return value


def _rollout_progress_probe(rollout, index: int):
    """Replay the prompt recorded in a saved training rollout exactly."""
    from distill.progress import Probe

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
            if message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
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
        raise ValueError(
            f"invalid progress run config: {config_path}: {error}"
        ) from error
    if config.get("backend") == "cartridge":
        snapshot = root / "generated/source.txt"
        if not snapshot.is_file():
            raise ValueError("cartridge progress run has no source snapshot")
        from distill.generation import teacher_system

        return (
            teacher_system(snapshot.read_bytes().decode("utf-8"))
            if config.get("recipe") == "facts"
            else None
        )
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
    from distill.generation import teacher_system as document_teacher_system

    return document_teacher_system(document)


def _continual_cli(args, errcon, fail) -> int:
    """Compile the document, load the starting policies, and run continual training."""
    from distill.generation import (
        ensure_fits,
        generate_facts,
        resolve_context,
        teacher_system as document_teacher_system,
        write_generation,
    )
    from distill.progress import Probe, ProbePoint, ProgressResult, load_probes
    from distill.stopping import LearningItem, answer_only_prompt, load_learning_items

    output = (
        args.cartridge
        if args.config.backend == "cartridge"
        else args.adapter or args.adapter_path
    )
    if args.config.backend == "cartridge" and Path(output).exists():
        return fail(f"refusing existing cartridge output: {output}")
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
        canaries = (
            [
                Probe(f"canary-{i}", _read_input(prompt), "", dynamic=True)
                for i, prompt in enumerate(args.canary, 1)
            ]
            if args.canary
            else load_probes(args.canaries)
            if not args.generate_only
            else []
        )
        items = load_learning_items(args.probes) if args.probes else None
        if args.eval_prompt is not None:
            prompt, heldout, completion = map(
                _read_input, (args.prompt, args.eval_prompt, args.completion)
            )
            if not all(text and text.strip() for text in (prompt, heldout, completion)):
                raise ValueError("inline training texts must not be empty")
            if prompt.strip() == heldout.strip():
                raise ValueError("--eval-prompt must differ from the training --prompt")
            items = [
                LearningItem("target", heldout, completion, (prompt,), fact=document)
            ]
        if any(not p.prompt or not p.prompt.strip() for p in canaries):
            raise ValueError("canary question must not be empty")
    except (FileNotFoundError, OSError, ValueError) as error:
        return fail(str(error))

    from mlx_lm import load

    sampling = args.config.sampling
    enable_thinking = sampling.enable_thinking
    student_system = _read_input(args.system)
    seed, max_tokens = sampling.seed, sampling.max_tokens
    questions_per_fact = (
        args.questions_per_fact if args.questions_per_fact is not None else 2
    )
    generator_name = args.generator_model or args.model
    student = student_tokenizer = None
    probes_path = Path(args.probes) if args.probes else None
    canaries_path = Path(args.canaries) if args.canaries else None
    if args.eval_prompt is not None or args.canary:
        generated_dir = Path(output) / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        if args.eval_prompt is not None:
            probes_path = generated_dir / "probes.jsonl"
            probes_path.write_text(
                "".join(json.dumps(asdict(item)) + "\n" for item in items)
            )
        if args.canary:
            canaries_path = generated_dir / "canaries.jsonl"
            canaries_path.write_text(
                "".join(json.dumps(asdict(probe)) + "\n" for probe in canaries)
            )
    display = CheckpointDisplay(
        errcon, show_special_tokens=args.show_special_tokens, enabled=not args.json
    )

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
                        generator,
                        generator_tokenizer,
                        messages,
                        limit,
                        "generator prompt",
                        thinking,
                    )
                    return sample_completion(
                        generator,
                        generator_tokenizer,
                        messages,
                        max_tokens=limit,
                        temperature=0.0,
                        seed=seed,
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
            if (
                generator_name == args.model
                and args.from_adapter is None
                and args.config.backend == "lora"
            ):
                student, student_tokenizer = generator, generator_tokenizer
            else:
                del generator, generator_tokenizer
                import mlx.core as mx

                mx.clear_cache()

        if student is None:
            if args.config.backend == "cartridge":
                revision = (
                    _recorded_cartridge_revision(
                        args.model, artifact=args.from_cartridge
                    )
                    if args.from_cartridge
                    else None
                )
                student, student_tokenizer, identity = _load_cartridge_base(
                    args.model, revision=revision
                )
            else:
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
        elif args.config.backend == "cartridge":
            from mlx_lm.utils import _download

            judge_path = _download(args.judge_model)
            judge, judge_tokenizer = load(judge_path)
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

        from distill.distill import run_continual

        backend_options = {}
        if args.config.backend == "cartridge":
            from distill.cartridge import Cartridge, digest, model_identity
            from distill.cartridge_training import CartridgeTrainingSession
            from distill.backends import CartridgePolicy
            from distill.memorization import MAX_SOURCE_TOKENS

            if (
                len(student_tokenizer.encode(document, add_special_tokens=False))
                > MAX_SOURCE_TOKENS
            ):
                raise ValueError("oversized source document")
            metadata = {"recipe": "facts", "source_hash": digest(document)}
            if args.from_cartridge:
                memory = Cartridge.load(
                    args.from_cartridge,
                    student,
                    student_tokenizer,
                    model_name=args.model,
                    identity=identity,
                )
                if (
                    args.cartridge_positions is not None
                    and args.cartridge_positions != memory.positions
                ):
                    raise ValueError(
                        "cartridge capacity must equal the starting artifact"
                    )
                memory = Cartridge(
                    memory.tensors,
                    identity,
                    metadata={
                        **memory.metadata,
                        **metadata,
                        "lineage": str(args.from_cartridge),
                    },
                )
            else:
                envelope = render_generation_prompt(
                    student_tokenizer,
                    prompt_messages("", teacher_context),
                    enable_thinking,
                )
                memory = Cartridge.initialize(
                    student,
                    student_tokenizer,
                    envelope,
                    args.config.cartridge.positions,
                    identity=identity,
                    metadata=metadata,
                )
            teacher_memory = memory if args.from_cartridge else None
            generated_dir = Path(output) / "generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            (generated_dir / "source.txt").write_text(document, encoding="utf-8")
            import shutil

            if canaries_path.resolve() != (generated_dir / "canaries.jsonl").resolve():
                shutil.copyfile(canaries_path, generated_dir / "canaries.jsonl")
            if args.probes:
                shutil.copyfile(args.probes, generated_dir / "probes.jsonl")
            if teacher_memory:
                teacher_memory.save(generated_dir / "teacher")

            def factory(model, tokenizer, teacher_system, **options):
                keep = {
                    key: options[key]
                    for key in (
                        "learning_rate",
                        "max_tokens",
                        "temperature",
                        "seed",
                        "enable_thinking",
                        "student_system",
                        "speculative_top_k",
                        "speculative_top_p",
                    )
                }
                return CartridgeTrainingSession(
                    model,
                    tokenizer,
                    memory,
                    teacher_system=teacher_system,
                    teacher_cartridge=teacher_memory,
                    **keep,
                )

            backend_options = {
                "backend": "cartridge",
                "session_factory": factory,
                "inference_policy": CartridgePolicy(
                    student, student_tokenizer, identity, memory
                )
                if args.from_cartridge
                else None,
                "artifact_config": {
                    "identity": identity,
                    "source_hash": digest(document),
                    "source_tokens": len(
                        student_tokenizer.encode(document, add_special_tokens=False)
                    ),
                    "positions": memory.positions,
                    "judge_identity": identity
                    if judge is student
                    else model_identity(
                        judge, judge_tokenizer, args.judge_model, model_path=judge_path
                    ),
                    "from_cartridge": args.from_cartridge,
                    "teacher_identity": {
                        "base": identity,
                        "cartridge_hash": digest(
                            (
                                generated_dir / "teacher/cartridge.safetensors"
                            ).read_bytes()
                        )
                        if teacher_memory
                        else None,
                        "cartridge": str(generated_dir / "teacher")
                        if teacher_memory
                        else None,
                    },
                    "probes": str(generated_dir / "probes.jsonl"),
                    "canaries": str(generated_dir / "canaries.jsonl"),
                },
            }

        def report(row):
            payload = row.get("progress")
            if payload:
                if display.result is None:
                    probes = [Probe(**p) for p in payload["probes"]]
                    display.result = ProgressResult(
                        str(output), [], probes, {p.name: [] for p in probes}
                    )
                display.result.steps.extend(payload["steps"])
                for probe in display.result.probes:
                    display.result.series[probe.name].extend(
                        ProbePoint(**p) for p in payload["series"][probe.name]
                    )
            rollout = row.get("training_rollout") or {}
            display.update(
                status=(
                    f"eval {row['iteration']}  learned={row['mastered']}/{row['total']}  "
                    f"changed={row['canary_changed']}  eval={row['evaluation_seconds']:.2f}s  "
                    + (
                        f"loss={row['loss']:.4f}  "
                        if row.get("loss") is not None
                        else ""
                    )
                    + f"teacher replacements={rollout.get('replaced_tokens', 0)}"
                )
            )

        def train_report(event):
            if event["event"] == "question":
                display.update(status=f"Training {event['batch_size']} question(s)")
                return
            timing = event["timing"]
            display.update(
                status=(
                    f"train {event['iteration']}: loss={event['loss']:.4f}  "
                    f"tokens={event['num_tokens']}  time={timing['total_seconds']:.2f}s  "
                    f"student={event.get('accepted_tokens', 0)}  "
                    f"teacher={event.get('replaced_tokens', 0)}"
                )
            )

        if not args.json:
            for probe in canaries:
                errcon.print(
                    f"{probe.name}: {probe.prompt}", style="grey50", markup=False
                )
        display.start()
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
            canaries_path=canaries_path,
            capture_progress=True,
            save_rollout_path=args.save_rollout,
            report=report,
            train_report=train_report,
            warn=lambda message: errcon.print(f"[yellow]warning:[/yellow] {message}"),
            **backend_options,
        )
    except KeyboardInterrupt:
        errcon.print(
            "interrupted; the last committed checkpoint remains resumable",
            style="yellow",
        )
        return 130
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        return fail(str(error))
    finally:
        display.stop()

    payload = {
        "success": result.success,
        "stop_reason": result.stop_reason,
        "selected_step": result.selected_step,
        (
            "cartridge_file" if args.config.backend == "cartridge" else "adapter_file"
        ): str(result.adapter_file) if result.adapter_file else None,
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


def _recorded_cartridge_revision(model_name, *, artifact=None, config=None):
    data = (
        json.loads((Path(artifact) / "cartridge.json").read_text())
        if artifact
        else config
    )
    identity = data.get("identity", {})
    if identity.get("model") != model_name:
        raise ValueError("cartridge model identity mismatch")
    revision = identity.get("revision")
    return None if revision == "in-memory" else revision


def _load_cartridge_base(model_name, revision=None):
    from mlx_lm import load
    from mlx_lm.utils import _download
    from distill.cartridge import model_identity, validate_model

    path = _download(model_name, revision=revision)
    model, tokenizer = load(path)
    validate_model(model)
    return (
        model,
        tokenizer,
        model_identity(model, tokenizer, model_name, model_path=path),
    )


def _resume_cartridge_facts(args, model, tokenizer, identity, errcon):
    from distill.cartridge import Cartridge, digest, model_identity
    from distill.cartridge_checkpoint import latest_checkpoint
    from distill.cartridge_training import CartridgeTrainingSession
    from distill.backends import CartridgePolicy
    from distill.distill import run_continual
    from distill.generation import teacher_system
    from distill.progress import load_probes
    from distill.stopping import load_learning_items
    from mlx_lm import load

    root = Path(args.resume)
    config = json.loads((root / "run_config.json").read_text())
    if identity != config["identity"]:
        raise ValueError("resume model identity mismatch")
    document = (root / "generated/source.txt").read_bytes().decode("utf-8")
    if digest(document) != config["source_hash"]:
        raise ValueError("source snapshot hash mismatch")
    checkpoint = latest_checkpoint(root)
    memory = Cartridge.load(
        checkpoint, model, tokenizer, model_name=args.model, identity=identity
    )
    teacher_memory = (
        Cartridge.load(
            root / "generated/teacher",
            model,
            tokenizer,
            model_name=args.model,
            identity=identity,
        )
        if config.get("from_cartridge")
        else None
    )

    if teacher_memory and digest(
        (root / "generated/teacher/cartridge.safetensors").read_bytes()
    ) != config["teacher_identity"].get("cartridge_hash"):
        raise ValueError("frozen starting-cartridge teacher identity mismatch")

    def factory(model, tokenizer, context, **options):
        keys = (
            "learning_rate",
            "max_tokens",
            "temperature",
            "seed",
            "enable_thinking",
            "student_system",
            "speculative_top_k",
            "speculative_top_p",
        )
        return CartridgeTrainingSession(
            model,
            tokenizer,
            memory,
            teacher_system=context,
            teacher_cartridge=teacher_memory,
            **{k: options[k] for k in keys},
        )

    judge_name = config["judge_model"]
    if judge_name == args.model:
        judge, judge_tokenizer = model, tokenizer
        judge_identity = identity
    else:
        from mlx_lm.utils import _download

        judge_revision = _recorded_cartridge_revision(
            judge_name, config={"identity": config["judge_identity"]}
        )
        judge_path = _download(judge_name, revision=judge_revision)
        judge, judge_tokenizer = load(judge_path)
        judge_identity = model_identity(
            judge, judge_tokenizer, judge_name, model_path=judge_path
        )
    if judge_identity != config.get("judge_identity"):
        raise ValueError("frozen mastery judge identity mismatch")
    keys = (
        "student_system",
        "eval_every",
        "batch_size",
        "learned_threshold",
        "mastery_patience",
        "mastery_max_tokens",
        "learning_rate",
        "max_tokens",
        "canary_max_tokens",
        "temperature",
        "seed",
        "enable_thinking",
        "speculative_top_k",
        "speculative_top_p",
        "kl_direction",
    )
    artifact_config = dict(config)
    artifact_config["iters"] = args.iters
    return run_continual(
        model,
        tokenizer,
        load_learning_items(root / "generated/probes.jsonl"),
        load_probes(root / "generated/canaries.jsonl"),
        teacher_system(document),
        root,
        model_name=args.model,
        iters=args.iters,
        judge_model=judge,
        judge_tokenizer=judge_tokenizer,
        judge_model_name=judge_name,
        session_factory=factory,
        backend="cartridge",
        resume_checkpoint=checkpoint,
        artifact_config=artifact_config,
        capture_progress=config.get("capture_progress", False),
        inference_policy=CartridgePolicy(model, tokenizer, identity, memory)
        if config.get("from_cartridge")
        else None,
        **{k: config[k] for k in keys},
        report=lambda row: errcon.print(
            f"eval {row['iteration']}: mastered={row['mastered']}/{row['total']} changed={row['canary_changed']}",
            style="grey50",
        ),
    )


def _cartridge_cli(args, console, errcon, fail):
    from distill.cartridge import Cartridge
    from distill.memorization import run_memorization
    from distill.progress import load_probes

    try:
        mode = args.config.mode
        if mode == "memorize":
            document = _read_input(args.memorize)
            if not document or not document.strip():
                raise ValueError("source document is empty")
            if Path(args.cartridge).exists():
                raise ValueError(
                    f"refusing existing cartridge output: {args.cartridge}"
                )
        elif mode == "complete":
            key = _read_input(args.complete_key)
            if not key:
                raise ValueError("completion key must not be empty")
        if mode == "complete":
            revision = _recorded_cartridge_revision(
                args.model, artifact=args.cartridge_path
            )
        elif mode == "resume":
            config = json.loads((Path(args.resume) / "run_config.json").read_text())
            revision = _recorded_cartridge_revision(args.model, config=config)
        else:
            revision = (
                _recorded_cartridge_revision(args.model, artifact=args.from_cartridge)
                if args.from_cartridge
                else None
            )
        model, tokenizer, identity = _load_cartridge_base(args.model, revision=revision)
        if mode == "complete":
            memory = Cartridge.load(
                args.cartridge_path,
                model,
                tokenizer,
                model_name=args.model,
                identity=identity,
            )
            if memory.metadata.get("recipe") != "memorization":
                raise ValueError(
                    "--complete-key requires a memorization artifact with a recorded recipe"
                )
            sample = sample_text(
                model,
                tokenizer,
                memory.render_key(key),
                cache_factory=memory.new_cache,
                **args.config.sampling.kwargs(raw=True),
            )
            result = score_generated(
                model,
                tokenizer,
                sample,
                model_name=args.model,
                top_k=args.top_k if args.top_k is not None else 3,
                cache_factory=memory.new_cache,
            )
            if args.json:
                payload = to_json_dict(result)
                payload["sample"] = {
                    "token_ids": sample.token_ids,
                    "finish_reason": sample.finish_reason,
                    "num_tokens": sample.num_tokens,
                    "content": sample.content,
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                render(
                    result,
                    console,
                    RenderOptions(
                        metric=args.metric or "logprob",
                        hide_context=args.hide_context,
                        stats=args.stats,
                        worst=args.worst,
                    ),
                )
            return 0
        if mode == "resume":
            config = json.loads((Path(args.resume) / "run_config.json").read_text())
            if config["recipe"] == "facts":
                result = _resume_cartridge_facts(
                    args, model, tokenizer, identity, errcon
                )
                payload = {
                    "success": result.success,
                    "stop_reason": result.stop_reason,
                    "selected_step": result.selected_step,
                    "cartridge_file": str(result.adapter_file)
                    if result.adapter_file
                    else None,
                    "selection_file": str(result.selection_file),
                }
            elif config["recipe"] == "memorization":
                payload = run_memorization(
                    model,
                    tokenizer,
                    None,
                    args.resume,
                    model_name=args.model,
                    identity=identity,
                    resume=True,
                    iters=args.iters,
                    passes=args.passes,
                    report=lambda row: errcon.print(
                        f"{'eval' if 'windows' in row else 'train'} {row['iteration']}",
                        style="grey50",
                    ),
                )
            else:
                raise ValueError("unsupported resume recipe")
        else:
            memory = None
            if args.from_cartridge:
                memory = Cartridge.load(
                    args.from_cartridge,
                    model,
                    tokenizer,
                    model_name=args.model,
                    identity=identity,
                )
                if (
                    args.cartridge_positions is not None
                    and args.cartridge_positions != memory.positions
                ):
                    raise ValueError(
                        "cartridge capacity must equal the starting artifact"
                    )
            options = args.config.cartridge
            training = args.config.training
            payload = run_memorization(
                model,
                tokenizer,
                document,
                args.cartridge,
                model_name=args.model,
                identity=identity,
                cartridge=memory,
                positions=options.positions,
                passes=options.passes,
                iters=args.iters,
                key_tokens=options.key_tokens,
                target_tokens=options.target_tokens,
                stride=options.stride,
                objective=options.objective,
                learning_rate=training.learning_rate,
                batch_size=training.batch_size,
                eval_every=training.eval_every,
                seed=args.config.sampling.seed,
                from_cartridge=args.from_cartridge,
                canaries=load_probes(args.canaries) if args.canaries else [],
                canary_max_tokens=training.canary_max_tokens,
                report=lambda row: errcon.print(
                    f"{'eval' if 'windows' in row else 'train'} {row['iteration']}",
                    style="grey50",
                ),
            )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            errcon.print(
                f"selected step {payload.get('selected_step')} at {payload.get('cartridge_file')}",
                style="grey50",
            )
        return 0 if payload["success"] else 1
    except KeyboardInterrupt:
        errcon.print(
            "interrupted; the last committed checkpoint remains resumable",
            style="yellow",
        )
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        return fail(str(error))


def _load_models(args):
    """Load weights only after command and input validation have succeeded."""
    from mlx_lm import load

    if args.config.backend == "cartridge":
        if args.cartridge_path:
            revision = _recorded_cartridge_revision(
                args.model, artifact=args.cartridge_path
            )
        else:
            from distill.progress import find_checkpoints

            first = find_checkpoints(args.progress)[0][1]
            revision = _recorded_cartridge_revision(args.model, artifact=first)
        model, tokenizer, identity = _load_cartridge_base(args.model, revision=revision)
        args.model_identity = identity
        if args.cartridge_path:
            from distill.cartridge import Cartridge

            memory = Cartridge.load(
                args.cartridge_path,
                model,
                tokenizer,
                model_name=args.model,
                identity=identity,
            )
            args.loaded_cartridge = memory
            args.cache_factory = memory.new_cache
    else:
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
    from distill.progress import Probe, load_probes

    inline_canaries = [
        Probe(f"canary-{i}", _read_input(prompt), "", dynamic=True)
        for i, prompt in enumerate(args.canary or [], 1)
    ]
    if args.probes is not None:
        probes = (
            load_probes(args.probes, prompt_mode=args.progress_prompt) + inline_canaries
        )
        if args.config.backend == "cartridge":
            from distill.progress import affected_canary

            canary = affected_canary(args.progress)
            if canary and canary.name not in {p.name for p in probes}:
                probes.append(canary)
        return probes, _read_input(args.system)
    if args.rollout is not None:
        index = args.index if args.index is not None else 0
        return [
            _rollout_progress_probe(load_rollout(args.rollout, index), index),
            *inline_canaries,
        ], None
    completion = None if args.config.sampled_progress else _read_input(args.completion)
    prompt = _read_input(args.prompt) or ""
    system = _read_input(args.system)
    if args.config.sampled_progress:
        return [Probe("sample", prompt, "", system), *inline_canaries], system
    if not completion:
        raise ValueError("--completion must not be empty")
    return [Probe("target", prompt, completion, system), *inline_canaries], system


def _progress_cli(args, console, errcon, fail) -> int:
    from distill.progress import (
        ProgressResult,
        progress_to_json,
        sample_progress,
        score_progress,
        recorded_progress,
    )

    if not any(
        (
            args.probes,
            args.completion is not None,
            args.sample,
            args.rollout,
            args.memory_keys,
        )
    ):
        try:
            result = recorded_progress(args.progress)
            if args.metric == "kl":
                return fail(
                    "recorded grids contain student probabilities; supply explicit probes to score KL"
                )
            if args.json:
                print(
                    json.dumps(progress_to_json(result), ensure_ascii=False, indent=2)
                )
            else:
                render_checkpoint_grid(
                    result,
                    console,
                    args.metric or "prob",
                    show_special_tokens=args.show_special_tokens,
                )
            return 0
        except (OSError, ValueError, KeyError) as error:
            return fail(str(error))

    try:
        if args.memory_keys:
            return _memory_progress_cli(args, console, errcon, fail)
        probes, system = _progress_probes(args)
    except (FileNotFoundError, IndexError, ValueError) as error:
        return fail(str(error))
    teacher_system, teacher_prompt = _teacher_inputs(args)
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    model, tokenizer, teacher_model, teacher_tokenizer, teacher_name = _load_models(
        args
    )
    config = args.config
    metric = args.metric or "prob"
    display = CheckpointDisplay(
        console,
        metric=metric,
        show_special_tokens=args.show_special_tokens,
        enabled=not args.json,
    )
    accumulating = None

    def on_start(result):
        nonlocal accumulating
        accumulating = result
        display.start()

    def on_point(probe, point):
        if accumulating is None:
            return
        count = min(len(points) for points in accumulating.series.values())
        if count:
            partial = ProgressResult(
                accumulating.checkpoint_root,
                accumulating.steps[:count],
                accumulating.probes,
                {name: points[:count] for name, points in accumulating.series.items()},
                accumulating.teacher_model,
            )
            display.update(partial, status=f"Scored checkpoint {partial.steps[-1]}")

    options = dict(
        enable_thinking=config.sampling.enable_thinking,
        top_k=args.top_k if args.top_k is not None else 0,
        default_system=system,
        teacher_model=teacher_model if config.teacher_mode else None,
        teacher_tokenizer=teacher_tokenizer if config.teacher_mode else None,
        teacher_system=teacher_system,
        teacher_prompt=teacher_prompt,
        teacher_name=teacher_name if config.teacher_mode else None,
        on_start=on_start,
        on_point=on_point,
    )
    evaluate = score_progress
    if config.sampled_progress:
        evaluate = sample_progress
        options.update(config.sampling.kwargs())
    if args.config.backend == "cartridge":
        options.update(model_name=args.model, model_identity=args.model_identity)
    try:
        result = evaluate(model, tokenizer, args.progress, probes, **options)
        display.update(result, status=f"{len(result.steps)} checkpoints evaluated")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        return fail(str(error))
    finally:
        display.stop()
    if args.json:
        print(json.dumps(progress_to_json(result), ensure_ascii=False, indent=2))

    return 0


def _memory_progress_cli(args, console, errcon, fail):
    from distill.memorization import load_memory_keys
    from distill.progress import memory_progress, progress_to_json, affected_canary

    try:
        config = json.loads((Path(args.progress) / "run_config.json").read_text())
        revision = _recorded_cartridge_revision(args.model, config=config)
        model, tokenizer, identity = _load_cartridge_base(args.model, revision=revision)
        if config.get("recipe") != "memorization" or config.get("identity") != identity:
            raise ValueError(
                "memory progress requires the matching memorization recipe and model identity"
            )
        windows = load_memory_keys(
            args.memory_keys, document_hash=config["source_hash"]
        )
        result = memory_progress(
            model,
            tokenizer,
            args.progress,
            windows if args.json else windows[:1],
            model_name=args.model,
            identity=identity,
            top_k=args.top_k or 0,
            canary=affected_canary(args.progress),
            canary_max_tokens=config["canary_max_tokens"],
            enable_thinking=config["enable_thinking"],
        )
        if args.json:
            print(json.dumps(progress_to_json(result), ensure_ascii=False, indent=2))
        else:
            render_checkpoint_grid(result, console, args.metric or "prob")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        return fail(str(error))


def _speculative_metadata(sample) -> dict:
    if sample.speculative_top_k is None and sample.speculative_top_p is None:
        return {}
    return {
        "speculative_top_k": sample.speculative_top_k,
        **(
            {"speculative_top_p": sample.speculative_top_p}
            if sample.speculative_top_p is not None
            else {}
        ),
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
            rollout = load_rollout(
                args.rollout, args.index if args.index is not None else 0
            )
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
    model, tokenizer, teacher_model, teacher_tokenizer, teacher_name = _load_models(
        args
    )
    max_tokens, temperature = sampling.max_tokens, sampling.temperature
    enable_thinking = sampling.enable_thinking
    top_k = args.top_k if args.top_k is not None else 3

    sample = None
    teacher_messages = None
    rollout_prompt = rollout_completion = None
    rollout_extra = {}
    if rollout_file:
        try:
            messages = apply_teacher_context(
                rollout.messages, system=system, prompt=prompt
            )
        except ValueError as e:
            return fail(str(e).replace("--teacher-prompt", "--prompt"))
        built = build_rollout(tokenizer, messages, enable_thinking=enable_thinking)
    elif args.sample:
        messages = (
            None
            if args.raw
            else prompt_messages(
                prompt if prompt is not None else "",
                system,
            )
        )
        prefix = prompt if args.raw else messages
        sampler = sample_text if args.raw else sample_completion
        sample_args = [model, tokenizer, prefix]
        sample_options = sampling.kwargs(raw=args.raw)
        if getattr(args, "cache_factory", None):
            sample_options["cache_factory"] = args.cache_factory
        if args.speculative_top_k is not None or args.speculative_top_p is not None:
            if args.raw:
                teacher_prefix = (
                    teacher_prompt if teacher_prompt is not None else prompt
                )
                sampler = sample_speculative_text
            else:
                try:
                    teacher_messages = apply_teacher_context(
                        messages,
                        system=teacher_system,
                        prompt=teacher_prompt,
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
                teacher_without_adapter=bool(
                    args.adapter_path and teacher_model is model
                ),
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
                "task": "distill-sample",
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
        rollout_extra = {"task": "distill-completion", "model": args.model}

    if args.save_rollout is not None:
        # Save before scoring so a later scoring failure never loses the completion.
        index = save_rollout(
            args.save_rollout, rollout_prompt, rollout_completion, extra=rollout_extra
        )
        errcon.print(
            f"saved rollout to {args.save_rollout} as index {index}", style="grey50"
        )

    cache_factory = getattr(args, "cache_factory", None)
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
                    teacher_tokenizer,
                    teacher_messages,
                    enable_thinking,
                )
                built_teacher = build_raw(teacher_prompt_text, sample.completion_text)
            else:
                built_teacher = build_rollout(
                    teacher_tokenizer, teacher_messages, enable_thinking=enable_thinking
                )
        try:
            if sample is not None and sample.token_ids is not None:
                result = score_generated(
                    model,
                    tokenizer,
                    sample,
                    model_name=args.model,
                    top_k=top_k,
                    cache_factory=cache_factory,
                    teacher_model=teacher_model,
                    teacher_tokenizer=teacher_tokenizer,
                    teacher_prompt_text=(
                        teacher_prompt
                        if args.raw and teacher_prompt is not None
                        else prompt or ""
                    )
                    if args.raw
                    else teacher_prompt_text,
                    teacher_name=teacher_name,
                    teacher_without_adapter=bool(
                        args.adapter_path and teacher_model is model
                    ),
                )
            else:
                result = score_text_pair(
                    model,
                    tokenizer,
                    built,
                    teacher_model,
                    teacher_tokenizer,
                    built_teacher,
                    model_name=args.model,
                    teacher_name=teacher_name,
                    top_k=top_k,
                    teacher_without_adapter=bool(
                        args.adapter_path and teacher_model is model
                    ),
                    **(
                        {"student_cache_factory": cache_factory}
                        if cache_factory
                        else {}
                    ),
                )
        except ValueError as e:
            errcon.print(f"[red]error:[/red] {e}")
            return 1
    elif sample is not None and sample.token_ids is not None:
        result = score_generated(
            model,
            tokenizer,
            sample,
            model_name=args.model,
            top_k=top_k,
            cache_factory=cache_factory,
        )
    else:
        result = score_text(
            model,
            tokenizer,
            built.text,
            built.spans,
            model_name=args.model,
            top_k=top_k,
            warnings=built.warnings,
            **({"cache_factory": cache_factory} if cache_factory else {}),
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
        errcon.print(
            "rollout: " + "  ".join(f"{k}={v}" for k, v in meta.items()), style="grey50"
        )
    if rollout_file and (system is not None or prompt is not None):
        overrides = [
            name
            for name, value in (("system", system), ("prompt", prompt))
            if value is not None
        ]
        errcon.print("student overrides: " + ", ".join(overrides), style="grey50")
    if teacher_mode:
        overrides = [
            name
            for name, v in (("system", teacher_system), ("prompt", teacher_prompt))
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
    errcon = Console(
        stderr=True,
        force_terminal=True if args.color == "always" else None,
        no_color=args.color == "never",
        color_system={"never": None, "auto": "auto", "always": "truecolor"}[args.color],
        width=args.width,
        height=25 if args.width is not None else None,
    )

    def fail(message: str) -> int:
        errcon.print(f"[red]error:[/red] {message}")
        return 2

    try:
        config = resolve_config(args, argv)
        if (
            config.mode == "progress"
            and not config.teacher_mode
            and args.memory_keys is None
            and any(
                (args.probes, args.completion is not None, args.sample, args.rollout)
            )
        ):
            system = _teacher_system_from_progress_run(args.progress)
            config = replace(
                config, auto_teacher_system=system, teacher_mode=system is not None
            )
        if args.metric == "kl" and not config.teacher_mode:
            return fail("--metric kl needs teacher-mode scoring (a --teacher-* flag)")
    except (OSError, ValueError) as error:
        return fail(str(error))
    args.config = config
    if config.mode in {"memorize", "resume", "complete"}:
        return _cartridge_cli(args, console, errcon, fail)
    if config.mode == "train":
        return _continual_cli(args, errcon, fail)
    try:
        if config.mode == "progress":
            return _progress_cli(args, console, errcon, fail)
        return _score_cli(args, console, errcon, fail)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        if config.backend == "cartridge":
            return fail(str(error))
        raise


if __name__ == "__main__":
    sys.exit(main())
