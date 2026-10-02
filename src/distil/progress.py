"""Score fixed prompt/completion probes across saved adapter checkpoints."""

import json
import math
import re
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

from distil.sample import Sample, sample_completion
from distil.scoring import score_text, score_text_pair, score_generated
from distil.templating import (
    apply_teacher_context,
    build_raw,
    prompt_messages,
    render_generation_prompt,
)


_STEP_RE = re.compile(r"step-(\d+)$")


@dataclass(frozen=True)
class Probe:
    name: str
    prompt: str
    completion: str
    system: str | None = None
    # A saved rollout can carry a full multi-turn prompt.  When present, use it
    # verbatim rather than reconstructing a one-user-message conversation.
    messages: tuple[dict, ...] | None = None
    dynamic: bool = False


@dataclass(frozen=True)
class ProbePoint:
    step: int
    mean_logprob: float
    mean_probability: float
    sequence_logprob: float
    sequence_probability: float
    tokens: list[dict]
    mean_teacher_logprob: float | None = None
    mean_kl: float | None = None
    completion: str | None = None
    content: str | None = None
    finish_reason: str | None = None
    num_tokens: int | None = None
    label: str | None = None


@dataclass
class ProgressResult:
    checkpoint_root: str
    steps: list[int]
    probes: list[Probe]
    series: dict[str, list[ProbePoint]]
    teacher_model: str | None = None


def load_probes(path: str | Path, prompt_mode: str = "test") -> list[Probe]:
    """Load one JSON probe per non-empty line."""
    if prompt_mode not in {"test", "train"}:
        raise ValueError("prompt_mode must be 'test' or 'train'")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"probe file not found or not a file: {path}")

    probes = []
    names = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            name = row["name"]
            completion = row["completion"]
            system = row.get("system")
            if prompt_mode == "train":
                raw_prompts = row["train_prompts"]
                if not isinstance(raw_prompts, list) or not raw_prompts:
                    raise ValueError("train_prompts must be a non-empty list")
                row_probes = [
                    Probe(
                        name=f"{name}/train-{index}",
                        # Match the on-policy training distribution exactly.
                        # The answer-only contract is used only for held-out
                        # mastery evaluation, not for training prompts.
                        prompt=prompt,
                        completion=completion,
                        system=system,
                    )
                    for index, prompt in enumerate(raw_prompts, 1)
                ]
            else:
                row_probes = [
                    Probe(
                        name=name,
                        prompt=row["prompt"],
                        completion=completion,
                        system=system,
                    )
                ]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid probe on line {line_number}: {error}") from error
        for probe in row_probes:
            required = (probe.name, probe.prompt, probe.completion)
            if not all(isinstance(value, str) and value for value in required):
                raise ValueError(
                    f"invalid probe on line {line_number}: fields must be non-empty strings"
                )
            if probe.system is not None and not isinstance(probe.system, str):
                raise ValueError(
                    f"invalid probe on line {line_number}: system must be a string"
                )
            if probe.name in names:
                raise ValueError(f"duplicate probe name: {probe.name}")
            names.add(probe.name)
            probes.append(probe)
    if not probes:
        raise ValueError(f"probe file is empty: {path}")
    return probes


def find_checkpoints(path: str | Path) -> list[tuple[int, Path]]:
    """Find numbered adapter checkpoints under PATH or PATH/checkpoints."""
    path = Path(path)
    directory = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")

    checkpoints = []
    for child in directory.iterdir():
        match = _STEP_RE.fullmatch(child.name)
        if match and (
            (child / "adapters.safetensors").is_file()
            or (
                (child / "cartridge.json").is_file()
                and (child / "cartridge.safetensors").is_file()
            )
        ):
            checkpoints.append((int(match.group(1)), child))
    checkpoints.sort()
    if not checkpoints:
        raise ValueError(f"no step-NNNN adapter checkpoints found in {directory}")
    return checkpoints


def _token_payload(token) -> dict:
    payload = {
        "text": token.text,
        "logprob": token.logprob,
        "probability": math.exp(token.logprob),
    }
    if token.top_k:
        payload["top"] = [
            [alternative.text, alternative.logprob] for alternative in token.top_k
        ]
    if token.kl is not None:
        payload.update(teacher_logprob=token.teacher_logprob, kl=token.kl)
        if token.teacher_top_k:
            payload["teacher_top"] = [
                [alternative.text, alternative.logprob]
                for alternative in token.teacher_top_k
            ]
    return payload


def _probe_messages(probe: Probe, default_system: str | None) -> list[dict]:
    """Return the exact saved prompt when a probe originated from a rollout."""
    if probe.messages is not None:
        return [dict(message) for message in probe.messages]
    system = probe.system if probe.system is not None else default_system
    return prompt_messages(probe.prompt, system)


def score_probe(
    model,
    tokenizer,
    probe: Probe,
    step: int,
    enable_thinking: bool | None,
    top_k: int = 0,
    default_system: str | None = None,
    teacher_model=None,
    teacher_tokenizer=None,
    teacher_system: str | None = None,
    teacher_prompt: str | None = None,
    teacher_without_adapter: bool = False,
    cache_factory=None,
) -> ProbePoint:
    if probe.dynamic:
        return sample_probe(
            model,
            tokenizer,
            probe,
            step,
            enable_thinking,
            max_tokens=32,
            temperature=0,
            seed=0,
            top_k=top_k,
            default_system=default_system,
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            teacher_system=teacher_system,
            teacher_prompt=teacher_prompt,
            teacher_without_adapter=teacher_without_adapter,
            cache_factory=cache_factory,
        )
    messages = _probe_messages(probe, default_system)
    prompt_text = render_generation_prompt(tokenizer, messages, enable_thinking)
    return _completion_point(
        model,
        tokenizer,
        messages,
        prompt_text,
        probe.completion,
        step=step,
        enable_thinking=enable_thinking,
        top_k=top_k,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        teacher_system=teacher_system,
        teacher_prompt=teacher_prompt,
        teacher_without_adapter=teacher_without_adapter,
        **({"cache_factory": cache_factory} if cache_factory else {}),
    )


def _completion_point(
    model,
    tokenizer,
    messages,
    prompt_text: str,
    completion: str,
    *,
    step: int,
    enable_thinking: bool | None,
    top_k: int,
    teacher_model,
    teacher_tokenizer,
    teacher_system: str | None,
    teacher_prompt: str | None,
    teacher_without_adapter: bool,
    sample: Sample | None = None,
    cache_factory=None,
) -> ProbePoint:
    """Score the displayed string and assemble fixed or sampled progress metadata."""
    built = build_raw(prompt_text, completion)
    if teacher_model is None:
        result = (
            score_generated(
                model, tokenizer, sample, top_k=top_k, cache_factory=cache_factory
            )
            if sample is not None and sample.token_ids is not None
            else score_text(
                model,
                tokenizer,
                built.text,
                built.spans,
                top_k=top_k,
                **({"cache_factory": cache_factory} if cache_factory else {}),
            )
        )
    else:
        teacher_tokenizer = teacher_tokenizer or tokenizer
        teacher_messages = apply_teacher_context(
            messages, system=teacher_system, prompt=teacher_prompt
        )
        teacher_prefix = render_generation_prompt(
            teacher_tokenizer,
            teacher_messages,
            enable_thinking,
        )
        if sample is not None and sample.token_ids is not None:
            result = score_generated(
                model,
                tokenizer,
                sample,
                top_k=top_k,
                cache_factory=cache_factory,
                teacher_model=teacher_model,
                teacher_tokenizer=teacher_tokenizer,
                teacher_prompt_text=teacher_prefix,
                teacher_without_adapter=teacher_without_adapter,
            )
        else:
            result = score_text_pair(
                model,
                tokenizer,
                built,
                teacher_model,
                teacher_tokenizer,
                build_raw(teacher_prefix, completion),
                top_k=top_k,
                teacher_without_adapter=teacher_without_adapter,
                **({"student_cache_factory": cache_factory} if cache_factory else {}),
            )
    scored = result.scored
    sequence_logprob = sum(token.logprob for token in scored)
    return ProbePoint(
        step=step,
        mean_logprob=result.mean_logprob,
        mean_probability=math.exp(result.mean_logprob),
        sequence_logprob=sequence_logprob,
        sequence_probability=math.exp(sequence_logprob),
        tokens=[_token_payload(token) for token in scored],
        mean_teacher_logprob=(
            result.mean_teacher_logprob if teacher_model is not None else None
        ),
        mean_kl=result.mean_kl if teacher_model is not None else None,
        completion=sample.completion_text if sample is not None else None,
        content=sample.content if sample is not None else None,
        finish_reason=sample.finish_reason if sample is not None else None,
        num_tokens=sample.num_tokens if sample is not None else None,
    )


def sample_probe(
    model,
    tokenizer,
    probe: Probe,
    step: int,
    enable_thinking: bool | None,
    *,
    max_tokens: int,
    temperature: float,
    seed: int | None = None,
    top_k: int = 0,
    default_system: str | None = None,
    teacher_model=None,
    teacher_tokenizer=None,
    teacher_system: str | None = None,
    teacher_prompt: str | None = None,
    teacher_without_adapter: bool = False,
    cache_factory=None,
) -> ProbePoint:
    messages = _probe_messages(probe, default_system)
    sample = sample_completion(
        model,
        tokenizer,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        enable_thinking=enable_thinking,
        **({"cache_factory": cache_factory} if cache_factory else {}),
    )
    return _completion_point(
        model,
        tokenizer,
        messages,
        sample.prompt_text,
        sample.completion_text,
        step=step,
        enable_thinking=enable_thinking,
        top_k=top_k,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        teacher_system=teacher_system,
        teacher_prompt=teacher_prompt,
        teacher_without_adapter=teacher_without_adapter,
        sample=sample,
        cache_factory=cache_factory,
    )


def score_progress(
    model,
    tokenizer,
    checkpoint_root: str | Path,
    probes: list[Probe],
    *,
    enable_thinking: bool | None = None,
    top_k: int = 0,
    default_system: str | None = None,
    teacher_model=None,
    teacher_tokenizer=None,
    teacher_system: str | None = None,
    teacher_prompt: str | None = None,
    teacher_name: str | None = None,
    on_start=None,
    on_point=None,
    model_name=None,
    model_identity=None,
) -> ProgressResult:
    """Score the base model and every checkpoint on each fixed probe."""
    evaluate = partial(
        score_probe,
        model,
        tokenizer,
        enable_thinking=enable_thinking,
        top_k=top_k,
        default_system=default_system,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        teacher_system=teacher_system,
        teacher_prompt=teacher_prompt,
        teacher_without_adapter=teacher_model is model,
    )
    return _run_progress(
        model,
        checkpoint_root,
        probes,
        evaluate,
        teacher_name,
        on_start,
        on_point,
        tokenizer=tokenizer,
        model_name=model_name,
        identity=model_identity,
    )


def _run_progress(
    model,
    checkpoint_root,
    probes,
    evaluate,
    teacher_name,
    on_start,
    on_point,
    *,
    tokenizer=None,
    model_name=None,
    identity=None,
) -> ProgressResult:
    """Own checkpoint ordering, adapter lifetime, and incremental result delivery."""
    checkpoints = find_checkpoints(checkpoint_root)
    has_step_zero = bool(checkpoints and checkpoints[0][0] == 0)
    cartridge_run = any((path / "cartridge.json").is_file() for _, path in checkpoints)
    steps = [step for step, _ in checkpoints]
    if cartridge_run:
        steps.insert(0, -1)
    elif not has_step_zero:
        steps.insert(0, 0)
    series = {probe.name: [] for probe in probes}
    result = ProgressResult(
        checkpoint_root=str(checkpoint_root),
        steps=steps,
        probes=probes,
        series=series,
        teacher_model=teacher_name,
    )
    if on_start is not None:
        on_start(result)

    current_factory = None
    current_label = None

    def evaluate_step(step: int) -> None:
        model.eval()
        for probe in probes:
            point = evaluate(
                probe,
                step,
                **(
                    {"cache_factory": current_factory, "teacher_without_adapter": False}
                    if current_factory
                    else {}
                ),
            )
            if current_label:
                from dataclasses import replace

                point = replace(point, label=current_label)
            series[probe.name].append(point)
            if on_point is not None:
                on_point(probe, point)

    if cartridge_run:
        current_label = "base"
        evaluate_step(-1)
    elif not has_step_zero:
        evaluate_step(0)
    transitions = checkpoint_transitions(checkpoint_root) if cartridge_run else {}
    for step, checkpoint in checkpoints:
        from distil.backends import artifact_backend

        if artifact_backend(checkpoint) == "cartridge":
            from distil.cartridge import Cartridge

            memory = Cartridge.load(
                checkpoint,
                model,
                tokenizer,
                model_name=model_name or "",
                identity=identity,
            )
            current_factory = memory.new_cache
            current_label = "initialized" if step == 0 else transitions.get(step)
            evaluate_step(step)
        else:
            current_factory = None
            current_label = None
            try:
                load_adapters(model, str(checkpoint))
                evaluate_step(step)
            finally:
                remove_lora_layers(model)
                model.eval()

    return result


def sample_progress(
    model,
    tokenizer,
    checkpoint_root: str | Path,
    probes: list[Probe],
    *,
    enable_thinking: bool | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
    top_k: int = 0,
    default_system: str | None = None,
    teacher_model=None,
    teacher_tokenizer=None,
    teacher_system: str | None = None,
    teacher_prompt: str | None = None,
    teacher_name: str | None = None,
    on_start=None,
    on_point=None,
    model_name=None,
    model_identity=None,
) -> ProgressResult:
    """Sample and then score one generated completion at each checkpoint."""
    evaluate = partial(
        sample_probe,
        model,
        tokenizer,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        top_k=top_k,
        default_system=default_system,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        teacher_system=teacher_system,
        teacher_prompt=teacher_prompt,
        teacher_without_adapter=teacher_model is model,
    )
    return _run_progress(
        model,
        checkpoint_root,
        probes,
        evaluate,
        teacher_name,
        on_start,
        on_point,
        tokenizer=tokenizer,
        model_name=model_name,
        identity=model_identity,
    )


def progress_to_json(result: ProgressResult) -> dict:
    def point_payload(point: ProbePoint) -> dict:
        return {key: value for key, value in asdict(point).items() if value is not None}

    return {
        "checkpoint_root": result.checkpoint_root,
        **({"teacher_model": result.teacher_model} if result.teacher_model else {}),
        "steps": result.steps,
        "probes": [
            {
                key: value
                for key, value in asdict(probe).items()
                if key != "messages" and (key != "dynamic" or value)
            }
            for probe in result.probes
        ],
        "series": {
            name: [point_payload(point) for point in points]
            for name, points in result.series.items()
        },
    }


def memory_progress(
    model,
    tokenizer,
    checkpoint_root,
    windows,
    *,
    model_name,
    identity,
    top_k=0,
    on_start=None,
    on_point=None,
    canary=None,
    canary_max_tokens=32,
    enable_thinking=None,
):
    """Raw source records have their own recipe; they are never chat probes."""
    from distil.cartridge import Cartridge
    from distil.sample import Sample

    checkpoints = find_checkpoints(checkpoint_root)
    probes = [Probe(w.name, w.prompt, w.continuation) for w in windows]
    if canary:
        probes.append(canary)
    transitions = checkpoint_transitions(checkpoint_root)
    result = ProgressResult(
        str(checkpoint_root),
        [-1] + [step for step, _ in checkpoints],
        probes,
        {p.name: [] for p in probes},
    )
    if on_start:
        on_start(result)
    for step, checkpoint in [(-1, None), *checkpoints]:
        memory = (
            Cartridge.load(
                checkpoint, model, tokenizer, model_name=model_name, identity=identity
            )
            if checkpoint
            else None
        )
        for probe, window in zip(probes, windows):
            if (
                tokenizer.encode(
                    window.prompt + window.continuation, add_special_tokens=False
                )
                != window.ids
            ):
                raise ValueError("memory key tokenizer/recipe mismatch")
            sample = Sample(
                window.prompt,
                window.continuation,
                window.continuation,
                None,
                len(window.target_ids),
                token_ids=window.target_ids,
            )
            score = score_generated(
                model,
                tokenizer,
                sample,
                top_k=top_k,
                cache_factory=memory.new_cache if memory else None,
            )
            sequence = sum(t.logprob for t in score.scored)
            point = ProbePoint(
                step,
                score.mean_logprob,
                math.exp(score.mean_logprob),
                sequence,
                math.exp(sequence),
                [_token_payload(t) for t in score.scored],
                label="base"
                if step == -1
                else "initialized"
                if step == 0
                else transitions.get(step),
            )
            result.series[probe.name].append(point)
            if on_point:
                on_point(probe, point)
        if canary:
            point = sample_probe(
                model,
                tokenizer,
                canary,
                step,
                enable_thinking,
                max_tokens=canary_max_tokens,
                temperature=0,
                seed=0,
                cache_factory=memory.new_cache if memory else None,
            )
            from dataclasses import replace

            point = replace(
                point,
                label="base"
                if step == -1
                else "initialized"
                if step == 0
                else transitions.get(step),
            )
            result.series[canary.name].append(point)
            if on_point:
                on_point(canary, point)
    return result


def checkpoint_transitions(root):
    path = Path(root) / "eval_metrics.jsonl"
    if not path.exists():
        return {}
    transitions = {}
    leaking = False
    learned = False
    for line in path.read_text().splitlines():
        row = json.loads(line)
        step = row["iteration"]
        changed = bool(row.get("canary_changed"))
        mastered = row.get("mastered", 0) == row.get("total", -1)
        if step and changed and not leaking:
            transitions[step] = "leak"
        elif step and not changed and leaking:
            transitions[step] = "recovered"
        elif step and mastered and not learned:
            transitions[step] = "learned"
        leaking = changed
        learned = mastered
    return transitions


def affected_canary(root):
    """Return the first recorded retention alarm, if the run has one."""
    path = Path(root) / "eval_metrics.jsonl"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        for canary in row.get("canaries", []):
            if canary.get("changed"):
                records = load_probes(Path(root) / "generated/canaries.jsonl")
                from dataclasses import replace

                return next(
                    (
                        replace(p, dynamic=True)
                        for p in records
                        if p.name == canary["name"]
                    ),
                    None,
                )
    return None
