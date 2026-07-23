"""Score fixed prompt/completion probes across saved adapter checkpoints."""

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

from distil.scoring import score_text, score_text_pair
from distil.templating import apply_teacher_context, build_raw


_STEP_RE = re.compile(r"step-(\d+)$")


@dataclass(frozen=True)
class Probe:
    name: str
    prompt: str
    completion: str
    system: str | None = None


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


@dataclass
class ProgressResult:
    checkpoint_root: str
    steps: list[int]
    probes: list[Probe]
    series: dict[str, list[ProbePoint]]
    teacher_model: str | None = None


def load_probes(path: str | Path) -> list[Probe]:
    """Load one JSON probe per non-empty line."""
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
            probe = Probe(
                name=row["name"],
                prompt=row["prompt"],
                completion=row["completion"],
                system=row.get("system"),
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid probe on line {line_number}: {error}") from error
        required = (probe.name, probe.prompt, probe.completion)
        if not all(isinstance(value, str) and value for value in required):
            raise ValueError(
                f"invalid probe on line {line_number}: fields must be non-empty strings"
            )
        if probe.system is not None and not isinstance(probe.system, str):
            raise ValueError(f"invalid probe on line {line_number}: system must be a string")
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
        if match and (child / "adapters.safetensors").is_file():
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
) -> ProbePoint:
    system = probe.system if probe.system is not None else default_system
    messages = [] if system is None else [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": probe.prompt})
    template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    # Score only the supplied completion, not end-of-turn template tokens.
    built = build_raw(prompt_text, probe.completion)
    if teacher_model is None:
        result = score_text(model, tokenizer, built.text, built.spans, top_k=top_k)
    else:
        teacher_tokenizer = teacher_tokenizer or tokenizer
        teacher_messages = apply_teacher_context(
            messages, system=teacher_system, prompt=teacher_prompt
        )
        teacher_prefix = teacher_tokenizer.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        result = score_text_pair(
            model,
            tokenizer,
            built,
            teacher_model,
            teacher_tokenizer,
            build_raw(teacher_prefix, probe.completion),
            top_k=top_k,
            teacher_without_adapter=teacher_without_adapter,
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
        mean_teacher_logprob=(result.mean_teacher_logprob if teacher_model is not None else None),
        mean_kl=result.mean_kl if teacher_model is not None else None,
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
) -> ProgressResult:
    """Score the base model and every checkpoint on each fixed probe."""
    checkpoints = find_checkpoints(checkpoint_root)
    has_step_zero = bool(checkpoints and checkpoints[0][0] == 0)
    steps = [step for step, _ in checkpoints]
    if not has_step_zero:
        steps.insert(0, 0)
    series = {probe.name: [] for probe in probes}

    def score_step(step: int) -> None:
        model.eval()
        for probe in probes:
            series[probe.name].append(
                score_probe(
                    model,
                    tokenizer,
                    probe,
                    step,
                    enable_thinking,
                    top_k=top_k,
                    default_system=default_system,
                    teacher_model=teacher_model,
                    teacher_tokenizer=teacher_tokenizer,
                    teacher_system=teacher_system,
                    teacher_prompt=teacher_prompt,
                    teacher_without_adapter=teacher_model is model,
                )
            )

    if not has_step_zero:
        score_step(0)
    for step, checkpoint in checkpoints:
        try:
            load_adapters(model, str(checkpoint))
            score_step(step)
        finally:
            remove_lora_layers(model)
            model.eval()

    return ProgressResult(
        checkpoint_root=str(checkpoint_root),
        steps=steps,
        probes=probes,
        series=series,
        teacher_model=teacher_name,
    )


def progress_to_json(result: ProgressResult) -> dict:
    def point_payload(point: ProbePoint) -> dict:
        return {key: value for key, value in asdict(point).items() if value is not None}

    return {
        "checkpoint_root": result.checkpoint_root,
        **({"teacher_model": result.teacher_model} if result.teacher_model else {}),
        "steps": result.steps,
        "probes": [asdict(probe) for probe in result.probes],
        "series": {
            name: [point_payload(point) for point in points]
            for name, points in result.series.items()
        },
    }
