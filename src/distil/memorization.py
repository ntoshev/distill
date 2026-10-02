"""Text-first source windows, exact pass scheduling, and paired recall controls."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from time import perf_counter
import mlx.core as mx
from distil.cartridge import Cartridge, RECIPE, canonical, digest, model_identity
from distil.cartridge_training import CartridgeTrainingSession
from distil.cartridge_checkpoint import (
    commit_checkpoint,
    copy_export,
    latest_checkpoint,
    recover_logs,
    validate_resume_config,
    write_json_atomic,
)
from distil.scoring import tokenize_with_offsets
from distil.sample import sample_text, sample_completion
from distil.templating import prompt_messages

MAX_SOURCE_TOKENS = 32768


@dataclass(frozen=True)
class SourceWindow:
    name: str
    document_hash: str
    key_span: tuple[int, int]
    target_span: tuple[int, int]
    key: str
    continuation: str
    prompt: str
    ids: list[int]
    target_positions: list[int]
    target_ids: list[int]
    token_checksum: str
    recipe_version: int = 1


@dataclass
class CompiledWindows:
    windows: list[SourceWindow]
    shifted: list[SourceWindow]
    source_tokens: int
    uncovered_regions: list[tuple[int, int]]
    partial_windows: int
    skipped: list[dict]


def compile_windows(
    tokenizer,
    document: str,
    *,
    key_tokens=32,
    target_tokens=128,
    stride=32,
    max_source_tokens=MAX_SOURCE_TOKENS,
    recipe=None,
):
    if not isinstance(document, str) or not document.strip():
        raise ValueError("source document is empty")
    if min(key_tokens, target_tokens, stride) <= 0:
        raise ValueError("source window sizes must be positive")
    source_ids, offsets = tokenize_with_offsets(tokenizer, document)
    if len(source_ids) > max_source_tokens:
        raise ValueError(
            f"oversized source: {len(source_ids)} tokens exceeds {max_source_tokens}"
        )
    recipe = recipe or RECIPE
    if recipe != RECIPE:
        raise ValueError("unsupported source-window recipe")
    skipped = []
    partial = 0

    def compile_at(start, label):
        nonlocal partial
        end = min(len(source_ids), start + target_tokens)
        key_start, target_start, target_end = (
            offsets[start - key_tokens][0],
            offsets[start][0],
            offsets[end - 1][1],
        )
        if target_start <= key_start or target_end <= target_start:
            skipped.append(
                {
                    "start_token": start,
                    "reason": "empty or overlapping token boundaries",
                    "panel": label,
                }
            )
            return None
        key = document[key_start:target_start]
        continuation = document[target_start:target_end]
        prompt = recipe["prefix"] + key + recipe["suffix"]
        ids, local_offsets = tokenize_with_offsets(tokenizer, prompt + continuation)
        prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
        positions = [
            i
            for i, (s, e) in enumerate(local_offsets)
            if s >= len(prompt) and e > len(prompt)
        ]
        # The raw prompt must be a stable token prefix, and the continuation must
        # match the source trajectory exactly. Never silently score a crossing token.
        if (
            not prefix_ids
            or not positions
            or positions[0] != len(prefix_ids)
            or ids[: len(prefix_ids)] != prefix_ids
            or ids[len(prefix_ids) :] != source_ids[start:end]
            or tokenizer.decode(ids) != prompt + continuation
        ):
            skipped.append(
                {
                    "start_token": start,
                    "reason": "unstable token boundary",
                    "panel": label,
                }
            )
            return None
        partial += int(label == "practiced" and end - start < target_tokens)
        return SourceWindow(
            f"{label}-{start:06d}",
            digest(document),
            (key_start, target_start),
            (target_start, target_end),
            key,
            continuation,
            prompt,
            ids,
            positions,
            ids[len(prefix_ids) :],
            digest(canonical(ids)),
        )

    windows = [
        w
        for start in range(key_tokens, len(source_ids), stride)
        if (w := compile_at(start, "practiced"))
    ]
    practiced_starts = {w.target_span[0] for w in windows}
    shifted = [
        w
        for start in range(key_tokens + max(1, stride // 2), len(source_ids), stride)
        if (w := compile_at(start, "shifted"))
        and w.target_span[0] not in practiced_starts
    ]
    if not windows:
        raise ValueError(
            "source has no valid continuation windows; increase its length or change window boundaries"
        )
    covered = sorted(w.target_span for w in windows)
    uncovered = []
    cursor = 0
    for start, end in covered:
        if start > cursor:
            uncovered.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(document):
        uncovered.append((cursor, len(document)))
    return CompiledWindows(
        windows, shifted, len(source_ids), uncovered, partial, skipped
    )


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


class WindowSchedule:
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError("window schedule must not be empty")
        self.size = size
        self.rng = random.Random(seed)
        self.order = list(range(size))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.pass_index = 0

    @property
    def completed_passes(self):
        return self.pass_index + self.cursor / self.size

    def next_batch(self, size):
        if size <= 0:
            raise ValueError("batch size must be positive")
        if self.cursor == self.size:
            self.pass_index += 1
            self.cursor = 0
            self.order = list(range(self.size))
            self.rng.shuffle(self.order)
        batch = self.order[self.cursor : min(self.size, self.cursor + size)]
        self.cursor += len(batch)
        return batch

    def state(self):
        return {
            "size": self.size,
            "order": self.order,
            "cursor": self.cursor,
            "pass_index": self.pass_index,
            "rng": self.rng.getstate(),
        }

    def restore(self, state):
        if (
            state["size"] != self.size
            or sorted(state["order"]) != list(range(self.size))
            or not 0 <= state["cursor"] <= self.size
        ):
            raise ValueError("invalid window scheduler state")
        self.order, self.cursor, self.pass_index = (
            state["order"],
            state["cursor"],
            state["pass_index"],
        )
        self.rng.setstate(_tuples(state["rng"]))


def _panel(windows, size=8):
    if len(windows) <= size:
        return windows
    return [windows[round(i * (len(windows) - 1) / (size - 1))] for i in range(size)]


def evaluate_window(
    model, tokenizer, window, *, cache_factory=None, document_prefix=""
):
    prefix = document_prefix + window.prompt
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids = prefix_ids + window.target_ids
    from distil.prefill import select_logits

    logits = select_logits(
        model,
        ids,
        list(range(len(prefix_ids) - 1, len(ids) - 1)),
        step_size=128,
        cache_factory=cache_factory,
    )
    chosen = mx.take_along_axis(logits, mx.array(window.target_ids)[:, None], axis=-1)[
        :, 0
    ]
    nll = (mx.logsumexp(logits, axis=-1) - chosen).mean().item()
    sample = sample_text(
        model,
        tokenizer,
        prefix,
        max_tokens=len(window.target_ids),
        temperature=0,
        cache_factory=cache_factory,
    )
    actual = sample.token_ids
    accuracy = sum(a == b for a, b in zip(actual, window.target_ids)) / len(
        window.target_ids
    )
    first_error = next(
        (
            i
            for i, t in enumerate(window.target_ids)
            if i >= len(actual) or actual[i] != t
        ),
        None,
    )
    return {
        "nll": nll,
        "perplexity": math.exp(min(700, nll)),
        "exact_token_match": actual == window.target_ids,
        "positional_token_accuracy": accuracy,
        "first_error": first_error,
        "literal_text": sample.completion_text,
        "generated_ids": actual,
        "target_ids": window.target_ids,
        "finish_reason": sample.finish_reason,
    }


def evaluate_memory(model, tokenizer, compiled, initial, trained, document):
    rng = list(mx.random.state)
    started = perf_counter()
    rows = []
    try:
        for kind, windows in [
            ("practiced", _panel(compiled.windows)),
            ("shifted", _panel(compiled.shifted)),
        ]:
            for window in windows:
                conditions = {}
                for label, factory, prefix in [
                    ("key_only", None, ""),
                    ("full_document", None, document + "\n\n"),
                    ("initialized", initial.new_cache, ""),
                    ("trained", trained.new_cache, ""),
                ]:
                    conditions[label] = evaluate_window(
                        model,
                        tokenizer,
                        window,
                        cache_factory=factory,
                        document_prefix=prefix,
                    )
                rows.append(
                    {"panel": kind, "name": window.name, "conditions": conditions}
                )
    finally:
        mx.random.state = rng
    return {"windows": rows, "evaluation_seconds": perf_counter() - started}


def _canary_answers(model, tokenizer, canaries, memory, max_tokens, thinking):
    rng = list(mx.random.state)
    try:
        return {
            p.name: sample_completion(
                model,
                tokenizer,
                prompt_messages(p.prompt, p.system),
                max_tokens=max_tokens,
                temperature=0,
                seed=0,
                enable_thinking=thinking,
                cache_factory=memory.new_cache if memory else None,
            ).content.strip()
            for p in canaries
        }
    finally:
        mx.random.state = rng


def _write_records(path, records):
    path.write_text(
        "".join(
            json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in records
        )
    )


def load_memory_keys(path, *, document_hash=None):
    windows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        window = SourceWindow(**data)
        if (
            window.recipe_version != 1
            or window.token_checksum != digest(canonical(window.ids))
            or not window.target_positions
            or window.ids[window.target_positions[0] :] != window.target_ids
            or window.target_positions
            != list(range(window.target_positions[0], len(window.ids)))
            or (document_hash is not None and window.document_hash != document_hash)
        ):
            raise ValueError("invalid recipe-aware source-window record")
        windows.append(window)
    if not windows or len({w.name for w in windows}) != len(windows):
        raise ValueError("memory key records must be nonempty with unique names")
    return windows


def run_memorization(
    model,
    tokenizer,
    document,
    output,
    *,
    model_name,
    cartridge=None,
    resume=False,
    identity=None,
    positions=128,
    passes=None,
    iters=None,
    key_tokens=32,
    target_tokens=128,
    stride=32,
    learning_rate=0.005,
    batch_size=1,
    eval_every=5,
    seed=0,
    objective="ntp",
    canaries=None,
    canary_max_tokens=32,
    enable_thinking=None,
    from_cartridge=None,
    report=None,
):
    if not resume and passes is None and iters is None:
        passes = 4
    root = Path(output)
    canaries = list(canaries or [])
    if passes is not None and iters is not None:
        raise ValueError("use only one of --passes or --iters")
    if (
        (passes is not None and passes <= 0)
        or (iters is not None and iters <= 0)
        or batch_size <= 0
        or eval_every <= 0
    ):
        raise ValueError(
            "training budgets, batch size and evaluation interval must be positive"
        )
    if resume:
        config = json.loads((root / "run_config.json").read_text())
        if (
            config.get("recipe") != "memorization"
            or config.get("backend") != "cartridge"
        ):
            raise ValueError("resume requires an integrated cartridge memorization run")
        selection = (
            json.loads((root / "selection.json").read_text())
            if (root / "selection.json").exists()
            else {}
        )
        if selection.get("reason") == "locality":
            raise ValueError("cannot resume a run stopped for contamination")
        old_budget = (
            config["iters"] if config["iters"] is not None else config["passes"]
        )
        new_budget = iters if config["iters"] is not None else passes
        if new_budget is None or new_budget <= old_budget:
            raise ValueError("resume requires an increased total budget")
        if (config["iters"] is None) != (iters is None):
            raise ValueError("resume cannot change budget units")
        checkpoint = latest_checkpoint(root)
        document = (root / "generated/source.txt").read_bytes().decode("utf-8")
        if digest(document) != config["source_hash"]:
            raise ValueError("source snapshot hash mismatch")
        memory = Cartridge.load(
            checkpoint,
            model,
            tokenizer,
            model_name=model_name,
            identity=identity or config["identity"],
        )
        if (identity or model_identity(model, tokenizer, model_name)) != config[
            "identity"
        ]:
            raise ValueError("resume model identity mismatch")
        config["iters" if iters is not None else "passes"] = new_budget
        learning_rate, batch_size, eval_every, seed, objective = [
            config[k]
            for k in ("learning_rate", "batch_size", "eval_every", "seed", "objective")
        ]
        key_tokens, target_tokens, stride = [
            config[k] for k in ("key_tokens", "target_tokens", "stride")
        ]
        canaries = []
        if (root / "generated/canaries.jsonl").exists():
            from distil.progress import load_probes

            canaries = load_probes(root / "generated/canaries.jsonl")
        canary_max_tokens, enable_thinking = (
            config["canary_max_tokens"],
            config["enable_thinking"],
        )
        passes, iters = config["passes"], config["iters"]
    else:
        if root.exists():
            raise ValueError(f"refusing existing cartridge output: {root}")
        identity = identity or model_identity(model, tokenizer, model_name)
        compiled = compile_windows(
            tokenizer,
            document,
            key_tokens=key_tokens,
            target_tokens=target_tokens,
            stride=stride,
        )
        max_context = model.args.max_position_embeddings
        if (
            len(tokenizer.encode(document + "\n\n", add_special_tokens=False))
            + key_tokens
            + target_tokens
            > max_context
        ):
            raise ValueError("source document teacher exceeds model context limit")
        memory = cartridge or Cartridge.initialize(
            model,
            tokenizer,
            document,
            positions,
            identity=identity,
            metadata={
                "source_hash": digest(document),
                "prompt_recipe": RECIPE,
                "recipe": "memorization",
            },
        )
        if memory.model_identity != identity:
            raise ValueError("starting cartridge identity mismatch")
        memory = Cartridge(
            memory.tensors,
            identity,
            metadata={
                **memory.metadata,
                "recipe": "memorization",
                "source_hash": digest(document),
                "prompt_recipe": RECIPE,
                "lineage": str(from_cartridge) if from_cartridge else None,
            },
        )
        if memory.positions + key_tokens + target_tokens > max_context:
            raise ValueError("cartridge prompt exceeds model context limit")
        root.mkdir(parents=True)
        generated = root / "generated"
        generated.mkdir()
        (generated / "source.txt").write_text(document, encoding="utf-8")
        if canaries:
            (generated / "canaries.jsonl").write_text(
                "".join(
                    json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in canaries
                )
            )
        config = {
            "schema_version": 2,
            "backend": "cartridge",
            "recipe": "memorization",
            "model": model_name,
            "identity": identity,
            "source_hash": digest(document),
            "source_tokens": compiled.source_tokens,
            "positions": memory.positions,
            "passes": passes,
            "iters": iters,
            "key_tokens": key_tokens,
            "target_tokens": target_tokens,
            "stride": stride,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "eval_every": eval_every,
            "seed": seed,
            "objective": objective,
            "canary_max_tokens": canary_max_tokens,
            "enable_thinking": enable_thinking,
            "prompt_recipe": RECIPE,
            "reduction": "token-weighted",
            "warmup": 5,
            "gradient_clip": 1.0,
            "logit_chunk_size": 32,
            "from_cartridge": str(from_cartridge) if from_cartridge else None,
            "teacher_identity": identity,
            "selection_policy": "final_fixed_budget",
            "panel_size": 8,
        }
        _write_records(generated / "windows.jsonl", compiled.windows)
        _write_records(generated / "practiced_keys.jsonl", _panel(compiled.windows))
        _write_records(generated / "shifted_keys.jsonl", _panel(compiled.shifted))
        write_json_atomic(
            generated / "source_manifest.json",
            {
                "source_hash": digest(document),
                "source_tokens": compiled.source_tokens,
                "uncovered_regions": compiled.uncovered_regions,
                "partial_windows": compiled.partial_windows,
                "skipped": compiled.skipped,
                "window_count": len(compiled.windows),
                "window_scored_tokens": sum(
                    len(w.target_ids) for w in compiled.windows
                ),
            },
        )
        (root / "training_metrics.jsonl").write_text("")
        (root / "eval_metrics.jsonl").write_text("")
        mx.random.seed(seed)
    compiled = compile_windows(
        tokenizer,
        document,
        key_tokens=key_tokens,
        target_tokens=target_tokens,
        stride=stride,
    )
    session = CartridgeTrainingSession(
        model, tokenizer, memory, learning_rate=learning_rate, objective=objective
    )
    schedule = WindowSchedule(len(compiled.windows), seed)
    if resume:
        state = session.restore(checkpoint)
        validate_resume_config(root, state)
        recover_logs(root, state["log_offsets"])
        schedule.restore(state["schedule"])
        baseline = state["canary_baselines"]
        safe_step = state["safe_step"]
        scored_tokens = state["scored_tokens"]
        initial = Cartridge.load(
            root / "checkpoints/step-0000",
            model,
            tokenizer,
            model_name=model_name,
            identity=config["identity"],
        )
    else:
        initial = memory
        baseline = _canary_answers(
            model,
            tokenizer,
            canaries,
            memory if from_cartridge else None,
            canary_max_tokens,
            enable_thinking,
        )
        safe_step = None
        scored_tokens = 0
    write_json_atomic(root / "run_config.json", config)

    def evaluation():
        nonlocal safe_step
        trained = session.inference_cartridge()
        answers = _canary_answers(
            model, tokenizer, canaries, trained, canary_max_tokens, enable_thinking
        )
        changes = [
            {
                "name": name,
                "baseline": baseline[name],
                "content": answer,
                "changed": answer != baseline[name],
            }
            for name, answer in answers.items()
        ]
        row = {
            "iteration": session.iteration,
            **evaluate_memory(model, tokenizer, compiled, initial, trained, document),
            "canaries": changes,
            "canary_changed": sum(c["changed"] for c in changes),
            "completed_window_passes": schedule.completed_passes,
            "source_equivalent_passes": scored_tokens / compiled.source_tokens,
        }
        if not row["canary_changed"]:
            safe_step = session.iteration
        with (root / "eval_metrics.jsonl").open("a") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        commit_checkpoint(
            session,
            root,
            {
                "schedule": schedule.state(),
                "canary_baselines": baseline,
                "safe_step": safe_step,
                "scored_tokens": scored_tokens,
                "source_hash": config["source_hash"],
            },
        )
        if report:
            report(row)
        return row["canary_changed"] == 0

    reason = "budget"
    safe = True if resume else evaluation()
    if not safe:
        reason = "locality"
    else:
        while (
            session.iteration < iters
            if iters is not None
            else schedule.completed_passes < passes
        ):
            batch = [compiled.windows[i] for i in schedule.next_batch(batch_size)]
            row = session.step_windows(batch, teacher_prefix=document + "\n\n")
            scored_tokens += row["num_tokens"]
            row.update(
                completed_window_passes=schedule.completed_passes,
                source_equivalent_passes=scored_tokens / compiled.source_tokens,
            )
            with (root / "training_metrics.jsonl").open("a") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            final = (
                session.iteration >= iters
                if iters is not None
                else schedule.completed_passes >= passes
            )
            if report:
                report(row)
            if session.iteration % eval_every == 0 or final:
                if not evaluation():
                    reason = "locality"
                    break
    export = None
    if safe_step is not None:
        export = copy_export(root / f"checkpoints/step-{safe_step:04d}", root)
    selection = {
        "stopped": "completed" if reason == "budget" else "aborted",
        "reason": reason,
        "selected_step": safe_step,
        "success": reason == "budget",
        "safe": safe_step is not None,
        "completed_updates": session.iteration,
        "completed_window_passes": schedule.completed_passes,
        "source_equivalent_passes": scored_tokens / compiled.source_tokens,
        "budget_complete": reason == "budget",
        "tensor_bytes": memory.nbytes,
        "prefix_positions": memory.positions,
        "source_tokens": compiled.source_tokens,
        "trainable_parameters": memory.trainable_count,
        "peak_memory_bytes": mx.get_peak_memory(),
        "cartridge_file": str(export) if export else None,
    }
    write_json_atomic(root / "selection.json", selection)
    return selection
