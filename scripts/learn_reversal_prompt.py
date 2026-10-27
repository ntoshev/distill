#!/usr/bin/env python3
"""Run and audit PEZ/GCG prompt-length experiments on token-list reversal.

uv run python scripts/learn_reversal_prompt.py --phase pilot --output-dir outputs/reversal-pilot
uv run python scripts/learn_reversal_prompt.py --phase sweep --output-dir outputs/reversal-main
"""

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.utils import _download

from distill.prompt_optimization import (GCG, PEZ, SoftPrompt, NEUTRAL_INITIAL_PROMPTS, COPY_INITIAL_PROMPTS, PromptLayout, PromptObjective,
    PromptVocabulary, evaluate_prompt, evaluate_soft_prompt, reversal_splits)


MANUAL_PROMPTS = [
    "Reverse",
    "Reverse.",
    "Reverse words",
    "Reverse words.",
    "Reverse words only.",
    "Reverse the words. Output only the reversed list.",
    "Reverse the order of the space-separated words. Output only those words, separated by spaces.",
    "Return the words in reverse order. Do not add, remove, or change any word. No other text.",
]
MANUAL_PROMPTS.extend(NEUTRAL_INITIAL_PROMPTS.values())
MANUAL_PROMPTS.extend(COPY_INITIAL_PROMPTS.values())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def frozen_signature(model):
    return [(name, array) for name, array in tree_flatten(model.parameters())]


def check_frozen(model, original):
    current = frozen_signature(model)
    # MLX arrays are immutable; replacing any parameter is a weight mutation.
    return (not tree_flatten(model.trainable_parameters()) and len(current) == len(original)
            and all(n == old_n and a is old_a
                    for (n, a), (old_n, old_a) in zip(current, original, strict=True)))


def correctness_gate(objective, prompt_ids, cases):
    """Independent dense-token forward reference and real-model directional FD."""
    batch = objective.batch([prompt_ids] * len(cases), cases)
    # Use the same batch shape for autodiff, dense reference, and differences.
    loss, grad = objective.loss_and_grad(prompt_ids, cases, microbatch=len(cases))
    repeated_loss, repeated_grad = objective.loss_and_grad(prompt_ids, cases, microbatch=len(cases))
    repeat_gradient_error = mx.abs(grad - repeated_grad).max().item()
    if abs(loss - repeated_loss) > 1e-5 or repeat_gradient_error > 1e-5:
        raise AssertionError("Repeated frozen-model losses/gradients differ")
    ids, targets, rows, masks = batch
    references = []
    for training in (True, False):
        objective.model.train(training)
        logits = objective.model(ids).astype(mx.float32)
        selected = logits[mx.arange(len(cases))[:, None], rows]
        ce = mx.logsumexp(selected, axis=-1) - mx.take_along_axis(selected, targets[..., None], axis=-1)[..., 0]
        references.append(((ce * masks).sum(-1) / masks.sum(-1)).mean().item())
    dense, inferred = references
    scored = objective.score([prompt_ids], cases)[0]
    vectors = objective.trunk.embed_tokens(mx.array(prompt_ids)).astype(mx.float32)
    norm = mx.linalg.norm(grad).item()
    if norm <= 1e-8:
        raise AssertionError("Prompt gradient is zero")
    direction = grad / norm
    fd = []
    for eps in (.001, .003, .01, .03):
        with objective.differentiable():
            plus = objective.losses(vectors + eps * direction, batch).mean().item()
            minus = objective.losses(vectors - eps * direction, batch).mean().item()
        with objective.inference():
            native_plus = objective.losses(vectors + eps * direction, batch).mean().item()
            native_minus = objective.losses(vectors - eps * direction, batch).mean().item()
        derivative = (plus - minus) / (2 * eps)
        fd.append({"epsilon": eps, "finite_difference": derivative, "analytic": norm,
                   "relative_error": abs(derivative - norm) / max(norm, 1e-8),
                   "minus_loss": minus, "plus_loss": plus,
                   "native_finite_difference": (native_plus - native_minus) / (2 * eps),
                   "native_minus_loss": native_minus, "native_plus_loss": native_plus})
    result = {"gradient_loss": loss, "score_loss": scored, "dense_reference_loss": dense,
              "repeat_loss_error": abs(loss - repeated_loss), "repeat_gradient_error": repeat_gradient_error,
              "ordinary_inference_loss": inferred, "gradient_norm": norm,
              "dense_error": abs(loss - dense), "score_error": abs(inferred - scored),
              "inference_error": abs(loss - inferred), "finite_differences": fd}
    if result["dense_error"] > .015 or result["score_error"] > .015:
        raise AssertionError(f"Answer CE disagrees with dense reference: {result}")
    if min(r["relative_error"] for r in fd) > .15:
        raise AssertionError(f"Finite differences disagree with prompt gradient: {result}")
    if not any(r["minus_loss"] < loss < r["plus_loss"] for r in fd):
        raise AssertionError(f"Gradient does not identify a descent direction: {result}")
    if not any(r["native_minus_loss"] < inferred < r["native_plus_loss"] for r in fd):
        raise AssertionError(f"Gradient does not identify a native-inference descent direction: {result}")
    return result


def pez_options(args):
    return {"optimizer": args.pez_optimizer, "normalize_shadow": args.pez_normalize_shadow,
            "metric": args.pez_metric, "gradient_microbatch": args.gradient_microbatch,
            "gradient_clip": args.pez_gradient_clip}


def train_arm(objective, vocab, splits, args, method, length, seed, run_dir):
    config = {"method": method, "length": length, "seed": seed, "steps": args.steps,
              "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "top_k": args.top_k, "candidates": args.candidates, "eval_every": args.eval_every}
    config["initialization"] = args.initialization
    config["checkpoint_selection"] = args.checkpoint_selection
    if args.checkpoint_selection == "accuracy":
        config["validation_generation"] = {"max_tokens": args.max_tokens,
            "batch_size": args.generation_batch_size, "require_eos": True, "temperature": 0}
    config["pez_projection"] = "feasible_coordinate_neighbors_64" if method == "pez" else None
    if method == "pez":
        config["pez_options"] = pez_options(args)
        config["optimizer"] = {
            "adam": {"name": "Adam", "bias_correction": True, "betas": [.9, .999], "epsilon": 1e-8},
            "adafactor": {"name": "Adafactor", "relative_step": False, "scale_parameter": False,
                          "warmup_init": False, "clip_threshold": 1.0, "decay_rate": -.8},
            "sgd": {"name": "SGD", "momentum": 0., "weight_decay": 0., "dampening": 0., "nesterov": False},
        }[args.pez_optimizer]
    elif method == "soft":
        config["optimizer"] = {"name": "Adam", "bias_correction": True, "betas": [.9, .999], "epsilon": 1e-8}
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["config"] != config:
            raise ValueError(f"Existing run has different config: {run_dir}")
        print(f"reuse {run_dir.name}", flush=True)
        return result
    if (run_dir / "training.jsonl").exists():
        raise ValueError(f"Interrupted run at {run_dir}; use a fresh output directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    init_rng = random.Random(seed)
    initial = vocab.initial(length, init_rng, initialization=args.initialization)
    if method == "soft":
        arm = SoftPrompt(vocab, initial, learning_rate=args.learning_rate)
    elif method == "pez":
        arm = PEZ(vocab, initial, learning_rate=args.learning_rate, **pez_options(args))
    else:
        arm = GCG(vocab, initial, seed=seed, top_k=args.top_k,
                  candidates=args.candidates, guided=method == "gcg")
    rng = random.Random(seed + 100000)
    old_forward, old_grad = objective.forward_examples, objective.gradient_examples
    started = perf_counter()
    best_ids = list(initial)
    def score(cases):
        return (objective.score_vectors(arm.vectors, arm.ids, cases) if method == "soft"
                else objective.score([arm.ids], cases)[0])
    best_loss = score(splits["validation"])
    best_step = 0
    validation_seconds = 0.
    generation_seconds, generated_examples = 0., 0
    generation_cache = {}
    def make_checkpoint(step, loss):
        nonlocal generation_seconds, generated_examples
        checkpoint = {"step": step, "ids": list(arm.ids),
                      "prompt": objective.tokenizer.decode(arm.ids), "validation_loss": loss}
        if args.checkpoint_selection == "accuracy":
            key = tuple(arm.ids) if method != "soft" else step
            if key in generation_cache:
                metrics, rows, original_step = generation_cache[key]
                checkpoint["generation_reused_from_step"] = original_step
            else:
                if method == "soft":
                    metrics, rows = evaluate_soft_prompt(objective, arm.ids, arm.vectors,
                                                         splits["validation"], max_tokens=args.max_tokens)
                else:
                    metrics, rows = evaluate_prompt(objective.model, objective.tokenizer, objective.layout,
                        arm.ids, splits["validation"], batch_size=args.generation_batch_size, max_tokens=args.max_tokens)
                generation_cache[key] = metrics, rows, step
                generation_seconds += metrics["seconds"]
                generated_examples += len(rows)
            checkpoint["validation_accuracy"] = metrics["accuracy"]
            checkpoint["validation_correct"] = metrics["correct"]
            directory = run_dir / "checkpoint_generation"
            directory.mkdir(exist_ok=True)
            (directory / f"step-{step:04d}_outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return checkpoint
    def checkpoint_key(checkpoint):
        return ((checkpoint["validation_accuracy"], -checkpoint["validation_loss"])
                if args.checkpoint_selection == "accuracy" else (-checkpoint["validation_loss"],))
    checkpoint = make_checkpoint(0, best_loss)
    best_key = checkpoint_key(checkpoint)
    best_accuracy = checkpoint.get("validation_accuracy")
    append_jsonl(run_dir / "checkpoints.jsonl", checkpoint)
    write_json(run_dir / "best_prompt.json", checkpoint)
    if method == "soft":
        mx.save_safetensors(str(run_dir / "best_soft.safetensors"), {"vectors": arm.vectors})
    for step in range(1, args.steps + 1):
        cases = rng.sample(splits["train"], args.batch_size)
        t = perf_counter()
        stats = arm.step(objective, cases)
        stats.update({"step": step, "step_seconds": perf_counter() - t,
                      "ids": arm.ids, "prompt": objective.tokenizer.decode(arm.ids)})
        append_jsonl(run_dir / "training.jsonl", stats)
        if step % args.eval_every == 0 or step == args.steps:
            t = perf_counter()
            val_loss = score(splits["validation"])
            validation_seconds += perf_counter() - t
            checkpoint = make_checkpoint(step, val_loss)
            append_jsonl(run_dir / "checkpoints.jsonl", checkpoint)
            if checkpoint_key(checkpoint) > best_key:
                best_key = checkpoint_key(checkpoint)
                best_ids, best_loss, best_step = list(arm.ids), val_loss, step
                best_accuracy = checkpoint.get("validation_accuracy")
                write_json(run_dir / "best_prompt.json", checkpoint)
                if method == "soft":
                    mx.save_safetensors(str(run_dir / "best_soft.safetensors"), {"vectors": arm.vectors})
            print(f"{run_dir.name} step {step}/{args.steps} val CE {val_loss:.4f} "
                  f"best {best_loss:.4f} accuracy {checkpoint.get('validation_accuracy')} "
                  f"prompt {checkpoint['prompt']!r}", flush=True)
    result = {"config": config, "initial_ids": initial,
              "initial_prompt": objective.tokenizer.decode(initial), "ids": best_ids,
              "prompt": "<continuous vectors; no discrete text>" if method == "soft" else objective.tokenizer.decode(best_ids),
              "best_step": best_step, "discrete_export": method != "soft",
              "validation_loss": best_loss, "canonical": objective.layout.valid(best_ids),
              "validation_accuracy": best_accuracy,
              "forward_examples": objective.forward_examples - old_forward,
              "gradient_examples": objective.gradient_examples - old_grad,
              "training_seconds": perf_counter() - started, "validation_seconds": validation_seconds,
              "validation_generation_seconds": generation_seconds, "validation_generated_examples": generated_examples,
              "repaired_projections": getattr(arm, "rejected", 0)}
    result["train_loss"] = (objective.score_vectors(mx.load(str(run_dir / "best_soft.safetensors"))["vectors"],
                             best_ids, splits["train"]) if method == "soft"
                            else objective.score([best_ids], splits["train"])[0])
    write_json(result_path, result)
    return result


def evaluate_saved(model, tok, layout, splits, ids, directory, args, *, names=None, objective=None, vectors=None):
    config = {"prompt_ids": ids, "max_tokens": args.max_tokens,
              "generation_batch_size": args.generation_batch_size, "temperature": 0,
              "enable_thinking": False, "require_eos": True}
    if vectors is not None:
        config["soft_vectors_sha256"] = hashlib.sha256((directory / "best_soft.safetensors").read_bytes()).hexdigest()
    config_path = directory / "evaluation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Evaluation config differs at {directory}")
    write_json(config_path, config)
    scores = {}
    for name in names or ("validation", "test", "unseen_words", "longer"):
        path = directory / f"{name}_metrics.json"
        outputs = directory / f"{name}_outputs.jsonl"
        if path.exists() and outputs.exists():
            scores[name] = json.loads(path.read_text())
            continue
        if vectors is None:
            score, rows = evaluate_prompt(model, tok, layout, ids, splits[name],
                                          batch_size=args.generation_batch_size, max_tokens=args.max_tokens)
        else:
            score, rows = evaluate_soft_prompt(objective, ids, vectors, splits[name], max_tokens=args.max_tokens)
        directory.mkdir(parents=True, exist_ok=True)
        outputs.write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_json(path, score)
        scores[name] = score
        print(f"eval {directory.name}/{name}: {score['correct']}/{score['count']} "
              f"({score['accuracy']:.1%}) {score['seconds']:.1f}s", flush=True)
    return scores


def report(output):
    rows = []
    for path in sorted(output.glob("runs/*/result.json")):
        run = json.loads(path.read_text())
        scores = {name: json.loads((path.parent / f"{name}_metrics.json").read_text())
                  for name in ("validation", "test", "unseen_words", "longer")
                  if (path.parent / f"{name}_metrics.json").exists()}
        rows.append({**run, "evaluation": scores})
    write_json(output / "summary.json", rows)
    groups = {}
    for row in rows:
        key = (row["config"]["method"], row["config"]["length"])
        groups.setdefault(key, []).append(row)
    aggregates = []
    for (method, length), group in sorted(groups.items()):
        aggregate = {"method": method, "length": length,
                     "seeds": [r["config"]["seed"] for r in group],
                     "mean_training_seconds": statistics.mean(r["training_seconds"] for r in group),
                     "mean_validation_loss": statistics.mean(r["validation_loss"] for r in group)}
        for name in ("validation", "test", "unseen_words", "longer"):
            scores = [r["evaluation"][name]["accuracy"] for r in group if name in r["evaluation"]]
            if scores:
                aggregate[name] = {"mean": statistics.mean(scores), "min": min(scores), "max": max(scores),
                                   "seed_sd": statistics.stdev(scores) if len(scores) > 1 else None,
                                   "evaluated_seeds": len(scores)}
        aggregates.append(aggregate)
    write_json(output / "aggregate.json", aggregates)
    controls = []
    manual_path = output / "manual_validation.json"
    if manual_path.exists():
        for i, manual in enumerate(json.loads(manual_path.read_text())):
            directory = output / "controls" / f"manual-{i:02d}"
            scores = {name: json.loads((directory / f"{name}_metrics.json").read_text())
                      for name in ("validation", "test", "unseen_words", "longer")
                      if (directory / f"{name}_metrics.json").exists()}
            controls.append({"name": directory.name, "prompt": manual["prompt"],
                             "length": manual["length"], "evaluation": scores})
    write_json(output / "controls.json", controls)
    def prompt_cell(prompt):
        # Literal prompt text must not break a Markdown table or become HTML.
        literal = html.escape(prompt).replace("|", "&#124;").replace("\n", "\\n")
        return f"<code>{literal}</code>" if literal else "(empty)"
    lines = ["# Learned reversal prompts", "",
             "Exact match from greedy generation using the exported system prompt. "
             "Only outer whitespace is ignored; EOS is required. The recorded checkpoint selection rule uses "
             "validation CE or validation exact-match accuracy with CE as the tie-breaker. "
             "Continuous controls are labeled separately and generate using learned input vectors.", "",
             "## Manual controls", "",
             "All instructions were fixed before test evaluation. The best instruction "
             "was selected by validation accuracy; the other rows give length comparisons.", "",
             "| Tokens | Validation | Test (correct/count; 95% Wilson interval) | Unseen words | Longer lists | Prompt |",
             "|---:|---:|---|---:|---:|---|"]
    for control in controls:
        e = control["evaluation"]
        fmt = lambda key: f"{e[key]['accuracy']:.1%}" if key in e else "pending"
        if "test" in e:
            test = e["test"]
            low, high = test["wilson_95"]
            test_text = f"{test['accuracy']:.1%} ({test['correct']}/{test['count']}; {low:.1%}–{high:.1%})"
        else:
            test_text = "pending"
        lines.append(f"| {control['length']} | {fmt('validation')} | {test_text} | "
                     f"{fmt('unseen_words')} | {fmt('longer')} | {prompt_cell(control['prompt'])} |")
    lines.extend(["", "## Across optimizer seeds", "",
             "Ranges describe variation across optimizer seeds on the same held-out examples. "
             "Each run records its initialization; neutral starts are shared across seeds.", "",
             "| Method | Tokens | Seeds | Test mean (range) | Unseen mean | Longer mean |",
             "|---|---:|---:|---:|---:|---:|"])
    for r in aggregates:
        test = r.get("test")
        test_text = f"{test['mean']:.1%} ({test['min']:.1%}–{test['max']:.1%})" if test else "pending"
        fmt = lambda key: f"{r[key]['mean']:.1%}" if key in r else "pending"
        lines.append(f"| {r['method']} | {r['length']} | {len(r['seeds'])} | {test_text} | "
                     f"{fmt('unseen_words')} | {fmt('longer')} |")
    lines.extend(["", "## Individual prompts", "",
             "| Method | Tokens | Seed | Test | Unseen words | Longer lists | Train seconds | Prompt |",
             "|---|---:|---:|---:|---:|---:|---:|---|"])
    for r in rows:
        c, e = r["config"], r["evaluation"]
        fmt = lambda key: f"{e[key]['accuracy']:.1%}" if key in e else "pending"
        lines.append(f"| {c['method']} | {c['length']} | {c['seed']} | {fmt('test')} | "
                     f"{fmt('unseen_words')} | {fmt('longer')} | {r['training_seconds']:.1f} | {prompt_cell(r['prompt'])} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-6bit")
    parser.add_argument("--phase", choices=["pilot", "sweep", "evaluate", "benchmark", "report", "all"], default="all")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--methods", nargs="+", choices=["pez", "gcg", "random", "soft"], default=["pez", "gcg"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=.03)
    parser.add_argument("--pez-optimizer", choices=["adam", "adafactor", "sgd"], default="adam")
    parser.add_argument("--pez-normalize-shadow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pez-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--gradient-microbatch", type=int, default=2)
    parser.add_argument("--pez-gradient-clip", type=float, default=None)
    parser.add_argument("--initialization", choices=["random", "neutral", "copy"], default="random")
    parser.add_argument("--checkpoint-selection", choices=["ce", "accuracy"], default="ce")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--shift-size", type=int, default=128)
    parser.add_argument("--dataset-seed", type=int, default=1729)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--evaluation-splits", nargs="+", default=["validation", "test", "unseen_words", "longer"],
                        choices=["validation", "test", "unseen_words", "longer"])
    parser.add_argument("--benchmark-system-id", default=os.environ.get("DISTIL_BENCHMARK_SYSTEM_ID", platform.platform()))
    parser.add_argument("--evaluate-during-sweep", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.phase == "report":
        report(args.output_dir)
        return
    if min(args.steps, args.batch_size, args.eval_every, args.candidates, args.top_k, args.gradient_microbatch,
           args.generation_batch_size, args.max_tokens, *args.lengths) < 1:
        parser.error("Counts and lengths must be positive")
    if args.batch_size > args.train_size:
        parser.error("Batch size exceeds training set")
    if not math_is_positive(args.learning_rate):
        parser.error("Learning rate must be finite and positive")
    if args.pez_gradient_clip is not None and not math_is_positive(args.pez_gradient_clip):
        parser.error("Gradient clip must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = _download(args.model)
    model, tok = load(model_path)
    model.freeze()
    original = frozen_signature(model)
    layout = PromptLayout(tok)
    objective = PromptObjective(model, tok, layout)
    splits = reversal_splits(tok, seed=args.dataset_seed, train_size=args.train_size, val_size=args.val_size,
                            test_size=args.test_size, shift_size=args.shift_size)
    dataset = {name: [c.to_dict() for c in cases] for name, cases in splits.items()}
    dataset_path = args.output_dir / "dataset.json"
    if dataset_path.exists() and json.loads(dataset_path.read_text()) != dataset:
        raise ValueError("Dataset/config differs from existing output directory")
    write_json(dataset_path, dataset)
    provenance = {"model": args.model, "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                  "model_snapshot": str(model_path.resolve()),
                  "python": sys.version, "platform": platform.platform(),
                  "packages": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers")},
                  "argv": sys.argv, "task": "reverse space-separated single-BPE-token words",
                  "selection": "minimum validation answer CE; no test feedback"}
    manifest_path = args.output_dir / "manifest.json"
    manifest = {k: provenance[k] for k in ("model", "model_snapshot", "dataset_sha256", "packages", "task")}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Model, dataset, or software differs from existing experiment")
    write_json(manifest_path, manifest)
    provenance["source_sha256"] = {
        str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "src/distill/prompt_optimization.py")}
    for relative, digest in provenance["source_sha256"].items():
        path = Path(__file__).resolve().parents[1] / relative
        target = args.output_dir / "source_snapshots" / digest / path.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    write_json(args.output_dir / f"provenance_{args.phase}.json", provenance)

    if args.phase in ("pilot", "all"):
        manual = []
        for i, prompt in enumerate(["", *MANUAL_PROMPTS]):
            ids = layout.encode(prompt)
            directory = args.output_dir / "controls" / f"manual-{i:02d}"
            scores = evaluate_saved(model, tok, layout, splits, ids, directory, args, names=["validation"])
            manual.append({"prompt": prompt, "ids": ids, "length": len(ids), "scores": scores})
        write_json(args.output_dir / "manual_validation.json", manual)
        winner = max(manual[1:], key=lambda r: r["scores"]["validation"]["accuracy"])
        print(f"best manual: {winner['prompt']!r} ({winner['length']} tokens)", flush=True)
        ids = layout.encode("Reverse words.")
        gate = correctness_gate(objective, ids, splits["train"][:4])
        write_json(args.output_dir / "correctness_gate.json", gate)
        print(f"gradient gate passed: CE {gate['gradient_loss']:.4f}; norm {gate['gradient_norm']:.4f}", flush=True)

    vocab = None
    if args.phase in ("benchmark", "all"):
        vocab = PromptVocabulary(tok, objective.trunk.embed_tokens, layout)
        ids = vocab.initial(4, random.Random(0), initialization=args.initialization)
        cases = splits["train"][:args.batch_size]
        # Warm each operation, then collect seven independent timings. Keep
        # optimizer setup outside timing and reset state for comparable rounds.
        timings = {}
        for method in ("pez", "gcg"):
            samples = []
            for i in range(9):
                arm = PEZ(vocab, ids, args.learning_rate, **pez_options(args)) if method == "pez" else GCG(
                    vocab, ids, candidates=args.candidates, top_k=args.top_k)
                mx.synchronize()
                start = perf_counter()
                arm.step(objective, cases)
                mx.synchronize()
                if i >= 2:
                    samples.append(perf_counter() - start)
            timings[method] = {"median_seconds": statistics.median(samples), "samples": samples}
            print(f"benchmark {method}: {timings[method]['median_seconds']:.3f}s/update", flush=True)
        samples = []
        for i in range(9):
            score, _ = evaluate_prompt(model, tok, layout,
                                      layout.encode("Reverse the words. Output only the reversed list."),
                                      splits["train"][:32], batch_size=args.generation_batch_size,
                                      max_tokens=args.max_tokens)
            if i >= 2:
                samples.append(score["seconds"])
        timings["generation_32_cases"] = {"median_seconds": statistics.median(samples),
                                           "samples": samples, "last_accuracy": score["accuracy"]}
        print(f"benchmark generation: {statistics.median(samples):.3f}s/32 cases", flush=True)
        write_json(args.output_dir / "optimization_benchmark.json", {"system_id": args.benchmark_system_id,
                   "batch_size": args.batch_size, "length": 4, "candidates": args.candidates,
                   "learning_rate": args.learning_rate, "pez_options": pez_options(args), "timings": timings})

    if args.phase in ("sweep", "all"):
        if vocab is None:
            vocab = PromptVocabulary(tok, objective.trunk.embed_tokens, layout)
        write_json(args.output_dir / "vocabulary.json", {"size": len(vocab.ids), "ids": vocab.ids})
        print(f"allowed prompt vocabulary: {len(vocab.ids)} tokens", flush=True)
        for length in args.lengths:
            for seed in args.seeds:
                for method in args.methods:
                    directory = args.output_dir / "runs" / f"{method}-l{length:02d}-s{seed}"
                    train_arm(objective, vocab, splits, args, method, length, seed, directory)
                    if args.evaluate_during_sweep:
                        result = json.loads((directory / "result.json").read_text())
                        vectors = mx.load(str(directory / "best_soft.safetensors"))["vectors"] if method == "soft" else None
                        evaluate_saved(model, tok, layout, splits, result["ids"], directory, args,
                                       names=["validation"], objective=objective, vectors=vectors)
                    if not check_frozen(model, original):
                        raise AssertionError("Frozen model weights changed")
                    mx.clear_cache()

    if args.phase in ("evaluate", "all"):
        for path in sorted((args.output_dir / "runs").glob("*/result.json")):
            run = json.loads(path.read_text())
            soft = run["config"]["method"] == "soft"
            if not soft and (len(layout.encode(run["prompt"])) != run["config"]["length"] or not layout.valid(run["ids"])):
                raise AssertionError(f"Prompt export/length failed: {path}")
            vectors = mx.load(str(path.parent / "best_soft.safetensors"))["vectors"] if soft else None
            evaluate_saved(model, tok, layout, splits, run["ids"], path.parent, args,
                           names=args.evaluation_splits, objective=objective, vectors=vectors)
        manual_path = args.output_dir / "manual_validation.json"
        if manual_path.exists():
            manual = json.loads(manual_path.read_text())
            for i, control in enumerate(manual):
                evaluate_saved(model, tok, layout, splits, control["ids"],
                               args.output_dir / "controls" / f"manual-{i:02d}", args,
                               names=args.evaluation_splits)
            winner = max(manual[1:], key=lambda r: r["scores"]["validation"]["accuracy"])
            for name, ids in (("empty", []), ("manual-best", winner["ids"])):
                evaluate_saved(model, tok, layout, splits, ids, args.output_dir / "controls" / name, args,
                               names=args.evaluation_splits)
    write_json(args.output_dir / f"invariants_{args.phase}.json", {"frozen_weights_unchanged": check_frozen(model, original)})
    report(args.output_dir)


def math_is_positive(value):
    import math
    return math.isfinite(value) and value > 0


if __name__ == "__main__":
    main()
