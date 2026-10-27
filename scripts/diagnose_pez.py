#!/usr/bin/env python3
"""Controlled PEZ development ablations; never inspect held-out test outputs."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm import load

from distill.prompt_optimization import PEZ, PromptLayout, PromptObjective, PromptVocabulary, evaluate_prompt, reversal_splits


MODEL = "/Users/stefanpetrov/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-6bit/snapshots/fbb1092aa9daef007603ea9273cd24c225a409f1"
VARIANTS = {
    "old-adam-b2": {"optimizer": "adam", "bias_correction": False, "lr": .03, "batch_size": 2},
    "adam-b2": {"optimizer": "adam", "bias_correction": True, "lr": .03, "batch_size": 2},
    "adam-b16": {"optimizer": "adam", "bias_correction": True, "lr": .03, "batch_size": 16},
    "adam-small-b16": {"optimizer": "adam", "bias_correction": True, "lr": .003, "batch_size": 16},
    "adafactor-b16": {"optimizer": "adafactor", "lr": .03, "batch_size": 16},
    "adafactor-large-b16": {"optimizer": "adafactor", "lr": .1, "batch_size": 16},
    "sphere-small-b2": {"optimizer": "adam", "bias_correction": True, "lr": .003, "batch_size": 2, "normalize_shadow": True},
    "sphere-b2": {"optimizer": "adam", "bias_correction": True, "lr": .01, "batch_size": 2, "normalize_shadow": True},
    "euclidean-small-b2": {"optimizer": "adam", "bias_correction": True, "lr": .003, "batch_size": 2, "metric": "euclidean"},
    "adafactor-b2": {"optimizer": "adafactor", "lr": .01, "batch_size": 2},
    "sphere-small-b8": {"optimizer": "adam", "bias_correction": True, "lr": .001, "batch_size": 8, "normalize_shadow": True},
    "adam-small-b8": {"optimizer": "adam", "bias_correction": True, "lr": .0003, "batch_size": 8},
    "adafactor-b8": {"optimizer": "adafactor", "lr": .01, "batch_size": 8},
    "reference-adafactor-b8": {"optimizer": "adafactor", "lr": .03, "batch_size": 8, "gradient_clip": 1.},
    "clipped-adam-b8": {"optimizer": "adam", "bias_correction": True, "lr": .003, "batch_size": 8, "gradient_clip": 1.},
    "neutral-adam-b2": {"optimizer": "adam", "bias_correction": True, "lr": .003, "batch_size": 2, "gradient_clip": 1., "initialization": "neutral"},
    "neutral-adafactor-b2": {"optimizer": "adafactor", "lr": .01, "batch_size": 2, "gradient_clip": 1., "initialization": "neutral"},
    "english-adam-b2": {"optimizer": "adam", "bias_correction": True, "lr": .003, "batch_size": 2, "gradient_clip": 1., "initialization": "neutral", "normalize_shadow": True, "vocabulary": "english"},
    "english-adafactor-b2": {"optimizer": "adafactor", "lr": .01, "batch_size": 2, "gradient_clip": 1., "initialization": "neutral", "vocabulary": "english"},
    "neutral-adafactor-b16": {"optimizer": "adafactor", "lr": .01, "batch_size": 16, "gradient_clip": 1., "initialization": "neutral"},
    "neutral-sgd-b2": {"optimizer": "sgd", "lr": .3, "batch_size": 2, "gradient_clip": 1., "initialization": "neutral"},
    "neutral-sphere-sgd-b2": {"optimizer": "sgd", "lr": .3, "batch_size": 2, "gradient_clip": 1., "initialization": "neutral", "normalize_shadow": True},
    "copy-adafactor-b2": {"optimizer": "adafactor", "lr": .01, "batch_size": 2, "gradient_clip": 1., "initialization": "copy"},
    "copy-sphere-sgd-b2": {"optimizer": "sgd", "lr": .3, "batch_size": 2, "gradient_clip": 1., "initialization": "copy", "normalize_shadow": True},
}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lengths", nargs="+", type=int, default=[8])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=20261009)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = {"pid": os.getpid(), "status": "running", "started_unix": time.time()}
        save(output / "job.json", job)
        try:
            model, tokenizer = load(MODEL)
            layout = PromptLayout(tokenizer)
            objective = PromptObjective(model, tokenizer, layout)
            needs_ascii = any(VARIANTS[v].get("vocabulary") != "english" for v in args.variants)
            vocab = PromptVocabulary(tokenizer, objective.trunk.embed_tokens, layout) if needs_ascii else None
            dictionary = Path("/usr/share/dict/web2")
            words = {w.strip().lower() for w in dictionary.read_text().splitlines()}
            english_vocab = None
            splits = reversal_splits(tokenizer, seed=args.dataset_seed)
            save(output / "development_data.json", {name: [c.to_dict() for c in splits[name]] for name in ("train", "validation")})
            save(output / f"provenance-{time.time_ns()}.json", {"model_snapshot": MODEL, "dataset_seed": args.dataset_seed,
                 "source_sha256": hashlib.sha256(Path("src/distill/prompt_optimization.py").read_bytes()).hexdigest(),
                 "selection": "development validation CE and generation; no test feedback", "variants": VARIANTS})
            for source in (Path(__file__), Path("src/distill/prompt_optimization.py")):
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                snapshot = output / "source_snapshots" / digest / source.name
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(source.read_bytes())
            for length in args.lengths:
                for variant in args.variants:
                    config = {**VARIANTS[variant], "length": length, "steps": args.steps,
                              "seed": args.seed, "dataset_seed": args.dataset_seed}
                    if config.get("vocabulary") == "english":
                        if english_vocab is None:
                            english_vocab = PromptVocabulary(tokenizer, objective.trunk.embed_tokens, layout, allowed_words=words)
                            (output / "dictionary.txt").write_bytes(dictionary.read_bytes())
                            save(output / "english_vocabulary.json", {"size": len(english_vocab.ids), "ids": english_vocab.ids,
                                 "dictionary_sha256": hashlib.sha256(dictionary.read_bytes()).hexdigest()})
                        active_vocab = english_vocab
                    else:
                        active_vocab = vocab
                    initial = active_vocab.initial(length, random.Random(args.seed), initialization=config.get("initialization", "random"))
                    name = f"{variant}-l{length:02d}"
                    run = output / name
                    if (run / "result.json").exists():
                        if json.loads((run / "result.json").read_text())["config"] != config:
                            raise ValueError(f"Configuration differs: {run}")
                        print(f"reuse {name}", flush=True)
                        continue
                    if run.exists():
                        archived = output / "interrupted"
                        archived.mkdir(exist_ok=True)
                        run.rename(archived / f"{name}-{time.time_ns()}")
                    run.mkdir()
                    job["arm"] = name
                    save(output / "job.json", job)
                    arm = PEZ(active_vocab, initial, learning_rate=config["lr"],
                              normalize_shadow=config.get("normalize_shadow", False), metric=config.get("metric", "cosine"),
                              gradient_clip=config.get("gradient_clip"), optimizer=config["optimizer"])
                    if config["optimizer"] == "adam":
                        arm.optimizer = optim.Adam(config["lr"], bias_correction=config["bias_correction"])
                    elif config["optimizer"] == "adafactor":
                        arm.optimizer = optim.Adafactor(learning_rate=config["lr"], relative_step=False,
                                                       scale_parameter=False, warmup_init=False)
                    initial_norm = mx.linalg.norm(arm.shadow, axis=-1).mean().item()
                    best_ids = list(initial)
                    best_loss = initial_loss = objective.score([initial], splits["validation"])[0]
                    best_step = 0
                    rng = random.Random(args.seed + 100000)
                    started = time.perf_counter()
                    for step in range(1, args.steps + 1):
                        before = arm.shadow
                        stats = arm.step(objective, rng.sample(splits["train"], config["batch_size"]))
                        stats.update(step=step, ids=list(arm.ids), prompt=tokenizer.decode(arm.ids),
                                     update_l2=mx.linalg.norm(arm.shadow - before, axis=-1).mean().item(),
                                     shadow_l2=mx.linalg.norm(arm.shadow, axis=-1).mean().item())
                        with (run / "updates.jsonl").open("a") as log:
                            log.write(json.dumps(stats) + "\n")
                        if step % 10 == 0 or step == args.steps:
                            val = objective.score([arm.ids], splits["validation"])[0]
                            if val < best_loss:
                                best_ids, best_loss, best_step = list(arm.ids), val, step
                            raw = active_vocab.project(arm.shadow, metric=arm.metric)
                            print(f"{name} {step}/{args.steps} val={val:.4f} best={best_loss:.4f} "
                                  f"shadow/initial={stats['shadow_l2']/initial_norm:.1f} raw_valid={layout.valid(raw)} "
                                  f"prompt={stats['prompt']!r}", flush=True)
                    score, rows = evaluate_prompt(model, tokenizer, layout, best_ids, splits["validation"], batch_size=64)
                    (run / "validation_outputs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
                    result = {"config": config, "initial_ids": initial, "initial_validation_ce": initial_loss,
                              "initial_embedding_l2": initial_norm, "best_step": best_step,
                              "validation_ce": best_loss, "validation": score,
                              "ids": best_ids, "prompt": tokenizer.decode(best_ids),
                              "repaired_projections": arm.rejected,
                              "last_shadow_l2": mx.linalg.norm(arm.shadow, axis=-1).mean().item(),
                              "seconds": time.perf_counter()-started}
                    save(run / "result.json", result)
                    print(f"DONE {name}: {score['correct']}/{score['count']} {result['prompt']!r}", flush=True)
                    mx.clear_cache()
            job.update(status="complete", finished_unix=time.time())
        except BaseException as error:
            job.update(status="failed", error=repr(error), finished_unix=time.time())
            raise
        finally:
            save(output / "job.json", job)


if __name__ == "__main__":
    main()
