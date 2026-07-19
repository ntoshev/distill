#!/usr/bin/env python3
"""Evaluate fact acquisition at every checkpoint in an LR sweep."""

import argparse
import json
from pathlib import Path
from time import perf_counter

TARGET = "VIOLET-731"
TRAIN_PROMPT = "What is Project Lumen's launch code? Answer with the code only."
HELDOUT_PROMPT = "Give the launch code assigned to Project Lumen. Answer with the code only."


def answer(model, tokenizer, prompt: str) -> str:
    from distil.sample import sample_completion

    return sample_completion(
        model,
        tokenizer,
        [{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.0,
        seed=0,
        enable_thinking=False,
    ).content.strip()


def first_iteration(rows: list[dict], predicate) -> int | None:
    return next((row["iteration"] for row in rows if predicate(row)), None)


def select_best_checkpoint(rows: list[dict]) -> dict | None:
    """Jointly learned checkpoint with least drift, breaking ties by step."""
    learned = [row for row in rows if row["train_exact"] and row["heldout_exact"]]
    return (
        min(
            learned,
            key=lambda row: (
                row["canary_contaminated"],
                row["canary_changed"],
                row["iteration"],
            ),
        )
        if learned
        else None
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--model", help="override the model recorded in manifest.json")
    args = parser.parse_args()

    manifest = json.loads((args.sweep / "manifest.json").read_text())
    model_name = args.model or manifest["model"]

    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

    model, tokenizer = load(model_name)
    baseline = {
        "train": answer(model, tokenizer, TRAIN_PROMPT),
        "heldout": answer(model, tokenizer, HELDOUT_PROMPT),
    }
    summaries = []

    for spec in manifest["specs"]:
        label = spec["label"]
        learning_rate = spec["learning_rate"]
        run_path = args.sweep / f"lr-{label}"
        metrics = {
            row["iteration"]: row
            for row in (
                json.loads(line)
                for line in (run_path / "training_metrics.jsonl").read_text().splitlines()
            )
        }
        rows = []
        probe_file = run_path / "fact_probes.jsonl"
        checkpoints = sorted((run_path / "checkpoints").glob("step-*"))
        if not checkpoints:
            raise ValueError(f"no checkpoints found in {run_path / 'checkpoints'}")

        with probe_file.open("w") as f:
            for checkpoint in checkpoints:
                iteration = int(checkpoint.name.rsplit("-", 1)[1])
                started = perf_counter()
                try:
                    load_adapters(model, str(checkpoint))
                    train_answer = answer(model, tokenizer, TRAIN_PROMPT)
                    heldout_answer = answer(model, tokenizer, HELDOUT_PROMPT)
                finally:
                    remove_lora_layers(model)
                    model.eval()

                metric = metrics[iteration]
                canaries = metric["canaries"]
                sampling = metric.get("sampling", {})
                row = {
                    "iteration": iteration,
                    "learning_rate": learning_rate,
                    "cumulative_lr": learning_rate * iteration,
                    "train_answer": train_answer,
                    "heldout_answer": heldout_answer,
                    "train_exact": train_answer == TARGET,
                    "heldout_exact": heldout_answer == TARGET,
                    "canary_changed": sum(item["changed"] for item in canaries),
                    "canary_contaminated": sum(
                        item["contaminated"] for item in canaries
                    ),
                    "accepted_tokens": sampling.get("accepted_tokens"),
                    "replaced_tokens": sampling.get("replaced_tokens"),
                    "acceptance_rate": sampling.get("acceptance_rate"),
                    "evaluation_seconds": perf_counter() - started,
                    "train_seconds": metric["timing"]["train_seconds"],
                    "checkpoint_seconds": metric["timing"]["checkpoint_seconds"],
                    "canary_seconds": metric["timing"]["canary_seconds"],
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        best = select_best_checkpoint(rows)
        accepted = [
            row["accepted_tokens"]
            for row in rows
            if row["accepted_tokens"] is not None
        ]
        replaced = [
            row["replaced_tokens"]
            for row in rows
            if row["replaced_tokens"] is not None
        ]
        total_interleaved = sum(accepted) + sum(replaced)
        summaries.append(
            {
                "label": label,
                "learning_rate": learning_rate,
                "steps": len(rows),
                "first_train_exact": first_iteration(rows, lambda row: row["train_exact"]),
                "first_heldout_exact": first_iteration(rows, lambda row: row["heldout_exact"]),
                "first_both_exact": first_iteration(
                    rows, lambda row: row["train_exact"] and row["heldout_exact"]
                ),
                "first_canary_change": first_iteration(
                    rows, lambda row: row["canary_changed"] > 0
                ),
                "first_contamination": first_iteration(
                    rows, lambda row: row["canary_contaminated"] > 0
                ),
                "best_learned_checkpoint": best,
                "mean_acceptance_rate": (
                    sum(accepted) / total_interleaved if total_interleaved else None
                ),
                "mean_train_seconds": sum(row["train_seconds"] for row in rows) / len(rows),
                "mean_checkpoint_seconds": (
                    sum(row["checkpoint_seconds"] for row in rows) / len(rows)
                ),
                "mean_canary_seconds": (
                    sum(row["canary_seconds"] for row in rows) / len(rows)
                ),
            }
        )

    output = {
        "model": model_name,
        "target": TARGET,
        "baseline": baseline,
        "sampling": manifest.get("sampling", {"policy": "student", "top_k": None}),
        "runs": summaries,
    }
    (args.sweep / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
