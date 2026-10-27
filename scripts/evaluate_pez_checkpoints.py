#!/usr/bin/env python3
"""Check validation generation at saved PEZ checkpoints, without test feedback."""
import argparse
import hashlib
import json
from pathlib import Path

from mlx_lm import load

from distil.prompt_optimization import PromptLayout, PromptObjective, ReversalCase, evaluate_prompt

MODEL = "/Users/stefanpetrov/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-6bit/snapshots/fbb1092aa9daef007603ea9273cd24c225a409f1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    result = json.loads((run / "result.json").read_text())
    experiment = run.parent.parent if run.parent.name == "runs" else run.parent
    dataset_path = experiment / "dataset.json"
    if not dataset_path.exists():
        dataset_path = experiment / "development_data.json"
    data = json.loads(dataset_path.read_text())["validation"]
    cases = [ReversalCase(c["case_id"], tuple(c["words"])) for c in data]
    if any(c.answer != r["answer"] or c.question != r["question"] for c, r in zip(cases, data, strict=True)):
        raise ValueError("Saved validation cases disagree with the reversal oracle")
    checkpoint_path = run / "checkpoints.jsonl"
    if checkpoint_path.exists():
        checkpoints = [json.loads(line) for line in checkpoint_path.read_text().splitlines()]
    else:
        updates = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
        checkpoints = [{"step": 0, "ids": result["initial_ids"]}]
        checkpoints.extend(r for r in updates if r["step"] % 10 == 0 or r["step"] == result["config"]["steps"])
    model, tokenizer = load(MODEL)
    layout = PromptLayout(tokenizer)
    objective = PromptObjective(model, tokenizer, layout)
    output = run / "checkpoint_validation_generation"
    output.mkdir(exist_ok=True)
    evaluations, cache = [], {}
    for checkpoint in checkpoints:
        ids, step = checkpoint["ids"], checkpoint["step"]
        key = tuple(ids)
        if not layout.valid(ids):
            raise ValueError(f"Checkpoint {step} cannot be exported as ordinary text")
        if key not in cache:
            loss = objective.score([ids], cases)[0]
            metrics, rows = evaluate_prompt(model, tokenizer, layout, ids, cases, batch_size=64, max_tokens=24)
            cache[key] = loss, metrics, rows
        loss, metrics, rows = cache[key]
        evaluation = {"step": step, "ids": ids, "prompt": tokenizer.decode(ids), "validation_ce": loss,
                      "validation": metrics}
        evaluations.append(evaluation)
        (output / f"step-{step:04d}_outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"step {step}: CE {loss:.4f}, exact {metrics['correct']}/{metrics['count']}, {evaluation['prompt']!r}", flush=True)
    best_ce = min(evaluations, key=lambda r: r["validation_ce"])
    best_accuracy = max(evaluations, key=lambda r: (r["validation"]["accuracy"], -r["validation_ce"]))
    report = {"model_snapshot": MODEL, "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "split": "validation", "max_tokens": 24, "generation_batch_size": 64,
              "best_by_ce": best_ce, "best_by_accuracy": best_accuracy, "checkpoints": evaluations}
    (run / "checkpoint_generation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Best CE checkpoint: {best_ce['validation']['correct']}/{len(cases)}; "
          f"best accuracy checkpoint: {best_accuracy['validation']['correct']}/{len(cases)}", flush=True)


if __name__ == "__main__":
    main()
