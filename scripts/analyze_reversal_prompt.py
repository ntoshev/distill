#!/usr/bin/env python3
"""Audit saved reversal experiments without loading a model or generating text.

Checks the complete arm matrix, paired initialization, checkpoint selection,
minibatch acceptance, dataset isolation, and metrics against every saved output.
Writes audit.json and diagnostics.json in the experiment output directory.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


SPLITS = ("validation", "test", "unseen_words", "longer")


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def failure_type(row):
    output = row["output"].strip()
    if output == row["answer"]:
        return "correct" if row["finish_reason"] == "stop" else "missing_eos"
    if not output:
        return "empty"
    if output == row["question"]:
        return "copy"
    if Counter(output.split()) == Counter(row["words"]):
        return "wrong_order"
    return "other"


def audit_outputs(directory, dataset):
    failures = {}
    for split in SPLITS:
        rows = read_rows(directory / f"{split}_outputs.jsonl")
        metrics = read_json(directory / f"{split}_metrics.json")
        require(len(rows) == len(dataset[split]), f"Incomplete outputs: {directory}/{split}")
        kinds = Counter()
        for row, case in zip(rows, dataset[split], strict=True):
            require(all(row[key] == case[key] for key in ("case_id", "words", "question", "answer")),
                    f"Output was scored against a different case: {directory}/{split}")
            kind = failure_type(row)
            kinds[kind] += 1
            require(row["correct"] == (kind == "correct"), f"Incorrect exact-match flag: {directory}")
            require(row["copy"] == (row["output"].strip() == case["question"]), f"Incorrect copy flag: {directory}")
            require(row["empty"] == (not row["output"].strip()), f"Incorrect empty flag: {directory}")
            require(row["finish_reason"] in ("stop", "length"), f"Invalid finish reason: {directory}")
            require(row["num_tokens"] == len(row["token_ids"]) > 0, f"Invalid token count: {directory}")
        require(metrics["count"] == len(rows) and metrics["correct"] == kinds["correct"],
                f"Metrics disagree with outputs: {directory}/{split}")
        for key, count in (("accuracy", kinds["correct"]), ("copy_rate", kinds["copy"]),
                           ("empty_rate", kinds["empty"]),
                           ("truncation_rate", sum(r["finish_reason"] == "length" for r in rows))):
            require(math.isclose(metrics[key], count / len(rows), abs_tol=1e-12),
                    f"Incorrect {key}: {directory}/{split}")
        require(metrics["wilson_95"][0] <= metrics["accuracy"] <= metrics["wilson_95"][1],
                f"Invalid interval: {directory}/{split}")
        failures[split] = dict(kinds)
    return failures


def audit(output, lengths, seeds, methods=("pez", "gcg", "random", "soft")):
    dataset = read_json(output / "dataset.json")
    manifest = read_json(output / "manifest.json")
    require(hashlib.sha256((output / "dataset.json").read_bytes()).hexdigest() == manifest["dataset_sha256"],
            "Dataset differs from model/dataset manifest")
    for provenance in output.glob("provenance_*.json"):
        for relative, digest in read_json(provenance)["source_sha256"].items():
            snapshot = output / "source_snapshots" / digest / Path(relative).name
            require(hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest,
                    f"Source snapshot does not match provenance: {snapshot}")
    sequences = set()
    for split, cases in dataset.items():
        for case in cases:
            sequence = tuple(case["words"])
            require(sequence not in sequences and sequence != sequence[::-1], "Duplicate/palindromic input")
            sequences.add(sequence)
            require(case["question"] == " ".join(sequence) and case["answer"] == " ".join(sequence[::-1]),
                    "Reversal oracle disagreement")
            require((6 <= len(sequence) <= 8) if split == "longer" else (3 <= len(sequence) <= 5),
                    "Input length outside task definition")
    known = {word for case in dataset["train"] for word in case["words"]}
    unseen = {word for case in dataset["unseen_words"] for word in case["words"]}
    require(not known & unseen, "Unseen-word test contains training words")
    for split in ("validation", "test", "longer"):
        require(all(set(case["words"]) <= known for case in dataset[split]), f"Wrong vocabulary: {split}")

    expected = {(m, n, s) for m in methods if m != "soft" for n in lengths for s in seeds}
    if "soft" in methods:
        expected |= {("soft", 4, s) for s in seeds}
    runs = [(path.parent, read_json(path)) for path in sorted((output / "runs").glob("*/result.json"))]
    actual = {(r["config"]["method"], r["config"]["length"], r["config"]["seed"]) for _, r in runs}
    require(actual == expected and len(runs) == len(expected), f"Run matrix differs: missing={expected-actual}, extra={actual-expected}")
    initializations, diagnostics = {}, []
    for directory, run in runs:
        config = run["config"]
        key = (config["length"], config["seed"])
        initializations.setdefault(key, run["initial_ids"])
        require(initializations[key] == run["initial_ids"], f"Unpaired initialization: {directory}")
        require(run["canonical"] and len(run["ids"]) == config["length"], f"Invalid exported prompt: {directory}")
        checkpoints = read_rows(directory / "checkpoints.jsonl")
        steps = read_rows(directory / "training.jsonl")
        require([s["step"] for s in steps] == list(range(1, config["steps"] + 1)), f"Incomplete training: {directory}")
        if config.get("checkpoint_selection", "ce") == "accuracy":
            for checkpoint in checkpoints:
                generated = read_rows(directory / "checkpoint_generation" / f"step-{checkpoint['step']:04d}_outputs.jsonl")
                require(len(generated) == len(dataset["validation"]), f"Incomplete checkpoint generation: {directory}")
                correct = 0
                for row, case in zip(generated, dataset["validation"], strict=True):
                    require(all(row[k] == case[k] for k in ("case_id", "words", "question", "answer")),
                            f"Checkpoint used different validation cases: {directory}")
                    actual_correct = row["output"].strip() == case["answer"] and row["finish_reason"] == "stop"
                    require(row["correct"] == actual_correct, f"Checkpoint exact-match flag disagrees: {directory}")
                    correct += actual_correct
                require(checkpoint["validation_correct"] == correct and
                        checkpoint["validation_accuracy"] == correct / len(generated),
                        f"Checkpoint accuracy disagrees with generation: {directory}")
            best = max(checkpoints, key=lambda c: (c["validation_accuracy"], -c["validation_loss"]))
            require(run["validation_accuracy"] == best["validation_accuracy"], f"Selected accuracy mismatch: {directory}")
        else:
            best = min(checkpoints, key=lambda c: c["validation_loss"])
        require(run["best_step"] == best["step"] and run["ids"] == best["ids"]
                and run["validation_loss"] == best["validation_loss"], f"Checkpoint selection mismatch: {directory}")
        require(read_json(directory / "best_prompt.json") == best, f"Saved checkpoint mismatch: {directory}")
        require(checkpoints[0]["ids"] == run["initial_ids"], f"Initial checkpoint mismatch: {directory}")
        previous, changes = run["initial_ids"], 0
        for step in steps:
            require(len(step["ids"]) == config["length"] and math.isfinite(step["loss_before"]),
                    f"Invalid training state: {directory}")
            mutations = sum(a != b for a, b in zip(previous, step["ids"], strict=True))
            changes += int(mutations > 0)
            if config["method"] in ("gcg", "random"):
                require(mutations <= 1 and step["loss_after"] <= step["loss_before"],
                        f"Invalid coordinate acceptance: {directory}")
                require(step["candidates_scored"] <= config["candidates"], f"Exceeded candidate budget: {directory}")
            else:
                require(math.isfinite(step["gradient_norm"]) and step["gradient_norm"] > 0,
                        f"Invalid prompt gradient: {directory}")
            previous = step["ids"]
        if config["method"] == "soft":
            evaluation = read_json(directory / "evaluation_config.json")
            require(hashlib.sha256((directory / "best_soft.safetensors").read_bytes()).hexdigest()
                    == evaluation["soft_vectors_sha256"], f"Wrong continuous checkpoint: {directory}")
        diagnostics.append({"run": directory.name, "method": config["method"], "length": config["length"],
                            "seed": config["seed"], "initial_validation_ce": checkpoints[0]["validation_loss"],
                            "best_validation_ce": run["validation_loss"], "train_ce": run["train_loss"],
                            "best_step": run["best_step"], "changed_updates": changes,
                            "repaired_projections": run["repaired_projections"],
                            "step_median_seconds": statistics.median(s["step_seconds"] for s in steps),
                            "step_max_seconds": max(s["step_seconds"] for s in steps),
                            "forward_examples": run["forward_examples"], "gradient_examples": run["gradient_examples"],
                            "outputs": audit_outputs(directory, dataset)})
    controls = read_json(output / "manual_validation.json")
    for i, control in enumerate(controls):
        directory = output / "controls" / f"manual-{i:02d}"
        require(read_json(directory / "evaluation_config.json")["prompt_ids"] == control["ids"],
                f"Wrong manual prompt: {directory}")
        audit_outputs(directory, dataset)
    winner = max(controls[1:], key=lambda c: c["scores"]["validation"]["accuracy"])
    for name, ids in (("empty", []), ("manual-best", winner["ids"])):
        directory = output / "controls" / name
        require(read_json(directory / "evaluation_config.json")["prompt_ids"] == ids,
                f"Validation-selected baseline changed: {directory}")
        audit_outputs(directory, dataset)
    for phase in ("sweep", "evaluate"):
        gate = output / f"invariants_{phase}.json"
        # --phase all records one final gate after both training and evaluation.
        if not gate.exists():
            gate = output / "invariants_all.json"
        require(gate.exists(), f"Missing frozen-weight gate for {phase}")
        require(read_json(gate)["frozen_weights_unchanged"], "Model weights changed")
    result = {"passed": True, "run_count": len(runs), "manual_control_count": len(controls), "baseline_alias_count": 2,
              "dataset_sizes": {name: len(cases) for name, cases in dataset.items()},
              "audited_generated_outputs": (len(runs) + len(controls) + 2) * sum(len(dataset[n]) for n in SPLITS),
              "model": manifest["model"], "dataset_sha256": manifest["dataset_sha256"],
              "checks": ["complete arm matrix", "paired initialization", "dataset and oracle isolation", "source snapshot hashes",
                         "validation-only checkpoint selection", "coordinate acceptance and budgets",
                         "all saved outputs and metrics", "validation-selected baseline", "continuous checkpoint hashes", "frozen-weight gates"],
              "scope": "Artifact consistency; gradient correctness and token round trips are separate runtime tests."}
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=["pez", "gcg", "random", "soft"],
                        default=["pez", "gcg", "random", "soft"])
    args = parser.parse_args()
    print(json.dumps(audit(args.output_dir, args.lengths, args.seeds, args.methods), indent=2))


if __name__ == "__main__":
    main()
