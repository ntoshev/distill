#!/usr/bin/env python3
"""Run the reversal sweep and controls serially, with durable process status.

This supervisor can be launched in a separate session with redirected stdio.
Completed arms are reused; partial arms are archived before restarting. It does
not change optimizer behavior or use test outcomes to select configurations.
"""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import os
import time
import uuid


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-6bit")
    parser.add_argument("--dataset-seed", type=int, default=20261008)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=.03)
    parser.add_argument("--methods", nargs="+", choices=["pez", "gcg", "random", "soft"],
                        default=["pez", "gcg", "random", "soft"])
    parser.add_argument("--pez-optimizer", choices=["adam", "adafactor", "sgd"], default="adam")
    parser.add_argument("--pez-normalize-shadow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pez-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--gradient-microbatch", type=int, default=2)
    parser.add_argument("--pez-gradient-clip", type=float, default=None)
    parser.add_argument("--initialization", choices=["random", "neutral", "copy"], default="random")
    parser.add_argument("--checkpoint-selection", choices=["ce", "accuracy"], default="ce")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".suite.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"pid": os.getpid(), "status": "running", "completed_phases": [],
                  "started_unix": time.time(), "argv": sys.argv, "archived_runs": []}
        status_path = output / "suite_job.json"
        for run in sorted((output / "runs").glob("*-l*-s*")):
            if run.is_dir() and not (run / "result.json").exists():
                target = output / "interrupted" / f"{run.name}-{uuid.uuid4().hex[:8]}"
                target.parent.mkdir(exist_ok=True)
                run.rename(target)
                status["archived_runs"].append(str(target.relative_to(output)))
        save(status_path, status)
        common = [sys.executable, str(root / "scripts/learn_reversal_prompt.py"),
                  "--output-dir", str(output), "--model", args.model,
                  "--dataset-seed", str(args.dataset_seed), "--steps", str(args.steps),
                  "--batch-size", str(args.batch_size), "--candidates", str(args.candidates),
                  "--learning-rate", str(args.learning_rate), "--seeds", *map(str, args.seeds),
                  "--pez-optimizer", args.pez_optimizer, "--pez-metric", args.pez_metric,
                  "--initialization", args.initialization,
                  "--checkpoint-selection", args.checkpoint_selection, "--eval-every", str(args.eval_every),
                  "--gradient-microbatch", str(args.gradient_microbatch),
                  "--pez-normalize-shadow" if args.pez_normalize_shadow else "--no-pez-normalize-shadow"]
        if args.pez_gradient_clip is not None:
            common.extend(["--pez-gradient-clip", str(args.pez_gradient_clip)])
        phases = []
        if not (output / "correctness_gate.json").exists():
            phases.append(("pilot", ["--phase", "pilot"]))
        if not (output / "optimization_benchmark.json").exists():
            phases.append(("benchmark", ["--phase", "benchmark"]))
        lengths = ["--lengths", *map(str, args.lengths)]
        discrete = [method for method in args.methods if method != "soft"]
        if discrete:
            phases.append(("discrete_sweep", ["--phase", "sweep", "--methods", *discrete, *lengths]))
        if "soft" in args.methods:
            phases.append(("continuous_control", ["--phase", "sweep", "--methods", "soft", "--lengths", "4"]))
        phases.append(("final_evaluation", ["--phase", "evaluate"]))
        try:
            for name, options in phases:
                status.update({"phase": name, "phase_started_unix": time.time()})
                command = common + options
                print(f"Starting {name}", flush=True)
                with (output / f"suite_{name}.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                    status.update({"child_pid": child.pid, "child_command": command})
                    save(status_path, status)
                    code = child.wait()
                status["child_exit_code"] = code
                if code:
                    raise RuntimeError(f"{name} exited with code {code}; see suite_{name}.log")
                status["completed_phases"].append(name)
                save(status_path, status)
            status.update({"status": "complete", "finished_unix": time.time()})
            save(status_path, status)
            print("Suite complete", flush=True)
        except BaseException as error:
            status.update({"status": "failed", "error": str(error), "finished_unix": time.time()})
            save(status_path, status)
            raise


if __name__ == "__main__":
    main()
