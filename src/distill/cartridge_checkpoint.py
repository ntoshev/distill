"""Atomic evaluation-boundary commits and metric-log recovery."""

import json
import os
from pathlib import Path
import shutil
import uuid


def write_json_atomic(path, data):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w") as file:
        file.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def commit_checkpoint(session, root: Path, state: dict):
    directory = root / "checkpoints"
    directory.mkdir(exist_ok=True)
    name = f"step-{session.iteration:04d}"
    temporary = directory / ("." + name + "-" + uuid.uuid4().hex)
    temporary.mkdir()
    from distill.cartridge import canonical, digest

    state = dict(state)
    config_path = root / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        state["config_hash"] = digest(
            canonical(
                {
                    k: v
                    for k, v in config.items()
                    if k not in ("iters", "passes", "resolved_iters")
                }
            )
        )
    for log_name in ("training_metrics.jsonl", "eval_metrics.jsonl"):
        with (root / log_name).open("r+b") as file:
            os.fsync(file.fileno())
    state["log_offsets"] = {
        file: (root / file).stat().st_size
        for file in ("training_metrics.jsonl", "eval_metrics.jsonl")
    }
    try:
        session.save(temporary)
        session.save_resume(temporary, state)
        for path in temporary.iterdir():
            with path.open("rb") as file:
                os.fsync(file.fileno())
        final = directory / name
        if final.exists():
            # An orphan directory can remain if a crash occurred before latest.
            latest = (
                json.loads((root / "latest.json").read_text())
                if (root / "latest.json").exists()
                else {}
            )
            if latest.get("step", -1) >= session.iteration:
                raise ValueError("refusing to overwrite a committed checkpoint")
            shutil.rmtree(final)
        os.replace(temporary, final)
        write_json_atomic(
            root / "latest.json",
            {
                "schema_version": 2,
                "step": session.iteration,
                "checkpoint": f"checkpoints/{name}",
            },
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return final


def latest_checkpoint(root):
    root = Path(root)
    data = json.loads((root / "latest.json").read_text())
    expected = f"checkpoints/step-{data['step']:04d}"
    if data.get("checkpoint") != expected:
        raise ValueError("invalid latest checkpoint pointer")
    checkpoint = root / expected
    if not all(
        (checkpoint / file).is_file()
        for file in (
            "cartridge.json",
            "cartridge.safetensors",
            "resume.json",
            "resume.safetensors",
        )
    ):
        raise ValueError("latest checkpoint is incomplete")
    return checkpoint


def recover_logs(root: Path, offsets: dict):
    for name in ("training_metrics.jsonl", "eval_metrics.jsonl"):
        offset = offsets[name]
        if offset < 0 or (root / name).stat().st_size < offset:
            raise ValueError("metric log is shorter than committed checkpoint")
        with (root / name).open("r+b") as file:
            file.truncate(offset)


def copy_export(checkpoint, root):
    for name in ("cartridge.json", "cartridge.safetensors"):
        target = Path(root) / name
        temp = target.with_name("." + name + ".tmp")
        shutil.copyfile(Path(checkpoint) / name, temp)
        os.replace(temp, target)
    return Path(root) / "cartridge.safetensors"


def validate_resume_config(root, state):
    from distill.cartridge import canonical, digest

    config = json.loads((Path(root) / "run_config.json").read_text())
    actual = digest(
        canonical(
            {
                k: v
                for k, v in config.items()
                if k not in ("iters", "passes", "resolved_iters")
            }
        )
    )
    if state.get("config_hash") != actual:
        raise ValueError("committed run configuration mismatch")
