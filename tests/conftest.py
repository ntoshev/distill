import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable, TypeVar

import pytest

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
FIXTURE_DIR = Path(__file__).parent / "fixtures"
TIMING_BASELINES_PATH = Path(__file__).parent / "timing_baselines.json"
MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
MODEL_4B = "mlx-community/Qwen3.5-4B-6bit"
T = TypeVar("T")


def sample_skd_case(
    model,
    tokenizer,
    *,
    max_block_size=8,
    temperature=0.0,
    seed=0,
    batch_shared_model=True,
    teacher_without_adapter=False,
    on_decode_start=None,
    top_p=None,
):
    """Run the shared privileged-teacher SKD scenario used by tests."""
    from distil.sample import sample_speculative_completion
    from distil.templating import apply_teacher_context

    messages = [
        {"role": "user", "content": "What is Project Lumen's launch code?"}
    ]
    teacher_messages = apply_teacher_context(
        messages,
        system=(
            "Private reference: Project Lumen's launch code is VIOLET-731. "
            "Reply with it."
        ),
    )
    sample = sample_speculative_completion(
        model,
        tokenizer,
        messages,
        model,
        tokenizer,
        teacher_messages,
        **({"top_k": 3} if top_p is None else {"top_p": top_p}),
        max_tokens=16,
        temperature=temperature,
        seed=seed,
        enable_thinking=False,
        teacher_without_adapter=teacher_without_adapter,
        max_block_size=max_block_size,
        batch_shared_model=batch_shared_model,
        _on_decode_start=on_decode_start,
    )
    return sample, teacher_messages


def pytest_addoption(parser):
    selection = parser.getgroup("test selection")
    selection.addoption(
        "--run-slow",
        action="store_true",
        help="run tests marked slow",
    )

    group = parser.getgroup("timing benchmarks")
    group.addoption(
        "--timing-system-id",
        default=os.environ.get("DISTIL_BENCHMARK_SYSTEM_ID"),
        help="baseline system id used by tests marked benchmark",
    )
    group.addoption(
        "--timing-rounds",
        type=int,
        default=int(os.environ.get("DISTIL_BENCHMARK_ROUNDS", "7")),
        help="number of measured rounds per timing benchmark (default: 7)",
    )
    group.addoption(
        "--update-timing-baselines",
        action="store_true",
        default=os.environ.get("UPDATE_TIMING_BASELINES") == "1",
        help="record measured medians for the selected timing system id",
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="pass --run-slow to run slow tests")
    skip_benchmark = pytest.mark.skip(
        reason="set DISTIL_BENCHMARK_SYSTEM_ID or --timing-system-id to run timing benchmarks"
    )
    for item in items:
        if not config.getoption("--run-slow") and item.get_closest_marker("slow"):
            item.add_marker(skip_slow)
        if (
            not config.getoption("--timing-system-id")
            and item.get_closest_marker("benchmark")
        ):
            item.add_marker(skip_benchmark)


@dataclass(frozen=True)
class TimingMeasurement:
    median_seconds: float
    samples: tuple[float, ...]


class TimingBenchmark:
    def __init__(self, config):
        self.config = config
        self.system_id = config.getoption("--timing-system-id")
        self.rounds = config.getoption("--timing-rounds")
        self.update = config.getoption("--update-timing-baselines")
        if self.rounds < 1:
            raise pytest.UsageError("--timing-rounds must be positive")

    def __call__(
        self,
        name: str,
        operation: Callable[[], T],
        *,
        warmup_rounds: int = 2,
    ) -> tuple[T, TimingMeasurement]:
        baselines = json.loads(TIMING_BASELINES_PATH.read_text())
        systems = baselines.setdefault("systems", {})
        if self.update:
            system = systems.setdefault(self.system_id, {"benchmarks": {}})
        else:
            system = systems.get(self.system_id)
        if system is None:
            pytest.fail(
                f"no timing baselines for system id {self.system_id!r}; rerun with "
                "UPDATE_TIMING_BASELINES=1 to create them",
                pytrace=False,
            )
        benchmarks = system.setdefault("benchmarks", {})
        baseline = benchmarks.get(name)
        if baseline is None and not self.update:
            pytest.fail(
                f"no {name!r} timing baseline for system id {self.system_id!r}; rerun with "
                "UPDATE_TIMING_BASELINES=1 to create it",
                pytrace=False,
            )

        result = None
        for _ in range(warmup_rounds):
            result = operation()

        samples = []
        for _ in range(self.rounds):
            started = perf_counter()
            result = operation()
            samples.append(perf_counter() - started)

        measurement = TimingMeasurement(median(samples), tuple(samples))
        if self.update:
            tolerance = baseline.get("regression_tolerance", 0.20) if baseline else 0.20
            benchmarks[name] = {
                "median_seconds": round(measurement.median_seconds, 6),
                "regression_tolerance": tolerance,
            }
            TIMING_BASELINES_PATH.write_text(json.dumps(baselines, indent=2) + "\n")
            limit = measurement.median_seconds * (1 + tolerance)
        else:
            expected = baseline["median_seconds"]
            tolerance = baseline["regression_tolerance"]
            limit = expected * (1 + tolerance)
            assert measurement.median_seconds <= limit, (
                f"{name} regressed on {self.system_id}: median "
                f"{measurement.median_seconds:.3f}s > {limit:.3f}s "
                f"({expected:.3f}s baseline + {tolerance:.0%}); "
                f"samples={[round(sample, 3) for sample in samples]}"
            )

        reporter = self.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            action = "recorded" if self.update else "passed"
            reporter.write_line(
                f"timing {action}: {name} on {self.system_id}: "
                f"median {measurement.median_seconds:.3f}s, limit {limit:.3f}s"
            )
        return result, measurement

    def relative(
        self,
        name: str,
        operation: Callable[[], T],
        *,
        reference_name: str,
        reference_operation: Callable[[], object],
        units: Callable[[T], int] | None = None,
        reference_units: Callable[[object], int] | None = None,
        unit_name: str = "unit",
        duration: Callable[[T], float] | None = None,
        reference_duration: Callable[[object], float] | None = None,
        warmup_rounds: int = 2,
    ) -> tuple[T, TimingMeasurement]:
        """Compare an operation with a reference measured in the same run."""
        baselines = json.loads(TIMING_BASELINES_PATH.read_text())
        systems = baselines.setdefault("systems", {})
        if self.update:
            system = systems.setdefault(self.system_id, {"benchmarks": {}})
        else:
            system = systems.get(self.system_id)
        if system is None:
            pytest.fail(
                f"no timing baselines for system id {self.system_id!r}; rerun with "
                "UPDATE_TIMING_BASELINES=1 to create them",
                pytrace=False,
            )
        benchmarks = system.setdefault("benchmarks", {})
        baseline = benchmarks.get(name)
        if baseline is None and not self.update:
            pytest.fail(
                f"no {name!r} timing baseline for system id {self.system_id!r}; rerun "
                "with UPDATE_TIMING_BASELINES=1 to create it",
                pytrace=False,
            )

        def measure(fn, duration_fn=None, units_fn=None):
            result = None
            for _ in range(warmup_rounds):
                result = fn()

            samples = []
            for _ in range(self.rounds):
                started = perf_counter()
                result = fn()
                elapsed = perf_counter() - started
                elapsed = elapsed if duration_fn is None else duration_fn(result)
                if units_fn is not None:
                    unit_count = units_fn(result)
                    if unit_count <= 0:
                        raise ValueError(f"{unit_name} count must be positive")
                    elapsed /= unit_count
                samples.append(elapsed)
            return result, TimingMeasurement(median(samples), tuple(samples))

        _, reference = measure(
            reference_operation, reference_duration, reference_units
        )
        result, measurement = measure(operation, duration, units)
        measured_multiple = measurement.median_seconds / reference.median_seconds
        if self.update:
            tolerance = baseline.get("regression_tolerance", 0.20) if baseline else 0.20
            baseline_multiple = measured_multiple
            benchmarks[name] = {
                "reference": reference_name,
                "median_multiple": round(measured_multiple, 6),
                "regression_tolerance": tolerance,
            }
            TIMING_BASELINES_PATH.write_text(json.dumps(baselines, indent=2) + "\n")
        else:
            if baseline.get("reference") != reference_name:
                pytest.fail(
                    f"{name!r} baseline references {baseline.get('reference')!r}, "
                    f"expected {reference_name!r}",
                    pytrace=False,
                )
            baseline_multiple = baseline["median_multiple"]
            tolerance = baseline["regression_tolerance"]
        max_multiple = baseline_multiple * (1 + tolerance)
        limit = reference.median_seconds * max_multiple
        normalized = units is not None and reference_units is not None
        measured = (
            f"{1000 * measurement.median_seconds:.2f} ms/{unit_name}"
            if normalized
            else f"{measurement.median_seconds:.3f}s"
        )
        reference_measured = (
            f"{1000 * reference.median_seconds:.2f} ms/{unit_name}"
            if normalized
            else f"{reference.median_seconds:.3f}s"
        )
        limit_measured = (
            f"{1000 * limit:.2f} ms/{unit_name}"
            if normalized
            else f"{limit:.3f}s"
        )
        assert measurement.median_seconds <= limit, (
            f"{name} regressed relative to {reference_name} on {self.system_id}: "
            f"median {measured} > {limit_measured} "
            f"({baseline_multiple:.2f}x baseline + {tolerance:.0%}; "
            f"reference median {reference_measured}); "
            f"samples={[round(sample, 3) for sample in measurement.samples]}; "
            f"reference_samples="
            f"{[round(sample, 3) for sample in reference.samples]}"
        )

        reporter = self.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            action = "recorded" if self.update else "passed"
            reporter.write_line(
                f"timing {action}: {name} on {self.system_id}: "
                f"median {measured} = {measured_multiple:.2f}x "
                f"{reference_name} ({reference_measured}), "
                f"baseline {baseline_multiple:.2f}x + {tolerance:.0%}, "
                f"limit {max_multiple:.2f}x"
            )
        return result, measurement


@pytest.fixture
def timing_benchmark(pytestconfig):
    return TimingBenchmark(pytestconfig)


@pytest.fixture(scope="session")
def model_and_tokenizer():
    from mlx_lm import load

    return load(MODEL)


@pytest.fixture(scope="session")
def model(model_and_tokenizer):
    return model_and_tokenizer[0]


@pytest.fixture(scope="session")
def tokenizer(model_and_tokenizer):
    return model_and_tokenizer[1]


@pytest.fixture(scope="session")
def model_4b_and_tokenizer():
    from mlx_lm import load

    return load(MODEL_4B)


@pytest.fixture
def snapshot():
    """Compare a string against an on-disk snapshot in tests/snapshots/.

    Missing snapshots are created (and should be committed to git). Run with
    UPDATE_SNAPSHOTS=1 to regenerate after an intentional change.
    """

    def check(name: str, content: str):
        path = SNAPSHOT_DIR / name
        if os.environ.get("UPDATE_SNAPSHOTS") == "1" or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return
        expected = path.read_text()
        assert content == expected, (
            f"output differs from snapshot {path}; "
            "rerun with UPDATE_SNAPSHOTS=1 if the change is intentional"
        )

    return check


@pytest.fixture(scope='session')
def cartridge_model_and_tokenizer():
    from mlx_lm import load
    from mlx_lm.utils import _download
    from distil.cartridge import model_identity
    name='mlx-community/Qwen3-4B-Instruct-2507-4bit'
    path=_download(name,revision='50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b')
    model,tokenizer=load(path)
    identity=model_identity(model,tokenizer,name,model_path=path)
    return model,tokenizer,identity
