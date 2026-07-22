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
T = TypeVar("T")


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
