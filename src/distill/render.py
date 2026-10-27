"""Render scores and checkpoint progress to the terminal, and scores to JSON."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

from distill.scoring import ScoreResult, TokenScore

if TYPE_CHECKING:
    from distill.progress import ProgressResult

# Color stops for the likelihood gradient. Hex literals are intentional: many
# editors show inline color previews for them.
LP_LO = -10.0
EXTREME_LOGPROB = -7.5
_LOGPROB_STOPS = [
    (-10.0, "#ff1744"),  # off-scale: near-impossible
    (-8.0, "#c62828"),  # very surprising: red
    (-7.0, "#dc3f24"),  # tail detail: red-orange
    (-6.0, "#e85d19"),  # tail detail: hot orange
    (-4.0, "#f97316"),  # surprising: orange
    (-2.0, "#facc15"),  # somewhat unlikely: yellow
    (-1.0, "#d1d42f"),  # plausible: yellow-green
    (-0.5, "#65a30d"),  # likely: olive
    (-0.1, "#00a86b"),  # very likely: emerald
    (0.0, "#00c853"),  # expected: green
]
_PROB_STOPS = [
    (0.0, "#c62828"),
    (0.25, "#f97316"),
    (0.5, "#facc15"),
    (0.75, "#65a30d"),
    (0.9, "#00a86b"),
    (1.0, "#00c853"),
]
# KL(teacher ‖ student) in nats: 0 = the two distributions agree (green),
# growing disagreement runs the gradient in reverse toward red.
KL_HI = 10.0
EXTREME_KL = 7.5
_KL_STOPS = [
    (0.0, "#00c853"),  # agreement
    (0.05, "#00a86b"),
    (0.15, "#65a30d"),
    (0.4, "#d1d42f"),
    (1.0, "#facc15"),
    (2.0, "#f97316"),
    (4.0, "#e85d19"),
    (6.0, "#dc3f24"),
    (8.0, "#c62828"),
    (10.0, "#ff1744"),  # off-scale: severe disagreement
]

CONTEXT_STYLE = "grey50"


def _lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _interpolate_stops(value: float, stops: list[tuple[float, str]]) -> str:
    if value <= stops[0][0]:
        return stops[0][1]
    if value >= stops[-1][0]:
        return stops[-1][1]

    for (lo, lo_color), (hi, hi_color) in zip(stops, stops[1:]):
        if lo <= value <= hi:
            t = (value - lo) / (hi - lo)
            return _rgb_to_hex(_lerp(_hex_to_rgb(lo_color), _hex_to_rgb(hi_color), t))
    return stops[-1][1]


def _legend_style(logprob: float, metric: str) -> str:
    color = color_for(logprob, metric)
    r, g, b = _hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    foreground = "black" if luminance > 0.48 else "white"
    return f"bold {foreground} on {color}"


def color_for(value: float, metric: str = "logprob") -> str:
    """Map a per-token value to a hex color on the red↔green gradient.

    metric "logprob": position on the scale is the logprob clamped to [-10, 0].
    metric "prob": position is exp(logprob), linear in probability.
    metric "kl": KL(teacher ‖ student) in nats clamped to [0, 10], reversed
    (0 = agreement = green).
    """
    if metric == "prob":
        t = min(max(math.exp(value), 0.0), 1.0)
        return _interpolate_stops(t, _PROB_STOPS)
    elif metric == "kl":
        return _interpolate_stops(min(max(value, 0.0), KL_HI), _KL_STOPS)
    else:
        return _interpolate_stops(min(max(value, LP_LO), 0.0), _LOGPROB_STOPS)


def style_for(
    value: float, metric: str = "logprob", *, teacher_sampled: bool = False
) -> str:
    """Terminal style for a scored token."""
    color = color_for(value, metric)
    extreme = value >= EXTREME_KL if metric == "kl" else value <= EXTREME_LOGPROB
    if teacher_sampled:
        emphasis = "bold underline2" if extreme else "underline2"
    else:
        emphasis = ""
    return f"{emphasis} {color}" if emphasis else color


def render_skd_rollout(rollout: dict, console: Console, *, label: str = "skd") -> None:
    """Render one trained rollout using privileged-teacher token likelihoods."""
    line = Text(no_wrap=True, overflow="ellipsis")
    line.append(f"{label}  ", style="grey50")
    line.append("Q: ", style="grey50")
    line.append(" ".join(str(rollout.get("prompt", "")).split()), style="bold")
    line.append("  →  ", style="grey50")

    feedback = rollout.get("teacher_feedback") or []
    logprobs = []
    for token in feedback:
        text = str(token.get("text", "")).replace("\r", "").replace("\n", "↵")
        logprob = token.get("logprob")
        if not text or logprob is None:
            continue
        value = float(logprob)
        logprobs.append(value)
        line.append(
            text,
            style=style_for(
                value,
                teacher_sampled=bool(token.get("teacher_sampled")),
            ),
        )

    if not logprobs:
        line.append(
            " ".join(str(rollout.get("content", "")).split()),
            style="grey70",
        )
    else:
        mean_probability = math.exp(sum(logprobs) / len(logprobs))
        line.append(f"  teacher p={mean_probability:.1%}", style="grey50")
    console.print(line, soft_wrap=False)


@dataclass
class RenderOptions:
    metric: str = "logprob"  # logprob | prob | kl (kl needs teacher-mode scores)
    hide_context: bool = False
    stats: bool = False  # legend + summary line
    worst: int = (
        0  # rows in the most-surprising-tokens / disagreements table (0 disables)
    )
    teacher_sampled: list[bool] | None = None  # provenance of scored sample tokens


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _downsample(values: list[float], width: int) -> list[float]:
    if len(values) <= width:
        return values
    sampled = []
    for column in range(width):
        start = column * len(values) // width
        end = (column + 1) * len(values) // width
        sampled.append(sum(values[start:end]) / (end - start))
    return sampled


def _progress_value(point, metric: str) -> float:
    if metric == "kl":
        return point.mean_kl
    if metric == "logprob":
        return point.mean_logprob
    return point.mean_probability


def _progress_level(point, metric: str) -> float:
    if metric == "kl":
        return 1 - math.exp(-max(point.mean_kl, 0.0))
    return point.mean_probability


def _progress_text(value: float, metric: str) -> str:
    return f"{value:6.1%}" if metric == "prob" else f"{value:7.3f}"


def render_progress(
    result: ProgressResult, console: Console, metric: str = "prob"
) -> None:
    """Render checkpoint probe rows token-by-token."""
    layout = render_progress_rows_header(result, console, metric)
    for probe in result.probes:
        points = result.series[probe.name]
        for point in points:
            render_progress_row_point(probe, point, console, metric, layout)


def render_progress_tokens_header(
    result: ProgressResult,
    console: Console,
    metric: str = "prob",
    *,
    sampled: bool | None = None,
) -> tuple[int, bool]:
    if len(result.probes) != 1:
        raise ValueError("token progress rendering requires exactly one probe")
    probe = result.probes[0]
    points = result.series[probe.name]
    label_width = max(4, len(f"step-{result.steps[-1]:04d}"))
    labels = {
        "prob": "token probability",
        "logprob": "token log probability",
        "kl": "token KL(teacher ‖ student)",
    }
    sampled = (
        any(point.completion is not None for point in points)
        if sampled is None
        else sampled
    )
    subject = "sampled completion" if sampled else "fixed completion"
    console.print(f"{labels[metric]} within the {subject}", style="grey50")
    if sampled:
        has_teacher = result.teacher_model is not None or any(
            point.mean_teacher_logprob is not None for point in points
        )
        metric_columns = "teacher p  student p  KL      tokens  finish"
        if not has_teacher:
            metric_columns = "student p  tokens  finish"
        console.print(
            f"{'step':<{label_width}}  completion  {metric_columns}", style="grey50"
        )
    console.print()
    return label_width, sampled


def render_progress_token_point(
    point,
    console: Console,
    metric: str = "prob",
    *,
    label_width: int,
    sampled: bool,
) -> None:
    label = getattr(point, "label", None) or (
        "base" if point.step == 0 else f"step-{point.step:04d}"
    )
    line = Text(f"{label:<{label_width}}  ")
    for token in point.tokens:
        if metric == "kl":
            value = token["kl"]
        else:
            value = token["logprob"]
        line.append(token["text"], style=style_for(value, metric))
    if sampled:
        student_p = math.exp(point.mean_logprob)
        if point.mean_teacher_logprob is not None:
            teacher_p = math.exp(point.mean_teacher_logprob)
            kl = point.mean_kl if point.mean_kl is not None else float("nan")
            line.append(
                f"  teacher p={teacher_p:5.1%}  student p={student_p:5.1%}  "
                f"KL={kl:6.3f}",
                style="grey50",
            )
        else:
            line.append(f"  student p={student_p:5.1%}", style="grey50")
        if point.num_tokens is not None:
            line.append(f"  n={point.num_tokens}", style="grey50")
        if point.finish_reason:
            line.append(f"  {point.finish_reason}", style="grey50")
    else:
        line.append(
            f"  {_progress_text(_progress_value(point, metric), metric)}",
            style="grey50",
        )
    console.print(line)


def render_progress_rows_header(
    result: ProgressResult,
    console: Console,
    metric: str = "prob",
    *,
    sampled: bool | None = None,
) -> dict:
    if len(result.probes) == 1:
        label_width, sampled = render_progress_tokens_header(
            result, console, metric, sampled=sampled
        )
        return {"single": True, "label_width": label_width, "sampled": sampled}

    step_width = max(4, len(f"step-{result.steps[-1]:04d}"))
    probe_width = min(30, max(len(probe.name) for probe in result.probes))
    labels = {
        "prob": "student token probability",
        "logprob": "student token log probability",
        "kl": "token KL(teacher ‖ student)",
    }
    console.print(f"{labels[metric]} over checkpoint probes", style="grey50")
    console.print(
        f"{'step':<{step_width}}  {'probe':<{probe_width}}  completion  "
        "teacher p  student p  KL",
        style="grey50",
    )
    console.print()
    return {
        "single": False,
        "step_width": step_width,
        "probe_width": probe_width,
        "sampled": bool(sampled),
    }


def render_progress_row_point(
    probe,
    point,
    console: Console,
    metric: str,
    layout: dict,
) -> None:
    if layout["single"]:
        render_progress_token_point(
            point,
            console,
            metric,
            label_width=layout["label_width"],
            sampled=layout["sampled"],
        )
        return

    label = getattr(point, "label", None) or (
        "base" if point.step == 0 else f"step-{point.step:04d}"
    )
    line = Text(
        f"{label:<{layout['step_width']}}  "
        f"{probe.name[: layout['probe_width']]:<{layout['probe_width']}}  "
    )
    for token in point.tokens:
        value = token["kl"] if metric == "kl" else token["logprob"]
        line.append(token["text"], style=style_for(value, metric))
    if point.mean_teacher_logprob is not None:
        line.append(
            f"  teacher p={math.exp(point.mean_teacher_logprob):5.1%}  "
            f"student p={math.exp(point.mean_logprob):5.1%}  "
            f"KL={point.mean_kl:6.3f}",
            style="grey50",
        )
    else:
        line.append(f"  student p={math.exp(point.mean_logprob):5.1%}", style="grey50")
    console.print(line)


def render_progress_tokens(
    result: ProgressResult, console: Console, metric: str = "prob"
) -> None:
    """Render one fixed completion token-by-token at every checkpoint."""
    if len(result.probes) != 1:
        raise ValueError("token progress rendering requires exactly one probe")
    probe = result.probes[0]
    points = result.series[probe.name]
    label_width, sampled = render_progress_tokens_header(result, console, metric)

    for point in points:
        render_progress_token_point(
            point, console, metric, label_width=label_width, sampled=sampled
        )


def _token_value(tok: TokenScore, metric: str) -> float | None:
    return tok.kl if metric == "kl" else tok.logprob


def render(
    result: ScoreResult, console: Console, opts: RenderOptions | None = None
) -> None:
    opts = opts or RenderOptions()

    body = Text()
    scored_index = 0
    for tok in result.tokens:
        value = _token_value(tok, opts.metric)
        if tok.scored and value is not None:
            from_teacher = bool(
                opts.teacher_sampled
                and scored_index < len(opts.teacher_sampled)
                and opts.teacher_sampled[scored_index]
            )
            body.append(
                tok.text,
                style=style_for(value, opts.metric, teacher_sampled=from_teacher),
            )
            scored_index += 1
        elif not opts.hide_context:
            body.append(tok.text, style=CONTEXT_STYLE)
    console.print(body, soft_wrap=True, end="" if body.plain.endswith("\n") else "\n")

    scored = result.scored
    if not scored:
        console.print("[red]no tokens were scored[/red]")
        return

    if opts.stats:
        console.print()
        legend = Text("scale  ")
        if opts.metric == "kl":
            for kl in (0, 0.05, 0.15, 0.4, 1, 2, 4, 6, 8):
                legend.append(f" {kl:g} ", style=_legend_style(kl, opts.metric))
            legend.append("  KL(teacher ‖ student), nats")
        else:
            for lp in (-10, -8, -7, -6, -4, -2, -1, -0.5, -0.1):
                legend.append(f" {lp:g} ", style=_legend_style(lp, opts.metric))
            legend.append(
                "  logprob" if opts.metric == "logprob" else "  (prob-scaled)"
            )
        console.print(legend)
        line = (
            f"tokens scored: [bold]{len(scored)}[/bold] of {len(result.tokens)}   "
            f"mean logprob: [bold]{result.mean_logprob:.4f}[/bold]   "
            f"perplexity: [bold]{result.perplexity:.3f}[/bold]"
        )
        if result.teacher_model_name:
            line += (
                f"   teacher mean logprob: [bold]{result.mean_teacher_logprob:.4f}[/bold]   "
                f"mean KL: [bold]{result.mean_kl:.4f}[/bold]"
            )
        console.print(line)

    if opts.worst > 0:
        console.print()
        if opts.metric == "kl":
            console.print(_disagreement_table(scored, opts.worst))
        else:
            console.print(_worst_table(scored, opts.worst))


def _worst_table(scored: list[TokenScore], n: int) -> Table:
    table = Table(title=f"{n} most surprising tokens", title_justify="left")
    table.add_column("token", style="bold")
    table.add_column("logprob", justify="right")
    table.add_column("prob", justify="right")
    table.add_column("model preferred")
    for tok in sorted(scored, key=lambda t: t.logprob)[:n]:
        alts = ""
        if tok.top_k:
            alts = "  ".join(f"{a.text!r} {math.exp(a.logprob):.0%}" for a in tok.top_k)
        table.add_row(
            repr(tok.text),
            f"{tok.logprob:.3f}",
            f"{math.exp(tok.logprob):.2%}",
            alts,
        )
    return table


def _disagreement_table(scored: list[TokenScore], n: int) -> Table:
    """Positions where the teacher's and student's distributions differ most."""

    def preferred(alts):
        return f"{alts[0].text!r} {math.exp(alts[0].logprob):.0%}" if alts else ""

    table = Table(
        title=f"{n} largest teacher/student disagreements", title_justify="left"
    )
    table.add_column("token", style="bold")
    table.add_column("KL", justify="right")
    table.add_column("student lp", justify="right")
    table.add_column("teacher lp", justify="right")
    table.add_column("student preferred")
    table.add_column("teacher preferred")
    ranked = [t for t in scored if t.kl is not None]
    for tok in sorted(ranked, key=lambda t: -t.kl)[:n]:
        table.add_row(
            repr(tok.text),
            f"{tok.kl:.3f}",
            f"{tok.logprob:.3f}",
            f"{tok.teacher_logprob:.3f}",
            preferred(tok.top_k),
            preferred(tok.teacher_top_k),
        )
    return table


def to_json_dict(result: ScoreResult, ndigits: int = 2) -> dict:
    """Machine-readable (and snapshot-friendly) form; logprobs rounded."""
    scored = result.scored
    teacher = bool(result.teacher_model_name)
    payload = {
        "model": result.model_name,
        **({"teacher_model": result.teacher_model_name} if teacher else {}),
        "warnings": result.warnings,
        "num_tokens": len(result.tokens),
        "num_scored": len(scored),
        "mean_logprob": round(result.mean_logprob, 3) if scored else None,
        "perplexity": round(result.perplexity, 2) if scored else None,
    }
    if teacher:
        payload["mean_teacher_logprob"] = (
            round(result.mean_teacher_logprob, 3) if scored else None
        )
        payload["mean_kl"] = round(result.mean_kl, 3) if scored else None
    payload["tokens"] = [
        {
            "text": t.text,
            "scored": t.scored,
            **({"logprob": round(t.logprob, ndigits)} if t.scored else {}),
            **(
                {
                    "teacher_logprob": round(t.teacher_logprob, ndigits),
                    "kl": round(t.kl, ndigits),
                }
                if t.scored and t.kl is not None
                else {}
            ),
            **(
                {"top": [[a.text, round(a.logprob, ndigits)] for a in t.top_k]}
                if t.scored and t.top_k
                else {}
            ),
            **(
                {
                    "teacher_top": [
                        [a.text, round(a.logprob, ndigits)] for a in t.teacher_top_k
                    ]
                }
                if t.scored and t.teacher_top_k
                else {}
            ),
        }
        for t in result.tokens
    ]
    return payload


def checkpoint_grid(result, metric="prob", *, show_special_tokens=False, indices=None):
    """One checkpoint per row; color only the tokens actually stored in that cell."""
    table = Table(box=None, padding=(0, 1), collapse_padding=True, show_edge=False)
    table.add_column("step", style="grey50", no_wrap=True)
    for probe in result.probes:
        generated = probe.dynamic or any(
            p.completion is not None for p in result.series[probe.name]
        )
        table.add_column(
            f"{probe.name} · {'generated' if generated else 'scored'}", overflow="fold"
        )
    complete = min(
        [len(result.steps), *(len(result.series[p.name]) for p in result.probes)]
    )
    for row_index in (
        range(complete) if indices is None else (i for i in indices if i < complete)
    ):
        step = result.steps[row_index]
        points = [result.series[p.name][row_index] for p in result.probes]
        transition = points[0].label
        label = (
            "base"
            if transition == "base" or step in {-1, 0} and transition is None
            else str(step)
        )
        if transition and transition != "base":
            label += " " + transition
        cells = []
        for point in points:
            cell = Text()
            for token in point.tokens:
                special = token.get(
                    "special",
                    token["text"] in {"<|im_end|>", "<|endoftext|>", "</s>", "<eos>"},
                )
                if special and not show_special_tokens:
                    continue
                value = token.get("kl") if metric == "kl" else token["logprob"]
                if value is None:
                    raise ValueError(
                        "KL display requires teacher scores for every displayed token"
                    )
                cell.append(token["text"], style=style_for(value, metric))
            cell.append(f"  p={point.mean_probability:.1%}", style="grey50")
            cells.append(cell)
        table.add_row(label, *cells)
    return table


GRID_LEGEND = "p = geometric-mean student probability, including generated end tokens."


def render_checkpoint_grid(
    result, console, metric="prob", *, show_special_tokens=False
):
    console.print(
        checkpoint_grid(result, metric, show_special_tokens=show_special_tokens)
    )
    console.print(GRID_LEGEND, style="grey50")


class CheckpointDisplay:
    """Shared live grid and status line; redirected output gets one final grid."""

    def __init__(
        self, console, *, metric="prob", show_special_tokens=False, enabled=True
    ):
        self.console = console
        self.metric = metric
        self.show_special_tokens = show_special_tokens
        self.enabled = enabled
        self.result = None
        self.status = "Evaluating base model…"
        self.live = None

    def start(self):
        from rich.live import Live

        if self.enabled and getattr(self.console.file, "isatty", lambda: False)():
            self.live = Live(
                self._renderable(), console=self.console, refresh_per_second=4
            )
            self.live.start()
        return self

    def _renderable(self):
        from rich.console import Group

        parts = []
        if self.result and self.result.steps:
            # Keep the base and most recent checkpoints visible in a short terminal.
            budget = max(2, self.console.height - 8)
            indices = list(range(len(self.result.steps)))
            if self.live and len(indices) > budget:
                indices = [0, *indices[-(budget - 1) :]]
            parts.append(
                checkpoint_grid(
                    self.result,
                    self.metric,
                    show_special_tokens=self.show_special_tokens,
                    indices=indices,
                )
            )
            parts.append(Text(GRID_LEGEND, style="grey50"))
        parts.append(Text(self.status, style="grey50"))
        return Group(*parts)

    def update(self, result=None, status=None):
        if result is not None:
            self.result = result
        if status is not None:
            self.status = status
        if self.live:
            self.live.update(self._renderable(), refresh=True)

    def stop(self):
        if self.live:
            self.live.stop()
            self.live = None
        elif self.enabled and self.result:
            self.console.print(self._renderable())
