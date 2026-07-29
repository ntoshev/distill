"""Render scores and checkpoint progress to the terminal, and scores to JSON."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

from distil.scoring import ScoreResult, TokenScore

if TYPE_CHECKING:
    from distil.progress import ProgressResult

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
    worst: int = 0  # rows in the most-surprising-tokens / disagreements table (0 disables)
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
    """Render a checkpoint metric for several fixed completions."""
    label_width = min(30, max(len(probe.name) for probe in result.probes))
    chart_width = min(len(result.steps), max(12, console.width - label_width - 40))
    console.print(
        f"steps [bold]{result.steps[0]}[/bold] → [bold]{result.steps[-1]}[/bold]  "
        f"({len(result.steps)} checkpoints including base)"
    )
    labels = {
        "prob": "geometric-mean token probability",
        "logprob": "mean token log probability",
        "kl": "mean KL(teacher ‖ student)",
    }
    console.print(f"{labels[metric]} of each fixed completion", style="grey50")
    console.print()

    for probe in result.probes:
        points = result.series[probe.name]
        values = [_progress_value(point, metric) for point in points]
        levels = _downsample([_progress_level(point, metric) for point in points], chart_width)
        colors = _downsample(
            [point.mean_kl if metric == "kl" else point.mean_logprob for point in points],
            chart_width,
        )
        line = Text(f"{probe.name[:label_width]:<{label_width}}  ")
        for level, color in zip(levels, colors):
            index = min(len(_SPARK_CHARS) - 1, int(level * len(_SPARK_CHARS)))
            line.append(_SPARK_CHARS[index], style=color_for(color, metric))
        peak_index = max(range(len(values)), key=values.__getitem__)
        line.append(
            f"  {_progress_text(values[0], metric)} → {_progress_text(values[-1], metric)}  "
            f"max {_progress_text(values[peak_index], metric)}@{result.steps[peak_index]}"
        )
        console.print(line)


def render_progress_tokens(
    result: ProgressResult, console: Console, metric: str = "prob"
) -> None:
    """Render one fixed completion token-by-token at every checkpoint."""
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
    console.print(f"{labels[metric]} within the fixed completion", style="grey50")
    console.print()

    for point in points:
        label = "base" if point.step == 0 else f"step-{point.step:04d}"
        line = Text(f"{label:<{label_width}}  ")
        for token in point.tokens:
            value = token["kl"] if metric == "kl" else token["logprob"]
            line.append(token["text"], style=style_for(value, metric))
        line.append(
            f"  {_progress_text(_progress_value(point, metric), metric)}", style="grey50"
        )
        console.print(line)


def _token_value(tok: TokenScore, metric: str) -> float | None:
    return tok.kl if metric == "kl" else tok.logprob


def render(result: ScoreResult, console: Console, opts: RenderOptions | None = None) -> None:
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
            legend.append("  logprob" if opts.metric == "logprob" else "  (prob-scaled)")
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

    table = Table(title=f"{n} largest teacher/student disagreements", title_justify="left")
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
        payload["mean_teacher_logprob"] = round(result.mean_teacher_logprob, 3) if scored else None
        payload["mean_kl"] = round(result.mean_kl, 3) if scored else None
    payload["tokens"] = [
        {
            "text": t.text,
            "scored": t.scored,
            **({"logprob": round(t.logprob, ndigits)} if t.scored else {}),
            **(
                {"teacher_logprob": round(t.teacher_logprob, ndigits), "kl": round(t.kl, ndigits)}
                if t.scored and t.kl is not None
                else {}
            ),
            **(
                {"top": [[a.text, round(a.logprob, ndigits)] for a in t.top_k]}
                if t.scored and t.top_k
                else {}
            ),
            **(
                {"teacher_top": [[a.text, round(a.logprob, ndigits)] for a in t.teacher_top_k]}
                if t.scored and t.teacher_top_k
                else {}
            ),
        }
        for t in result.tokens
    ]
    return payload
