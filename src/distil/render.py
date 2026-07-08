"""Render a ScoreResult to the terminal with rich, and to JSON."""

import math
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.text import Text

from distil.scoring import ScoreResult, TokenScore

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


def color_for(logprob: float, metric: str = "logprob") -> str:
    """Map a token logprob to a hex color on the red->orange->yellow->green gradient.

    metric "logprob": position on the scale is the logprob clamped to [-10, 0].
    metric "prob": position is exp(logprob), linear in probability.
    """
    if metric == "prob":
        t = min(max(math.exp(logprob), 0.0), 1.0)
        return _interpolate_stops(t, _PROB_STOPS)
    else:
        return _interpolate_stops(min(max(logprob, LP_LO), 0.0), _LOGPROB_STOPS)


def style_for(logprob: float, metric: str = "logprob") -> str:
    """Terminal style for a scored token."""
    color = color_for(logprob, metric)
    if logprob <= EXTREME_LOGPROB:
        return f"bold underline {color}"
    return color


@dataclass
class RenderOptions:
    metric: str = "logprob"  # logprob | prob
    hide_context: bool = False
    stats: bool = False  # legend + summary line
    worst: int = 0  # rows in the most-surprising-tokens table (0 disables)


def render(result: ScoreResult, console: Console, opts: RenderOptions | None = None) -> None:
    opts = opts or RenderOptions()

    body = Text()
    for tok in result.tokens:
        if tok.scored:
            body.append(tok.text, style=style_for(tok.logprob, opts.metric))
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
        for lp in (-10, -8, -7, -6, -4, -2, -1, -0.5, -0.1):
            legend.append(f" {lp:g} ", style=_legend_style(lp, opts.metric))
        legend.append("  logprob" if opts.metric == "logprob" else "  (prob-scaled)")
        console.print(legend)
        console.print(
            f"tokens scored: [bold]{len(scored)}[/bold] of {len(result.tokens)}   "
            f"mean logprob: [bold]{result.mean_logprob:.4f}[/bold]   "
            f"perplexity: [bold]{result.perplexity:.3f}[/bold]"
        )

    if opts.worst > 0:
        console.print()
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


def to_json_dict(result: ScoreResult, ndigits: int = 2) -> dict:
    """Machine-readable (and snapshot-friendly) form; logprobs rounded."""
    scored = result.scored
    return {
        "model": result.model_name,
        "warnings": result.warnings,
        "num_tokens": len(result.tokens),
        "num_scored": len(scored),
        "mean_logprob": round(result.mean_logprob, 3) if scored else None,
        "perplexity": round(result.perplexity, 2) if scored else None,
        "tokens": [
            {
                "text": t.text,
                "scored": t.scored,
                **({"logprob": round(t.logprob, ndigits)} if t.scored else {}),
                **(
                    {"top": [[a.text, round(a.logprob, ndigits)] for a in t.top_k]}
                    if t.scored and t.top_k
                    else {}
                ),
            }
            for t in result.tokens
        ],
    }
