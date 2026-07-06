"""Render a ScoreResult to the terminal with rich, and to JSON."""

import math
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.text import Text

from distil.scoring import ScoreResult, TokenScore

# color stops for the likelihood gradient (logprob clamped to [LP_LO, 0]):
# red (surprising) -> yellow -> green (expected)
LP_LO = -8.0
_STOPS = [(200, 40, 40), (210, 180, 40), (40, 180, 70)]

CONTEXT_STYLE = "grey50"


def _lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def color_for(logprob: float, metric: str = "logprob") -> str:
    """Map a token logprob to a hex color on the red->yellow->green gradient.

    metric "logprob": position on the scale is the logprob clamped to [-8, 0].
    metric "prob": position is exp(logprob), linear in probability.
    """
    if metric == "prob":
        t = math.exp(logprob)
    else:
        t = 1.0 - min(max(logprob, LP_LO), 0.0) / LP_LO
    t = min(max(t, 0.0), 1.0)
    rgb = _lerp(_STOPS[0], _STOPS[1], t * 2) if t < 0.5 else _lerp(_STOPS[1], _STOPS[2], t * 2 - 1)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


@dataclass
class RenderOptions:
    metric: str = "logprob"  # logprob | prob
    hide_context: bool = False
    worst: int = 10  # rows in the most-surprising-tokens table (0 disables)
    show_legend: bool = True


def render(result: ScoreResult, console: Console, opts: RenderOptions | None = None) -> None:
    opts = opts or RenderOptions()
    for w in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")

    body = Text()
    for tok in result.tokens:
        if tok.scored:
            body.append(tok.text, style=color_for(tok.logprob, opts.metric))
        elif not opts.hide_context:
            body.append(tok.text, style=CONTEXT_STYLE)
    console.print(body)
    console.print()

    scored = result.scored
    if not scored:
        console.print("[red]no tokens were scored[/red]")
        return

    if opts.show_legend:
        legend = Text("scale  ")
        for lp in (-8, -6, -4, -2, -1, -0.5, -0.1):
            legend.append(f" {lp:g} ", style=f"bold white on {color_for(lp, opts.metric)}")
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
