"""Unit tests for color mapping, rendering, and JSON output. No model needed."""

import json

from rich.console import Console

from distil.render import RenderOptions, color_for, render, style_for, to_json_dict
from distil.scoring import ScoreResult, TokenScore, TopToken


def make_result() -> ScoreResult:
    pieces = [
        ("The prompt. ", None),
        ("A", -0.05),
        (" very", -0.4),
        (" likely", -1.2),
        (" sentence", -3.5),
        ("!", -7.9),
    ]
    tokens, pos = [], 0
    for i, (text, lp) in enumerate(pieces):
        tokens.append(
            TokenScore(
                token_id=i,
                text=text,
                start=pos,
                end=pos + len(text),
                scored=lp is not None,
                logprob=lp,
                top_k=[TopToken(99, " the", -0.3), TopToken(98, " a", -1.9)]
                if lp is not None and lp < -3
                else None,
            )
        )
        pos += len(text)
    return ScoreResult(
        text="".join(p for p, _ in pieces),
        tokens=tokens,
        model_name="test-model",
        warnings=["example warning"],
    )


def test_color_endpoints():
    assert color_for(0.0) == "#00c853"  # green at certainty
    assert color_for(-10.0) == "#ff1744"  # hot red at the clamp floor
    assert color_for(-8.0) == "#c62828"  # red in the low-probability tail
    assert color_for(-6.0) == "#e85d19"  # hot orange before the midpoint
    assert color_for(-4.0) == "#f97316"  # orange midpoint
    assert color_for(-100.0) == color_for(-10.0)  # clamped
    assert color_for(0.0, metric="prob") == "#00c853"
    assert color_for(-20.0, metric="prob") == color_for(-8.0)


def test_extreme_logprob_style_is_emphasized():
    assert style_for(-7.49) == color_for(-7.49)
    assert style_for(-7.5) == f"bold underline {color_for(-7.5)}"


def test_high_likelihood_colors_are_distinct():
    assert color_for(-1.0) == "#d1d42f"
    assert color_for(-0.5) == "#65a30d"
    assert color_for(-0.1) == "#00a86b"


def test_color_monotonic():
    # redness minus greenness strictly falls as likelihood rises
    def r_minus_g(lp):
        c = color_for(lp)
        return int(c[1:3], 16) - int(c[3:5], 16)

    vals = [r_minus_g(lp) for lp in (-10, -8, -6, -4, -2, -1, -0.1)]
    assert vals == sorted(vals, reverse=True)
    assert len(set(vals)) == len(vals)


def test_render_snapshot(snapshot):
    console = Console(record=True, force_terminal=True, color_system="truecolor", width=80)
    render(make_result(), console, RenderOptions(stats=True, worst=3))
    snapshot("render_synthetic.ansi", console.export_text(styles=True))


def test_render_default_is_body_only():
    console = Console(record=True, force_terminal=True, color_system="truecolor", width=80)
    render(make_result(), console)
    out = console.export_text()
    assert out == "The prompt. A very likely sentence!\n"


def test_render_hide_context():
    console = Console(record=True, force_terminal=True, color_system="truecolor", width=80)
    render(make_result(), console, RenderOptions(hide_context=True))
    assert "The prompt." not in console.export_text()


def test_json_dict():
    d = to_json_dict(make_result())
    assert d["num_scored"] == 5 and d["num_tokens"] == 6
    assert d["tokens"][0] == {"text": "The prompt. ", "scored": False}
    assert d["tokens"][1] == {"text": "A", "scored": True, "logprob": -0.05}
    assert d["tokens"][4]["top"] == [[" the", -0.3], [" a", -1.9]]
    json.dumps(d)  # serializable
