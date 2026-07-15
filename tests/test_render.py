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


def make_teacher_result() -> ScoreResult:
    """Synthetic two-pass result: same tokens, teacher fields filled."""
    result = make_result()
    kls = [0.01, 0.1, 0.8, 2.5, 9.0]
    t_lps = [-0.04, -0.3, -0.2, -1.0, -0.5]
    for tok, kl, t_lp in zip(result.scored, kls, t_lps):
        tok.kl = kl
        tok.teacher_logprob = t_lp
        tok.teacher_top_k = [TopToken(97, " indeed", -0.2)]
    result.teacher_model_name = "teacher-model"
    return result


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
    assert "teacher_model" not in d and "mean_kl" not in d
    json.dumps(d)  # serializable


def test_kl_color_endpoints():
    assert color_for(0.0, metric="kl") == "#00c853"  # agreement is green
    assert color_for(8.0, metric="kl") == "#c62828"
    assert color_for(10.0, metric="kl") == "#ff1744"
    assert color_for(100.0, metric="kl") == color_for(10.0, metric="kl")  # clamped
    assert color_for(-0.001, metric="kl") == color_for(0.0, metric="kl")  # numeric jitter


def test_kl_color_monotonic():
    # redness minus greenness strictly rises with disagreement
    def r_minus_g(kl):
        c = color_for(kl, metric="kl")
        return int(c[1:3], 16) - int(c[3:5], 16)

    vals = [r_minus_g(kl) for kl in (0, 0.05, 0.15, 0.4, 1, 2, 4, 6, 8, 10)]
    assert vals == sorted(vals)
    assert len(set(vals)) == len(vals)


def test_extreme_kl_style_is_emphasized():
    assert style_for(7.49, metric="kl") == color_for(7.49, metric="kl")
    assert style_for(7.5, metric="kl") == f"bold underline {color_for(7.5, metric='kl')}"


def test_teacher_render_snapshot(snapshot):
    console = Console(record=True, force_terminal=True, color_system="truecolor", width=100)
    render(make_teacher_result(), console, RenderOptions(metric="kl", stats=True, worst=3))
    snapshot("render_teacher_kl.ansi", console.export_text(styles=True))


def test_teacher_json_dict():
    d = to_json_dict(make_teacher_result())
    assert d["teacher_model"] == "teacher-model"
    assert d["mean_kl"] == round((0.01 + 0.1 + 0.8 + 2.5 + 9.0) / 5, 3)
    assert d["mean_teacher_logprob"] is not None
    tok = d["tokens"][1]
    assert tok["kl"] == 0.01 and tok["teacher_logprob"] == -0.04
    assert tok["teacher_top"] == [[" indeed", -0.2]]
    json.dumps(d)  # serializable
