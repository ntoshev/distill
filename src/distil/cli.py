"""Command-line interface."""

import argparse
import json
import sys

from rich.console import Console

from distil.render import RenderOptions, render, to_json_dict
from distil.rollout import load_rollout
from distil.scoring import score_text
from distil.templating import build_chat, build_raw, build_rollout


def _read_input(value: str | None) -> str | None:
    """Inline string, @path to read a file, or "-" for stdin."""
    if value is None:
        return None
    if value == "-":
        return sys.stdin.read()
    if value.startswith("@"):
        with open(value[1:]) as f:
            return f.read()
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="distil",
        description="Visualize per-token likelihood of a text/rollout according to an LLM "
        "(green = likely, red = surprising). Values support @file and '-' for stdin.",
    )
    p.add_argument("--model", required=True, help="mlx model path or HF repo id")
    src = p.add_argument_group("input (prompt/completion or rollout)")
    src.add_argument("--system", help="optional system prompt")
    src.add_argument("--prompt", help="prompt / context (not scored)")
    src.add_argument("--completion", help="text to score")
    src.add_argument("--raw", action="store_true", help="skip the chat template")
    src.add_argument("--rollout", help="verifiers/prime-rl results.jsonl")
    src.add_argument("--index", type=int, default=0, help="rollout row (default 0)")
    out = p.add_argument_group("output")
    out.add_argument("--metric", choices=["logprob", "prob"], default="logprob")
    out.add_argument("--stats", action="store_true", help="show color legend and summary stats")
    out.add_argument("--worst", type=int, default=0, help="table of N most surprising tokens")
    out.add_argument("--top-k", type=int, default=3, help="alternatives per surprising token")
    out.add_argument("--hide-context", action="store_true", help="do not echo unscored context")
    out.add_argument("--json", action="store_true", help="print JSON instead of colored text")
    out.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    out.add_argument("--width", type=int, help="force console width")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    force = True if args.color == "always" else None
    console = Console(
        force_terminal=force,
        no_color=args.color == "never",
        color_system=None if args.color == "never" else "truecolor",
        width=args.width,
    )
    errcon = Console(stderr=True)

    if bool(args.rollout) == bool(args.completion is not None):
        errcon.print("[red]error:[/red] provide either --completion or --rollout")
        return 2
    if args.raw and (args.system or args.rollout):
        errcon.print("[red]error:[/red] --raw only combines with --prompt/--completion")
        return 2

    meta = {}
    if args.rollout:
        try:
            rollout = load_rollout(args.rollout, args.index)
        except (FileNotFoundError, IndexError, ValueError) as e:
            errcon.print(f"[red]error:[/red] {e}")
            return 2
        meta = rollout.meta
    else:
        rollout = None
        completion = _read_input(args.completion)
        prompt = _read_input(args.prompt) or ""
        system = _read_input(args.system)

    from mlx_lm import load  # deferred: slow import

    model, tokenizer = load(args.model)

    if args.rollout:
        built = build_rollout(tokenizer, rollout.messages)
    else:
        if args.raw:
            built = build_raw(prompt, completion)
        else:
            built = build_chat(tokenizer, prompt, completion, system=system)

    result = score_text(
        model,
        tokenizer,
        built.text,
        built.spans,
        model_name=args.model,
        top_k=args.top_k,
        warnings=built.warnings,
    )

    if args.json:
        payload = to_json_dict(result)
        if meta:
            payload["rollout"] = meta
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for w in result.warnings:
        errcon.print(f"[yellow]warning:[/yellow] {w}")
    if meta:
        errcon.print("rollout: " + "  ".join(f"{k}={v}" for k, v in meta.items()), style="grey50")
    render(
        result,
        console,
        RenderOptions(
            metric=args.metric,
            hide_context=args.hide_context,
            stats=args.stats,
            worst=args.worst,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
