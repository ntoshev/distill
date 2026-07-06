"""Score every token of a text with a single forward pass of an mlx-lm model."""

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TopToken:
    token_id: int
    text: str
    logprob: float


@dataclass
class TokenScore:
    token_id: int
    text: str  # decoded piece of the full text
    start: int  # char offsets into the full text
    end: int
    scored: bool  # falls inside a scored span (and has a preceding context)
    logprob: float | None  # None only for position 0
    top_k: list[TopToken] | None = None  # alternatives, filled for scored tokens


@dataclass
class ScoreResult:
    text: str
    tokens: list[TokenScore]
    model_name: str
    warnings: list[str]

    @property
    def scored(self) -> list[TokenScore]:
        return [t for t in self.tokens if t.scored]

    @property
    def mean_logprob(self) -> float:
        s = self.scored
        return sum(t.logprob for t in s) / len(s) if s else float("nan")

    @property
    def perplexity(self) -> float:
        return math.exp(-self.mean_logprob)


def tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize `text` (which may contain special tokens rendered by a chat
    template) and return per-token character offsets."""
    try:
        enc = tokenizer._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    except Exception:
        pass
    # fallback: walk the streaming detokenizer and accumulate offsets
    ids = tokenizer.encode(text, add_special_tokens=False)
    detok = tokenizer.detokenizer
    detok.reset()
    offsets, pos = [], 0
    for tid in ids:
        detok.add_token(tid)
        end = len(detok.text)
        offsets.append((pos, end))
        pos = end
    detok.finalize()
    return ids, offsets


def score_ids(
    model,
    ids: list[int],
    alt_positions: set[int] | None = None,
    top_k: int = 3,
    chunk_size: int = 512,
) -> tuple[dict[int, float], dict[int, list[tuple[int, float]]]]:
    """Single forward pass over `ids`.

    Returns (logprobs, alternatives): logprobs[t] = log p(ids[t] | ids[:t]) for
    every t >= 1, and alternatives[t] = top_k (token_id, logprob) pairs of the
    predictive distribution at the positions in `alt_positions`.
    """
    alt_positions = alt_positions or set()
    logits = model(mx.array(ids)[None])[0]  # (seq, vocab)
    ids_arr = mx.array(ids)
    logprobs: dict[int, float] = {}
    alternatives: dict[int, list[tuple[int, float]]] = {}

    # each row t of `logits` predicts token t+1; process in chunks to avoid
    # materializing a float32 (seq, vocab) log-softmax all at once
    for row_start in range(0, len(ids) - 1, chunk_size):
        row_end = min(row_start + chunk_size, len(ids) - 1)
        rows = logits[row_start:row_end].astype(mx.float32)
        lse = mx.logsumexp(rows, axis=-1)
        chosen = mx.take_along_axis(
            rows, ids_arr[row_start + 1 : row_end + 1][:, None], axis=-1
        )[:, 0]
        for j, lp in enumerate((chosen - lse).tolist()):
            logprobs[row_start + j + 1] = lp

        alt_rows = [t - 1 - row_start for t in alt_positions if row_start <= t - 1 < row_end]
        if top_k > 0 and alt_rows:
            sub = rows[mx.array(alt_rows)]
            top_ids = mx.argpartition(-sub, kth=top_k - 1, axis=-1)[:, :top_k]
            top_lp = mx.take_along_axis(sub, top_ids, axis=-1) - lse[mx.array(alt_rows)][:, None]
            for j, r in enumerate(alt_rows):
                pairs = sorted(zip(top_ids[j].tolist(), top_lp[j].tolist()), key=lambda p: -p[1])
                alternatives[row_start + r + 1] = pairs
    return logprobs, alternatives


def score_text(
    model,
    tokenizer,
    text: str,
    spans: list[tuple[int, int]],
    model_name: str = "",
    top_k: int = 3,
    warnings: list[str] | None = None,
) -> ScoreResult:
    """Run one forward pass and attach log p(token_t | tokens_<t) to each token.

    A token is scored when its character range overlaps one of `spans`.
    Position 0 has nothing to condition on and is never scored.
    """
    warnings = list(warnings or [])
    ids, offsets = tokenize_with_offsets(tokenizer, text)
    if tokenizer.decode(ids) != text:
        warnings.append("tokenization does not round-trip the rendered text; offsets may be off")

    def in_span(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in spans)

    scored_set = {t for t in range(1, len(ids)) if in_span(*offsets[t])}
    logprobs, alternatives = score_ids(model, ids, alt_positions=scored_set, top_k=top_k)

    tokens = []
    for t, (tid, (start, end)) in enumerate(zip(ids, offsets)):
        alts = alternatives.get(t)
        tokens.append(
            TokenScore(
                token_id=tid,
                text=text[start:end],
                start=start,
                end=end,
                scored=t in scored_set,
                logprob=logprobs.get(t),
                top_k=[TopToken(a, tokenizer.decode([a]), lp) for a, lp in alts] if alts else None,
            )
        )
    return ScoreResult(text=text, tokens=tokens, model_name=model_name, warnings=warnings)
