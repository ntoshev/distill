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
    # teacher-mode extras (two-pass scoring), None in single-model runs
    teacher_logprob: float | None = None
    kl: float | None = None  # exact KL(teacher ‖ student) at this position, nats
    teacher_top_k: list[TopToken] | None = None


@dataclass
class ScoreResult:
    text: str
    tokens: list[TokenScore]
    model_name: str
    warnings: list[str]
    teacher_model_name: str = ""  # non-empty only for two-pass teacher scoring

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

    @property
    def mean_teacher_logprob(self) -> float:
        s = [t for t in self.scored if t.teacher_logprob is not None]
        return sum(t.teacher_logprob for t in s) / len(s) if s else float("nan")

    @property
    def mean_kl(self) -> float:
        s = [t for t in self.scored if t.kl is not None]
        return sum(t.kl for t in s) / len(s) if s else float("nan")


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

    scored_set = set(_scored_positions(ids, offsets, spans))
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


def _scored_positions(ids: list[int], offsets: list[tuple[int, int]], spans) -> list[int]:
    """Positions whose character range overlaps a span, in sequence order.
    Position 0 has nothing to condition on and is never scored."""

    def in_span(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in spans)

    return [t for t in range(1, len(ids)) if in_span(*offsets[t])]


def kl_pair_ids(
    student_model,
    student_ids: list[int],
    teacher_model,
    teacher_ids: list[int],
    pairs: list[tuple[int, int]],
    top_k: int = 3,
    chunk_size: int = 128,
) -> tuple[dict, dict, dict, dict, dict]:
    """One forward pass per model; exact per-position KL over the full vocabulary.

    `pairs` aligns scored positions across the two renderings: (s, t) with
    student_ids[s] == teacher_ids[t]. Both models' log-softmax rows for a chunk
    are materialized together in float32, so the chunk is smaller than in
    score_ids. Returns dicts keyed by the student position: student logprob,
    teacher logprob, KL(teacher ‖ student), and top_k alternatives of each.
    """
    logits_s = student_model(mx.array(student_ids)[None])[0]
    logits_t = teacher_model(mx.array(teacher_ids)[None])[0]
    # distinct weights may pad the vocabulary differently; KL is over the overlap
    vocab = min(logits_s.shape[-1], logits_t.shape[-1])

    lp_student: dict[int, float] = {}
    lp_teacher: dict[int, float] = {}
    kls: dict[int, float] = {}
    alts_s: dict[int, list[tuple[int, float]]] = {}
    alts_t: dict[int, list[tuple[int, float]]] = {}

    def ranked(top_ids, top_lps, j):
        return sorted(zip(top_ids[j].tolist(), top_lps[j].tolist()), key=lambda p: -p[1])

    for i in range(0, len(pairs), chunk_size):
        chunk = pairs[i : i + chunk_size]
        rows_s = logits_s[mx.array([s - 1 for s, _ in chunk])][:, :vocab].astype(mx.float32)
        rows_t = logits_t[mx.array([t - 1 for _, t in chunk])][:, :vocab].astype(mx.float32)
        rows_s = rows_s - mx.logsumexp(rows_s, axis=-1, keepdims=True)
        rows_t = rows_t - mx.logsumexp(rows_t, axis=-1, keepdims=True)
        kl = (mx.exp(rows_t) * (rows_t - rows_s)).sum(axis=-1)
        chosen = mx.array([student_ids[s] for s, _ in chunk])[:, None]
        chosen_s = mx.take_along_axis(rows_s, chosen, axis=-1)[:, 0]
        chosen_t = mx.take_along_axis(rows_t, chosen, axis=-1)[:, 0]
        if top_k > 0:
            tid_s = mx.argpartition(-rows_s, kth=top_k - 1, axis=-1)[:, :top_k]
            tlp_s = mx.take_along_axis(rows_s, tid_s, axis=-1)
            tid_t = mx.argpartition(-rows_t, kth=top_k - 1, axis=-1)[:, :top_k]
            tlp_t = mx.take_along_axis(rows_t, tid_t, axis=-1)
        for j, (s_pos, _) in enumerate(chunk):
            lp_student[s_pos] = chosen_s[j].item()
            lp_teacher[s_pos] = chosen_t[j].item()
            kls[s_pos] = kl[j].item()
            if top_k > 0:
                alts_s[s_pos] = ranked(tid_s, tlp_s, j)
                alts_t[s_pos] = ranked(tid_t, tlp_t, j)
    return lp_student, lp_teacher, kls, alts_s, alts_t


def score_text_pair(
    student_model,
    student_tokenizer,
    built_student,
    teacher_model,
    teacher_tokenizer,
    built_teacher,
    model_name: str = "",
    teacher_name: str = "",
    top_k: int = 3,
) -> ScoreResult:
    """Score the same completion under two contexts and attach exact per-token
    KL(teacher ‖ student).

    The two BuiltTexts render different conversations (the teacher typically
    sees a privileged system prompt) but share the completion string, so their
    scored positions must carry identical token ids in identical order; a
    mismatch means the teacher context changed how the completion tokenizes and
    is an error, never scored approximately. The displayed text is the student's.
    """
    warnings = list(built_student.warnings)
    warnings += [f"teacher: {w}" for w in built_teacher.warnings if w not in built_student.warnings]

    ids_s, offsets_s = tokenize_with_offsets(student_tokenizer, built_student.text)
    ids_t, offsets_t = tokenize_with_offsets(teacher_tokenizer, built_teacher.text)
    if student_tokenizer.decode(ids_s) != built_student.text:
        warnings.append("tokenization does not round-trip the rendered text; offsets may be off")

    scored_s = _scored_positions(ids_s, offsets_s, built_student.spans)
    scored_t = _scored_positions(ids_t, offsets_t, built_teacher.spans)
    if [ids_s[p] for p in scored_s] != [ids_t[p] for p in scored_t]:
        raise ValueError(
            "teacher and student scored tokens do not align "
            f"({len(scored_s)} student vs {len(scored_t)} teacher); the teacher "
            "context changes how the completion tokenizes (incompatible "
            "tokenizer, or a template edge case)"
        )
    pairs = list(zip(scored_s, scored_t))

    lp_s, lp_t, kls, alts_s, alts_t = kl_pair_ids(
        student_model, ids_s, teacher_model, ids_t, pairs, top_k=top_k
    )

    def decode_alts(alts):
        return [TopToken(a, student_tokenizer.decode([a]), lp) for a, lp in alts] if alts else None

    scored_set = set(scored_s)
    tokens = []
    for t, (tid, (start, end)) in enumerate(zip(ids_s, offsets_s)):
        tokens.append(
            TokenScore(
                token_id=tid,
                text=built_student.text[start:end],
                start=start,
                end=end,
                scored=t in scored_set,
                logprob=lp_s.get(t),
                top_k=decode_alts(alts_s.get(t)),
                teacher_logprob=lp_t.get(t),
                kl=kls.get(t),
                teacher_top_k=decode_alts(alts_t.get(t)),
            )
        )
    return ScoreResult(
        text=built_student.text,
        tokens=tokens,
        model_name=model_name,
        warnings=warnings,
        teacher_model_name=teacher_name,
    )
