"""Score every token of a text with a single forward pass of an mlx-lm model."""

import math
from dataclasses import dataclass

import mlx.core as mx

from distill.adapters import lora_disabled

_MODEL_ONLY_OFFSET = (-1, -1)


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
    teacher_sampled: bool = False  # teacher replaced the student proposal


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


@dataclass
class AlignedTokens:
    """Token sequences plus aligned generated-token positions."""

    student_ids: list[int]
    student_offsets: list[tuple[int, int]]
    teacher_ids: list[int]
    teacher_offsets: list[tuple[int, int]]
    pairs: list[tuple[int, int]]


def tokenize_with_offsets(
    tokenizer, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize `text` (which may contain special tokens rendered by a chat
    template) and return per-token character offsets."""
    try:
        enc = tokenizer._tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
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


def _tokenize_for_scoring(
    tokenizer, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize text, using BOS as invisible context when it is not rendered."""
    ids, offsets = tokenize_with_offsets(tokenizer, text)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and (not ids or ids[0] != bos_id):
        return [bos_id, *ids], [_MODEL_ONLY_OFFSET, *offsets]
    return ids, offsets


def _visible_ids(ids: list[int], offsets: list[tuple[int, int]]) -> list[int]:
    """Token ids represented by characters in the displayed text."""
    return [tid for tid, (start, _) in zip(ids, offsets) if start >= 0]


def score_ids(
    model,
    ids: list[int],
    alt_positions: set[int] | None = None,
    top_k: int = 3,
    chunk_size: int = 512,
    cache_factory=None,
) -> tuple[dict[int, float], dict[int, list[tuple[int, float]]]]:
    """Single forward pass over `ids`.

    Returns (logprobs, alternatives): logprobs[t] = log p(ids[t] | ids[:t]) for
    every t >= 1, and alternatives[t] = top_k (token_id, logprob) pairs of the
    predictive distribution at the positions in `alt_positions`.
    """
    alt_positions = alt_positions or set()
    logits = None if cache_factory else model(mx.array(ids)[None])[0]
    cache = cache_factory() if cache_factory else None
    ids_arr = mx.array(ids)
    logprobs: dict[int, float] = {}
    alternatives: dict[int, list[tuple[int, float]]] = {}

    # each row t of `logits` predicts token t+1; process in chunks to avoid
    # materializing a float32 (seq, vocab) log-softmax all at once
    for row_start in range(0, len(ids) - 1, chunk_size):
        row_end = min(row_start + chunk_size, len(ids) - 1)
        if cache is not None:
            rows = model(mx.array(ids[row_start:row_end])[None], cache=cache)[0].astype(
                mx.float32
            )
        else:
            rows = logits[row_start:row_end].astype(mx.float32)
        lse = mx.logsumexp(rows, axis=-1)
        chosen = mx.take_along_axis(
            rows, ids_arr[row_start + 1 : row_end + 1][:, None], axis=-1
        )[:, 0]
        for j, lp in enumerate((chosen - lse).tolist()):
            logprobs[row_start + j + 1] = lp

        alt_rows = [
            t - 1 - row_start for t in alt_positions if row_start <= t - 1 < row_end
        ]
        if top_k > 0 and alt_rows:
            sub = rows[mx.array(alt_rows)]
            top_ids = mx.argpartition(-sub, kth=top_k - 1, axis=-1)[:, :top_k]
            top_lp = (
                mx.take_along_axis(sub, top_ids, axis=-1)
                - lse[mx.array(alt_rows)][:, None]
            )
            for j, r in enumerate(alt_rows):
                pairs = sorted(
                    zip(top_ids[j].tolist(), top_lp[j].tolist()), key=lambda p: -p[1]
                )
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
    cache_factory=None,
) -> ScoreResult:
    """Run one forward pass and attach log p(token_t | tokens_<t) to each token.

    A token is scored when its character range overlaps one of `spans`.
    Position 0 has nothing to condition on and is never scored.
    """
    warnings = list(warnings or [])
    ids, offsets = _tokenize_for_scoring(tokenizer, text)
    if tokenizer.decode(_visible_ids(ids, offsets)) != text:
        warnings.append(
            "tokenization does not round-trip the rendered text; offsets may be off"
        )

    scored_set = set(_scored_positions(ids, offsets, spans))
    logprobs, alternatives = score_ids(
        model,
        ids,
        alt_positions=scored_set,
        top_k=top_k,
        **({"cache_factory": cache_factory} if cache_factory else {}),
    )

    tokens = []
    for t, (tid, (start, end)) in enumerate(zip(ids, offsets)):
        if start < 0:
            continue
        alts = alternatives.get(t)
        tokens.append(
            TokenScore(
                token_id=tid,
                text=text[start:end],
                start=start,
                end=end,
                scored=t in scored_set,
                logprob=logprobs.get(t),
                top_k=[TopToken(a, tokenizer.decode([a]), lp) for a, lp in alts]
                if alts
                else None,
            )
        )
    return ScoreResult(
        text=text, tokens=tokens, model_name=model_name, warnings=warnings
    )


def _scored_positions(
    ids: list[int], offsets: list[tuple[int, int]], spans
) -> list[int]:
    """Positions whose character range overlaps a span, in sequence order.
    Position 0 has nothing to condition on and is never scored."""

    def in_span(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in spans)

    return [t for t in range(1, len(ids)) if in_span(*offsets[t])]


def align_scored_tokens(
    student_tokenizer,
    built_student,
    teacher_tokenizer,
    built_teacher,
) -> AlignedTokens:
    """Strictly align generated tokens under student and teacher contexts."""
    ids_s, offsets_s = _tokenize_for_scoring(student_tokenizer, built_student.text)
    ids_t, offsets_t = _tokenize_for_scoring(teacher_tokenizer, built_teacher.text)
    scored_s = _scored_positions(ids_s, offsets_s, built_student.spans)
    scored_t = _scored_positions(ids_t, offsets_t, built_teacher.spans)
    if [ids_s[p] for p in scored_s] != [ids_t[p] for p in scored_t]:
        raise ValueError(
            "teacher and student scored tokens do not align "
            f"({len(scored_s)} student vs {len(scored_t)} teacher); the teacher "
            "context changes how the completion tokenizes (incompatible "
            "tokenizer, or a template edge case)"
        )
    return AlignedTokens(
        student_ids=ids_s,
        student_offsets=offsets_s,
        teacher_ids=ids_t,
        teacher_offsets=offsets_t,
        pairs=list(zip(scored_s, scored_t)),
    )


def kl_pair_ids(
    student_model,
    student_ids: list[int],
    teacher_model,
    teacher_ids: list[int],
    pairs: list[tuple[int, int]],
    top_k: int = 3,
    chunk_size: int = 128,
    teacher_without_adapter: bool = False,
    student_cache_factory=None,
    teacher_cache_factory=None,
) -> tuple[dict, dict, dict, dict, dict]:
    """One forward pass per model; exact per-position KL over the full vocabulary.

    `pairs` aligns scored positions across the two renderings: (s, t) with
    student_ids[s] == teacher_ids[t]. Both models' log-softmax rows for a chunk
    are materialized together in float32, so the chunk is smaller than in
    score_ids. Returns dicts keyed by the student position: student logprob,
    teacher logprob, KL(teacher ‖ student), and top_k alternatives of each.
    """
    from distill.prefill import select_logits

    policy_scoring = (
        student_cache_factory is not None or teacher_cache_factory is not None
    )
    if not pairs:
        return {}, {}, {}, {}, {}
    if any(
        s < 1
        or t < 1
        or s >= len(student_ids)
        or t >= len(teacher_ids)
        or student_ids[s] != teacher_ids[t]
        for s, t in pairs
    ):
        raise ValueError("teacher/student trajectory token IDs do not align")
    row_indices_s = {p: i for i, (p, _) in enumerate(pairs)}
    row_indices_t = {p: i for i, (_, p) in enumerate(pairs)}
    logits_s = (
        select_logits(
            student_model,
            student_ids,
            [s - 1 for s, _ in pairs],
            step_size=chunk_size,
            cache_factory=student_cache_factory,
        )
        if policy_scoring
        else student_model(mx.array(student_ids)[None])[0]
    )
    if teacher_without_adapter:
        # Force the student graph before mutating LoRA scales for the shared
        # model's teacher pass. This is required because MLX arrays are lazy.
        mx.eval(logits_s)
        with lora_disabled(teacher_model):
            logits_t = (
                select_logits(
                    teacher_model,
                    teacher_ids,
                    [t - 1 for _, t in pairs],
                    step_size=chunk_size,
                    cache_factory=teacher_cache_factory,
                )
                if policy_scoring
                else teacher_model(mx.array(teacher_ids)[None])[0]
            )
            mx.eval(logits_t)
    else:
        logits_t = (
            select_logits(
                teacher_model,
                teacher_ids,
                [t - 1 for _, t in pairs],
                step_size=chunk_size,
                cache_factory=teacher_cache_factory,
            )
            if policy_scoring
            else teacher_model(mx.array(teacher_ids)[None])[0]
        )
    if policy_scoring and logits_s.shape[-1] != logits_t.shape[-1]:
        raise ValueError("cartridge teacher/student vocabulary mismatch")
    # distinct weights may pad the vocabulary differently; KL is over the overlap
    vocab = min(logits_s.shape[-1], logits_t.shape[-1])

    lp_student: dict[int, float] = {}
    lp_teacher: dict[int, float] = {}
    kls: dict[int, float] = {}
    alts_s: dict[int, list[tuple[int, float]]] = {}
    alts_t: dict[int, list[tuple[int, float]]] = {}

    def ranked(top_ids, top_lps, j):
        return sorted(
            zip(top_ids[j].tolist(), top_lps[j].tolist()), key=lambda p: -p[1]
        )

    for i in range(0, len(pairs), chunk_size):
        chunk = pairs[i : i + chunk_size]
        rows_s = logits_s[
            mx.array([row_indices_s[s] if policy_scoring else s - 1 for s, _ in chunk])
        ][:, :vocab].astype(mx.float32)
        rows_t = logits_t[
            mx.array([row_indices_t[t] if policy_scoring else t - 1 for _, t in chunk])
        ][:, :vocab].astype(mx.float32)
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
    teacher_without_adapter: bool = False,
    student_cache_factory=None,
    teacher_cache_factory=None,
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
    warnings += [
        f"teacher: {w}"
        for w in built_teacher.warnings
        if w not in built_student.warnings
    ]

    aligned = align_scored_tokens(
        student_tokenizer, built_student, teacher_tokenizer, built_teacher
    )
    ids_s, offsets_s = aligned.student_ids, aligned.student_offsets
    if student_tokenizer.decode(_visible_ids(ids_s, offsets_s)) != built_student.text:
        warnings.append(
            "tokenization does not round-trip the rendered text; offsets may be off"
        )

    scored_s = [student_pos for student_pos, _ in aligned.pairs]

    lp_s, lp_t, kls, alts_s, alts_t = kl_pair_ids(
        student_model,
        ids_s,
        teacher_model,
        aligned.teacher_ids,
        aligned.pairs,
        top_k=top_k,
        teacher_without_adapter=teacher_without_adapter,
        **(
            {"student_cache_factory": student_cache_factory}
            if student_cache_factory
            else {}
        ),
        **(
            {"teacher_cache_factory": teacher_cache_factory}
            if teacher_cache_factory
            else {}
        ),
    )

    def decode_alts(alts):
        return (
            [TopToken(a, student_tokenizer.decode([a]), lp) for a, lp in alts]
            if alts
            else None
        )

    scored_set = set(scored_s)
    tokens = []
    for t, (tid, (start, end)) in enumerate(zip(ids_s, offsets_s)):
        if start < 0:
            continue
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


def token_pieces(tokenizer, ids):
    """Assign decoded characters to actual tokens, including byte fallback tokens."""
    pieces = []
    previous = ""
    for index in range(len(ids)):
        current = tokenizer.decode(ids[: index + 1])
        common = 0
        for a, b in zip(previous, current):
            if a != b:
                break
            common += 1
        remove = len(previous) - common
        while remove and pieces:
            tail = pieces[-1]
            if len(tail) <= remove:
                remove -= len(tail)
                pieces[-1] = ""
                # Retain the empty token's place while finding earlier text.
                j = len(pieces) - 2
                while remove and j >= 0:
                    take = min(remove, len(pieces[j]))
                    pieces[j] = pieces[j][:-take] if take else pieces[j]
                    remove -= take
                    j -= 1
                break
            pieces[-1] = tail[:-remove]
            remove = 0
        pieces.append(current[common:])
        previous = current
    return pieces


def score_generated(
    model,
    tokenizer,
    sample,
    *,
    model_name="",
    top_k=3,
    cache_factory=None,
    teacher_model=None,
    teacher_tokenizer=None,
    teacher_prompt_text=None,
    teacher_name="",
    teacher_cache_factory=None,
    teacher_without_adapter=False,
):
    """Rescore actual generated IDs. Decoded text is never re-encoded as targets."""
    if sample.token_ids is None:
        raise ValueError("generated scoring requires actual token IDs")
    prefix = tokenizer.encode(sample.prompt_text, add_special_tokens=False)
    ids = prefix + sample.token_ids
    positions = list(range(len(prefix), len(ids)))
    if not prefix:
        raise ValueError("generated scoring requires a nonempty model prompt")
    if teacher_model is None:
        student_lp, alternatives = score_ids(
            model,
            ids,
            alt_positions=set(positions),
            top_k=top_k,
            cache_factory=cache_factory,
        )
        teacher_lp, kls, teacher_alternatives = {}, {}, {}
    else:
        teacher_tokenizer = teacher_tokenizer or tokenizer
        teacher_prefix = teacher_tokenizer.encode(
            teacher_prompt_text, add_special_tokens=False
        )
        if not teacher_prefix or any(
            tokenizer.decode([i]) != teacher_tokenizer.decode([i])
            for i in sample.token_ids
        ):
            raise ValueError("teacher/student generated trajectory does not align")
        pairs = [
            (len(prefix) + i, len(teacher_prefix) + i)
            for i in range(len(sample.token_ids))
        ]
        student_lp, teacher_lp, kls, alternatives, teacher_alternatives = kl_pair_ids(
            model,
            ids,
            teacher_model,
            teacher_prefix + sample.token_ids,
            pairs,
            top_k=top_k,
            student_cache_factory=cache_factory,
            teacher_cache_factory=teacher_cache_factory,
            teacher_without_adapter=teacher_without_adapter,
        )
    tokens = [
        TokenScore(
            token_id=-1,
            text=sample.prompt_text,
            start=0,
            end=len(sample.prompt_text),
            scored=False,
            logprob=None,
        )
    ]
    cursor = len(sample.prompt_text)

    def alts(mapping, position):
        return [
            TopToken(i, tokenizer.decode([i]), lp)
            for i, lp in mapping.get(position, [])
        ] or None

    for index, (position, tid, piece) in enumerate(
        zip(positions, sample.token_ids, token_pieces(tokenizer, sample.token_ids))
    ):
        tokens.append(
            TokenScore(
                token_id=tid,
                text=piece,
                start=cursor,
                end=cursor + len(piece),
                scored=True,
                logprob=student_lp.get(position),
                top_k=alts(alternatives, position),
                teacher_logprob=teacher_lp.get(position),
                kl=kls.get(position),
                teacher_top_k=alts(teacher_alternatives, position),
                teacher_sampled=bool(
                    sample.teacher_sampled and sample.teacher_sampled[index]
                ),
            )
        )
        cursor += len(piece)
    return ScoreResult(
        text=sample.prompt_text + tokenizer.decode(sample.token_ids),
        tokens=tokens,
        model_name=model_name,
        teacher_model_name=teacher_name if teacher_model else None,
        warnings=[],
    )
