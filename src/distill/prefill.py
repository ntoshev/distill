"""Memory-bounded prompt prefill helpers for MLX language models."""

import copy

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, make_prompt_cache


class _RightPaddedArraysCache(ArraysCache):
    """Honor right-padding lengths after MLX-LM merges empty recurrent caches."""

    def prepare(self, lengths=None, **kwargs):
        # ArraysCache.merge initializes left_padding to zeros. Its mask prefers
        # that field over lengths, otherwise treating right padding as real input.
        if kwargs.get("right_padding") is not None:
            self.left_padding = None
        super().prepare(lengths=lengths, **kwargs)


def make_batch_prompt_cache(model, cache_factory=None):
    """Create a batch cache with a local fix for recurrent right-padding masks."""
    cache = cache_factory() if cache_factory else make_prompt_cache(model)
    for index, entry in enumerate(cache):
        if type(entry) is ArraysCache:
            replacement = _RightPaddedArraysCache(len(entry.state))
            replacement.state = entry.state
            replacement.left_padding = entry.left_padding
            replacement.lengths = entry.lengths
            cache[index] = replacement
    return cache


def clone_prompt_cache(cache):
    """Clone a prompt cache without duplicating model weights."""
    return copy.deepcopy(cache)


def merge_prompt_caches(caches):
    """Merge per-sequence caches into the batch-aware cache variants."""
    if not caches:
        return []
    merged = []
    for index in range(len(caches[0])):
        entry = caches[0][index]
        if not hasattr(entry, "merge"):
            raise ValueError(
                f"{type(entry).__name__} does not support batched prompt caches"
            )
        merged.append(type(entry).merge([cache[index] for cache in caches]))
    return merged


DEFAULT_PREFILL_STEP_SIZE = 2048


def _cache_state(cache):
    return [entry.state for entry in cache]


def _prefill_cache(model, ids: list[int], cache, step_size: int) -> None:
    """Advance ``cache`` over ``ids`` without materializing output logits."""
    for start in range(0, len(ids), step_size):
        piece = mx.array(ids[start : start + step_size])
        model(piece[None], cache=cache)
        # Evaluating only the cache state lets MLX prune the unused language-head
        # output. Clearing the free cache between pieces keeps long prompts from
        # accumulating allocator pressure.
        mx.eval(_cache_state(cache))
        mx.clear_cache()


def prefill_cache(
    model,
    ids: list[int],
    cache,
    *,
    step_size: int = DEFAULT_PREFILL_STEP_SIZE,
) -> None:
    """Populate a reusable prompt cache without retaining output logits."""
    if step_size <= 0:
        raise ValueError("prefill step size must be positive")
    if ids:
        _prefill_cache(model, ids, cache, step_size)


def prefill_last_logits(
    model,
    ids: list[int],
    cache,
    *,
    step_size: int = DEFAULT_PREFILL_STEP_SIZE,
):
    """Prefill a prompt and return only its final predictive-logit row.

    Short prompts retain the original single-forward path. Long prompts are
    processed into the KV cache in bounded pieces, with only the final token
    sent through the language head for an evaluated result.
    """
    if not ids:
        raise ValueError("prompt must contain at least one token")
    if step_size <= 0:
        raise ValueError("prefill step size must be positive")

    if len(ids) <= step_size:
        logits = model(mx.array(ids)[None], cache=cache)[0, -1].astype(mx.float32)
    else:
        _prefill_cache(model, ids[:-1], cache, step_size)
        logits = model(mx.array(ids[-1:])[None], cache=cache)[0, -1].astype(mx.float32)
    mx.eval(logits, _cache_state(cache))
    return logits


def select_logits(
    model,
    ids: list[int],
    rows: list[int],
    *,
    cache_factory=None,
    step_size: int = DEFAULT_PREFILL_STEP_SIZE,
):
    """Return selected predictive rows without forwarding a long prompt at once.

    ``rows`` are zero-based model-output positions. For long inputs, tokens
    before each selected run are consumed cache-only and selected runs are
    evaluated in bounded pieces. This preserves the full left context while
    avoiding a ``sequence_length x vocabulary`` logits allocation.
    """
    if not ids:
        raise ValueError("input must contain at least one token")
    if not rows:
        raise ValueError("at least one logit row must be selected")
    if step_size <= 0:
        raise ValueError("prefill step size must be positive")
    if any(row < 0 or row >= len(ids) for row in rows):
        raise ValueError("selected logit row is outside the input")
    if any(left >= right for left, right in zip(rows, rows[1:])):
        raise ValueError("selected logit rows must be strictly increasing")

    input_ids = mx.array(ids)
    if len(ids) <= step_size:
        logits = (
            model(input_ids[None], cache=cache_factory())
            if cache_factory
            else model(input_ids[None])
        )[0]
        selected = logits[mx.array(rows)].astype(mx.float32)
        mx.eval(selected)
        return selected

    from mlx_lm.models.cache import make_prompt_cache

    cache = cache_factory() if cache_factory else make_prompt_cache(model)
    selected_pieces = []
    cursor = 0
    run_start = rows[0]
    run_end = run_start
    runs = []
    for row in rows[1:]:
        if row == run_end + 1:
            run_end = row
        else:
            runs.append((run_start, run_end + 1))
            run_start = run_end = row
    runs.append((run_start, run_end + 1))

    for start, end in runs:
        _prefill_cache(model, ids[cursor:start], cache, step_size)
        for piece_start in range(start, end, step_size):
            piece_end = min(piece_start + step_size, end)
            logits = model(input_ids[piece_start:piece_end][None], cache=cache)[0]
            logits = logits.astype(mx.float32)
            mx.eval(logits, _cache_state(cache))
            selected_pieces.append(logits)
            mx.clear_cache()
        cursor = end

    selected = (
        selected_pieces[0]
        if len(selected_pieces) == 1
        else mx.concatenate(selected_pieces, axis=0)
    )
    mx.eval(selected)
    return selected
