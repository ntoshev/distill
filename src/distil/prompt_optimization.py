"""Frozen-model hard prompt optimization and a mechanically scored reversal task.

The task reverses space-separated word tokens, not characters or BPE fragments.
PEZ retains continuous iterates but evaluates real vocabulary embeddings. GCG
ranks single-token substitutions by the embedding gradient, then measures their
actual loss. Both enforce exact tokenizer round trips at the requested length.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import random
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim

from distil.chunked_kl import _head_logits, _parts
from distil.templating import render_generation_prompt


WORDS = """apple river chair table bread cloud grass stone book tree light water
flower bird moon sun rain wind fish horse house road milk sand leaf boat door
clock shoe paper glass gold silver star fire snow beach train plane spoon fork
plate cup bowl hat coat bag ball bell cake salt rice corn dog cat bear frog
duck goat lion mouse snake shell rock wood steel beach brush desk lamp park
path pond ring rope rose seed ship shop silk soap sock song soup team tent
town wall wave wheel wing wire wolf yard drum flag fruit grape green blue
red black white pink brown gray orange purple yellow garden window pencil""".split()

# Task-independent English starts; none gives the reversal rule or an example.
# These budgets are checked with the actual tokenizer, never assumed universal.
NEUTRAL_INITIAL_PROMPTS = {
    1: "Answer",
    2: "Answer briefly",
    4: "Give a concise answer",
    8: "Please respond with just the requested answer.",
    16: "Please read the user's message carefully and respond directly with just the requested answer.",
}
COPY_INITIAL_PROMPTS = {
    1: "Copy",
    2: "Copy words",
    4: "Copy the words only",
    8: "Copy words. Output only the word list",
    16: "Copy the input words exactly. Output only these words, separated by single spaces.",
}


@dataclass(frozen=True)
class ReversalCase:
    case_id: str
    words: tuple[str, ...]

    @property
    def question(self):
        return " ".join(self.words)

    @property
    def answer(self):
        return " ".join(reversed(self.words))

    def to_dict(self):
        return {**asdict(self), "words": list(self.words),
                "question": self.question, "answer": self.answer}


def reversal_splits(tokenizer, *, seed=1729, train_size=128, val_size=64,
                    test_size=512, shift_size=128):
    """Fresh permutations; held-out words and lengths are separate test splits.

    Every word is one model token both initially and after a space. Thus a list
    of N words is also N BPE tokens. Palindromes are excluded so copying cannot
    accidentally solve the task. Repeated words remain possible.
    """
    if min(train_size, val_size, test_size, shift_size) < 1:
        raise ValueError("All split sizes must be positive")
    words = sorted({w for w in WORDS
                    if len(tokenizer.encode(w, add_special_tokens=False)) == 1
                    and len(tokenizer.encode(" " + w, add_special_tokens=False)) == 1})
    if len(words) < 48:
        raise ValueError("Need at least 48 single-token task words")
    rng = random.Random(seed)
    rng.shuffle(words)
    known, unseen = words[:32], words[32:48]
    used = set()

    def make(name, count, vocab, lengths):
        result = []
        for i in range(count):
            while True:
                seq = tuple(rng.choices(vocab, k=rng.choice(lengths)))
                if seq != seq[::-1] and seq not in used:
                    break
            used.add(seq)
            case = ReversalCase(f"{name}-{i:04d}", seq)
            if (len(tokenizer.encode(case.question, add_special_tokens=False)) != len(seq)
                    or len(tokenizer.encode(case.answer, add_special_tokens=False)) != len(seq)):
                raise ValueError("Task word boundaries are not tokenizer-stable")
            result.append(case)
        return result

    return {
        "train": make("train", train_size, known, (3, 4, 5)),
        "validation": make("validation", val_size, known, (3, 4, 5)),
        "test": make("test", test_size, known, (3, 4, 5)),
        "unseen_words": make("unseen_words", shift_size, unseen, (3, 4, 5)),
        "longer": make("longer", shift_size, known, (6, 7, 8)),
    }


def wilson_interval(successes, count, z=1.96):
    if count < 1:
        raise ValueError("Empty evaluation set")
    p = successes / count
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / denom
    return [max(0., center - radius), min(1., center + radius)]


class PromptLayout:
    """Exact system-message slot with checks against ordinary text rendering."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        marker = "DISTIL_PROMPT_SLOT_37cb21"
        text = self.render(marker, "apple river chair")
        if text.count(marker) != 1:
            raise ValueError("Chat template must contain the system message once")
        self.prefix_text, _ = text.split(marker)
        self.prefix = self.encode(self.prefix_text)

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def render(self, prompt, question):
        return render_generation_prompt(self.tokenizer, [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ], enable_thinking=False)

    def suffix(self, question):
        marker = "DISTIL_PROMPT_SLOT_37cb21"
        before, after = self.render(marker, question).split(marker)
        if before != self.prefix_text:
            raise ValueError("Chat template prefix depends on user content")
        return self.encode(after)

    def valid(self, prompt_ids):
        text = self.tokenizer.decode(prompt_ids)
        return (self.encode(text) == list(prompt_ids)
                and self.encode(self.render(text, "apple river chair"))
                == self.prefix + list(prompt_ids) + self.suffix("apple river chair"))

    def context(self, prompt_ids, question):
        ids = self.prefix + list(prompt_ids) + self.suffix(question)
        actual = self.encode(self.render(self.tokenizer.decode(prompt_ids), question))
        if actual != ids:
            raise ValueError("Optimized tokens do not match exported chat prompt")
        return ids


class PromptVocabulary:
    """Printable ordinary ASCII tokens; all methods use the same search space."""

    def __init__(self, tokenizer, embedding, layout, *, allowed_words=None):
        self.layout = layout
        special = set(tokenizer._tokenizer.all_special_ids)
        ids = []
        for token_id in range(tokenizer.vocab_size):
            if token_id in special:
                continue
            text = tokenizer.decode([token_id])
            if (text and all(32 <= ord(c) < 127 for c in text)
                    and layout.encode(text) == [token_id]):
                if allowed_words is not None:
                    surface = text.strip().lower()
                    # The system dictionary omits many regular word forms,
                    # including "requested" and "words". Recognize generic
                    # inflections rather than adding task-specific words.
                    stems = [surface]
                    for suffix, replacements in (("s", ("",)), ("es", ("",)),
                                                  ("ies", ("y",)), ("ed", ("", "e")),
                                                  ("ing", ("", "e")), ("ly", ("",))):
                        if surface.endswith(suffix):
                            stems.extend(surface[:-len(suffix)] + r for r in replacements)
                    contractions = {"'s", "n't", "'re", "'ve", "'ll", "'d", "'m"}
                    punctuation = not any(c.isalnum() for c in text)
                    if not any(s in allowed_words for s in stems) and surface not in contractions and not punctuation:
                        continue
                ids.append(token_id)
        if not ids:
            raise ValueError("Empty allowed prompt vocabulary")
        self.ids = ids
        self.array = mx.array(ids)
        self.embeddings = embedding(self.array).astype(mx.float32)
        self.normalized = self.embeddings / mx.maximum(
            mx.linalg.norm(self.embeddings, axis=-1, keepdims=True), 1e-12)
        self.squared_norms = mx.square(self.embeddings).sum(axis=-1)
        mx.eval(self.embeddings, self.normalized)
        self.index = {token_id: i for i, token_id in enumerate(ids)}

    def embed(self, ids):
        return self.embeddings[mx.array([self.index[i] for i in ids])]

    def projection_scores(self, vectors, metric="cosine"):
        if metric == "euclidean":
            # Drop the query's squared norm, which is constant across tokens.
            return 2 * vectors @ self.embeddings.T - self.squared_norms
        if metric != "cosine":
            raise ValueError("Projection metric must be cosine or euclidean")
        unit = vectors / mx.maximum(mx.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
        return unit @ self.normalized.T

    def project(self, vectors, *, metric="cosine"):
        # Cosine nearest neighbors, as in the PEZ reference implementation.
        return self.array[mx.argmax(self.projection_scores(vectors, metric), axis=-1)].tolist()

    def project_feasible(self, vectors, previous, *, top_k=64, metric="cosine"):
        """Nearest-neighbor projection with the full-string export constraint.

        Try the unconstrained joint projection first. If BPE merges its pieces,
        greedily choose each coordinate's nearest feasible alternative while
        retaining a feasible complete sequence. The previous token is always an
        available fallback. Continuous PEZ state is never reset by this repair.
        """
        scores = self.projection_scores(vectors, metric)
        raw = self.array[mx.argmax(scores, axis=-1)].tolist()
        if self.layout.valid(raw):
            return raw, False
        k = min(top_k, len(self.ids))
        top = mx.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
        values = mx.take_along_axis(scores, top, axis=-1)
        top = mx.take_along_axis(top, mx.argsort(-values, axis=-1), axis=-1)
        alternatives = self.array[top].tolist()
        result = list(previous)
        if not self.layout.valid(result):
            raise ValueError("Projection fallback must be a feasible prompt")
        for pos, pool in enumerate(alternatives):
            for token in pool:
                proposal = list(result)
                proposal[pos] = token
                if self.layout.valid(proposal):
                    result = proposal
                    break
        return result, True

    def initial(self, length, rng, *, initialization="random"):
        if length < 1:
            raise ValueError("Prompt length must be positive")
        if initialization in ("neutral", "copy"):
            prompts = NEUTRAL_INITIAL_PROMPTS if initialization == "neutral" else COPY_INITIAL_PROMPTS
            if length not in prompts:
                raise ValueError("No instruction initialization at this token budget")
            ids = self.layout.encode(prompts[length])
            if len(ids) != length or not self.layout.valid(ids) or any(i not in self.index for i in ids):
                raise ValueError("Initial instruction does not fit this tokenizer's token budget")
            return ids
        if initialization != "random":
            raise ValueError("Initialization must be random, neutral, or copy")
        for _ in range(10000):
            ids = rng.choices(self.ids, k=length)
            if self.layout.valid(ids):
                return ids
        raise ValueError("Could not initialize a canonical prompt")

    def substitutions(self, ids, gradient, *, top_k, candidates, rng, guided=True):
        if not ids or candidates < 1 or top_k < 1:
            raise ValueError("Candidate count and top-k must be positive")
        if guided:
            scores = gradient @ self.embeddings.T
            k = min(top_k, len(self.ids))
            top = mx.argpartition(scores, kth=k - 1, axis=-1)[:, :k]
            # Sort for deterministic inclusion of the best linearized candidate.
            values = mx.take_along_axis(scores, top, axis=-1)
            top = mx.take_along_axis(top, mx.argsort(values, axis=-1), axis=-1)
            pool = self.array[top].tolist()
        else:
            pool = [self.ids] * len(ids)
        result, seen = [], {tuple(ids)}
        attempts = 0
        positions = list(range(len(ids)))
        rng.shuffle(positions)
        best_count = min(len(ids), max(1, candidates // 4))
        # A few best proposals, then uniform coordinate/top-k sampling. Shuffling
        # avoids starving later coordinates when the prompt exceeds the budget.
        while len(result) < candidates and attempts < candidates * 30:
            pos = positions[attempts] if attempts < best_count else rng.randrange(len(ids))
            token = pool[pos][0] if guided and attempts < best_count else rng.choice(pool[pos])
            proposal = list(ids)
            proposal[pos] = token
            key = tuple(proposal)
            if key not in seen and self.layout.valid(proposal):
                seen.add(key)
                result.append(proposal)
            attempts += 1
        return result


class PromptObjective:
    """Answer-only, per-sequence mean CE including the actual end-of-turn token.

    Padding is on the right, after all scored positions, and cannot affect them
    in the causal model. Vocabulary logits are allocated only for answer rows.
    Gradients are exclusively with respect to prompt embeddings.
    """

    def __init__(self, model, tokenizer, layout):
        self.model, self.tokenizer, self.layout = model, tokenizer, layout
        model.freeze()
        self.trunk, self.head = _parts(model)
        self.eos = tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Tokenizer needs an end-of-turn token")
        self.forward_examples = 0
        self.gradient_examples = 0

    @contextmanager
    def differentiable(self):
        training = self.model.training
        # Qwen3.5's inference recurrence kernel has no autodiff rule.
        self.model.train()
        try:
            yield
        finally:
            self.model.train(training)

    @contextmanager
    def inference(self):
        training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(training)

    def batch(self, prompt_ids, cases):
        sequences, targets, positions = [], [], []
        for ids, case in zip(prompt_ids, cases, strict=True):
            context = self.layout.context(ids, case.question)
            answer = self.layout.encode(case.answer) + [self.eos]
            sequences.append(context + answer[:-1])
            targets.append(answer)
            positions.append(list(range(len(context) - 1, len(context) - 1 + len(answer))))
        max_seq, max_ans = max(map(len, sequences)), max(map(len, targets))
        pad = lambda rows, size: mx.array([r + [0] * (size - len(r)) for r in rows])
        masks = mx.array([[1.] * len(r) + [0.] * (max_ans - len(r)) for r in targets])
        return pad(sequences, max_seq), pad(targets, max_ans), pad(positions, max_ans), masks

    def losses(self, prompt_vectors, batch):
        ids, targets, rows, masks = batch
        embedding = self.trunk.embed_tokens(ids)
        start = len(self.layout.prefix)
        length = prompt_vectors.shape[-2]
        if prompt_vectors.ndim == 2:
            prompt_vectors = mx.broadcast_to(prompt_vectors, (ids.shape[0], *prompt_vectors.shape))
        # Match ordinary token inference exactly; fp32 optimizer state survives
        # while the model consumes its native embedding dtype.
        embedding = mx.concatenate([embedding[:, :start],
                                    prompt_vectors.astype(embedding.dtype),
                                    embedding[:, start + length:]], axis=1)
        hidden = self.trunk(ids, input_embeddings=embedding)
        selected = hidden[mx.arange(ids.shape[0])[:, None], rows]
        logits = _head_logits(self.head, selected.reshape(-1, selected.shape[-1]))
        logits = logits.reshape(*targets.shape, -1)
        chosen = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        ce = mx.logsumexp(logits, axis=-1) - chosen
        return (ce * masks).sum(axis=-1) / masks.sum(axis=-1)

    def loss_and_grad(self, prompt_ids, cases, *, vectors=None, microbatch=2):
        if not cases or microbatch < 1:
            raise ValueError("Need cases and a positive gradient microbatch size")
        if len(cases) > microbatch:
            # Retain the effective batch size without retaining all activations
            # at once. Weight uneven last chunks by their example count.
            total_loss, total_gradient = 0., None
            for start in range(0, len(cases), microbatch):
                chunk = cases[start:start + microbatch]
                loss, gradient = self.loss_and_grad(prompt_ids, chunk, vectors=vectors, microbatch=microbatch)
                weight = len(chunk) / len(cases)
                total_loss += weight * loss
                term = weight * gradient
                total_gradient = term if total_gradient is None else total_gradient + term
                mx.eval(total_gradient)
            return total_loss, total_gradient
        batch = self.batch([prompt_ids] * len(cases), cases)
        if vectors is None:
            vectors = self.trunk.embed_tokens(mx.array(prompt_ids)).astype(mx.float32)
        with self.differentiable():
            loss, gradient = mx.value_and_grad(lambda p: self.losses(p, batch).mean())(vectors)
            mx.eval(loss, gradient)
        self.forward_examples += len(cases)
        self.gradient_examples += len(cases)
        if not math.isfinite(loss.item()) or not mx.all(mx.isfinite(gradient)).item():
            raise FloatingPointError("Nonfinite prompt loss or gradient")
        return loss.item(), gradient

    def score(self, prompts, cases, *, microbatch=64):
        if not prompts or not cases or microbatch < 1:
            raise ValueError("Need prompts, cases, and positive microbatch size")
        pairs = [(p, c) for p in prompts for c in cases]
        values = []
        with self.inference():
            for start in range(0, len(pairs), microbatch):
                chunk = pairs[start:start + microbatch]
                vectors = self.trunk.embed_tokens(mx.array([p for p, c in chunk])).astype(mx.float32)
                batch = self.batch([p for p, c in chunk], [c for p, c in chunk])
                loss = self.losses(vectors, batch)
                values.extend(loss.tolist())
        self.forward_examples += len(pairs)
        return [sum(values[i:i + len(cases)]) / len(cases)
                for i in range(0, len(values), len(cases))]

    def score_vectors(self, vectors, placeholder_ids, cases, *, microbatch=64):
        values = []
        with self.inference():
            for start in range(0, len(cases), microbatch):
                chunk = cases[start:start + microbatch]
                batch = self.batch([placeholder_ids] * len(chunk), chunk)
                values.extend(self.losses(vectors, batch).tolist())
        self.forward_examples += len(cases)
        return sum(values) / len(values)


class PEZ:
    def __init__(self, vocabulary, initial_ids, learning_rate=.03, *, normalize_shadow=False,
                 metric="cosine", optimizer="adam", gradient_microbatch=2, gradient_clip=None):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not initial_ids or not vocabulary.layout.valid(initial_ids):
            raise ValueError("Initial prompt must be nonempty and canonical")
        self.vocabulary = vocabulary
        self.ids = list(initial_ids)
        self.shadow = vocabulary.embed(self.ids)
        self.normalize_shadow = normalize_shadow
        self.metric = metric
        if metric not in ("cosine", "euclidean"):
            raise ValueError("Projection metric must be cosine or euclidean")
        if gradient_microbatch < 1:
            raise ValueError("Gradient microbatch must be positive")
        self.gradient_microbatch = gradient_microbatch
        if gradient_clip is not None and (not math.isfinite(gradient_clip) or gradient_clip <= 0):
            raise ValueError("Gradient clip must be finite and positive")
        self.gradient_clip = gradient_clip
        self.shadow_norms = mx.maximum(mx.linalg.norm(self.shadow, axis=-1, keepdims=True), 1e-12)
        # MLX defaults to uncorrected moments, unlike standard/PyTorch Adam.
        # That inflates early steps by 3–6x and changes the hard projection path.
        if optimizer == "adam":
            self.optimizer = optim.Adam(learning_rate=learning_rate, bias_correction=True)
        elif optimizer == "adafactor":
            # Match the PEZ language-model reference's fixed-rate Adafactor.
            self.optimizer = optim.Adafactor(learning_rate=learning_rate, relative_step=False,
                                            scale_parameter=False, warmup_init=False)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(learning_rate=learning_rate)
        else:
            raise ValueError("PEZ optimizer must be adam, adafactor, or sgd")
        self.rejected = 0

    def step(self, objective, cases):
        loss, gradient = objective.loss_and_grad(self.ids, cases, microbatch=self.gradient_microbatch)
        gradient_norm = mx.linalg.norm(gradient)
        if self.gradient_clip is not None:
            # Clip the whole prompt gradient, after effective-batch averaging,
            # as in the reference language-model PEZ training loop.
            gradient = gradient * mx.minimum(1., self.gradient_clip / mx.maximum(gradient_norm, 1e-12))
        self.shadow = self.optimizer.apply_gradients({"prompt": gradient}, {"prompt": self.shadow})["prompt"]
        if self.normalize_shadow:
            # Cosine projection is scale invariant. Preserve its continuous
            # direction while keeping future steps at the embedding scale.
            self.shadow = self.shadow * self.shadow_norms / mx.maximum(
                mx.linalg.norm(self.shadow, axis=-1, keepdims=True), 1e-12)
        mx.eval(self.shadow, self.optimizer.state)
        proposal, repaired = self.vocabulary.project_feasible(self.shadow, self.ids, metric=self.metric)
        self.ids = proposal
        self.rejected += int(repaired)
        return {"loss_before": loss, "gradient_norm": gradient_norm.item(),
                "applied_gradient_norm": mx.linalg.norm(gradient).item(),
                "repaired_projections": self.rejected}


class SoftPrompt:
    """Continuous control: same gradient path and optimizer, without projection."""

    def __init__(self, vocabulary, initial_ids, learning_rate=.03):
        self.ids = list(initial_ids)  # Only supplies layout/position placeholders.
        self.vectors = vocabulary.embed(self.ids)
        self.optimizer = optim.Adam(learning_rate=learning_rate, bias_correction=True)

    def step(self, objective, cases):
        loss, gradient = objective.loss_and_grad(self.ids, cases, vectors=self.vectors)
        self.vectors = self.optimizer.apply_gradients({"prompt": gradient}, {"prompt": self.vectors})["prompt"]
        mx.eval(self.vectors, self.optimizer.state)
        return {"loss_before": loss, "gradient_norm": mx.linalg.norm(gradient).item()}


class GCG:
    def __init__(self, vocabulary, initial_ids, *, seed=0, top_k=64, candidates=32, guided=True):
        if not initial_ids or not vocabulary.layout.valid(initial_ids):
            raise ValueError("Initial prompt must be nonempty and canonical")
        if top_k < 1 or candidates < 1:
            raise ValueError("Candidate count and top-k must be positive")
        self.vocabulary, self.ids = vocabulary, list(initial_ids)
        self.rng = random.Random(seed)
        self.top_k, self.candidates, self.guided = top_k, candidates, guided

    def step(self, objective, cases):
        if self.guided:
            gradient_loss, gradient = objective.loss_and_grad(self.ids, cases)
        else:
            gradient_loss = None
            gradient = None
        proposals = self.vocabulary.substitutions(
            self.ids, gradient, top_k=self.top_k, candidates=self.candidates,
            rng=self.rng, guided=self.guided)
        measured = objective.score([self.ids, *proposals], cases)
        before, losses = measured[0], measured[1:]
        best = min(range(len(losses)), key=losses.__getitem__) if losses else None
        # The unchanged prompt is an explicit candidate: never accept a worse
        # substitution on the very minibatch whose gradient proposed it.
        if best is not None and losses[best] < before:
            # Recheck both using the same small batch shape. Low precision
            # kernels can change slightly between candidate scoring chunks.
            before, after = objective.score([self.ids, proposals[best]], cases)
            if after < before:
                self.ids = proposals[best]
            else:
                after = before
        else:
            after = before
        return {"loss_before": before, "loss_after": after, "gradient_loss": gradient_loss,
                "gradient_norm": mx.linalg.norm(gradient).item() if gradient is not None else None,
                "candidates_scored": len(proposals)}


def _generate_batch(model, tokenizer, contexts, *, batch_size, max_tokens):
    """Ordinary mlx-lm batching, retaining its real token IDs and stop reasons."""
    from mlx_lm.generate import BatchGenerator
    generator = BatchGenerator(model, stop_tokens=[[t] for t in tokenizer.eos_token_ids],
                               completion_batch_size=batch_size,
                               prefill_batch_size=min(batch_size, 32))
    ids, content, finishes = {}, {}, {}
    try:
        uids = generator.insert(contexts, [max_tokens] * len(contexts))
        ids = {uid: [] for uid in uids}
        content = {uid: [] for uid in uids}
        while responses := generator.next_generated():
            for response in responses:
                ids[response.uid].append(response.token)
                if response.finish_reason != "stop":
                    content[response.uid].append(response.token)
                if response.finish_reason is not None:
                    finishes[response.uid] = response.finish_reason
        return [{"output": tokenizer.decode(content[uid]), "token_ids": ids[uid],
                 "finish_reason": finishes[uid], "num_tokens": len(ids[uid])} for uid in uids]
    finally:
        generator.close()


def evaluate_prompt(model, tokenizer, layout, prompt_ids, cases, *, batch_size=32, max_tokens=24):
    """Free-running greedy generation of the exported text prompt; no constraints."""
    if batch_size < 1 or max_tokens < 1 or not cases:
        raise ValueError("Need cases and positive generation limits")
    text = tokenizer.decode(prompt_ids)
    model.eval()
    started = perf_counter()
    rows = []
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        contexts = [layout.encode(layout.render(text, c.question)) for c in chunk]
        generated = _generate_batch(model, tokenizer, contexts, batch_size=batch_size, max_tokens=max_tokens)
        for case, generation in zip(chunk, generated, strict=True):
            output = generation["output"]
            # Outer whitespace only; extra text/punctuation and wrong ordering fail.
            rows.append({**case.to_dict(), **generation,
                         "correct": output.strip() == case.answer and generation["finish_reason"] == "stop",
                         "copy": output.strip() == case.question,
                         "empty": not output.strip()})
    return _evaluation_metrics(rows, perf_counter() - started), rows


def _evaluation_metrics(rows, seconds):
    successes = sum(row["correct"] for row in rows)
    return {"count": len(rows), "correct": successes, "accuracy": successes / len(rows),
            "wilson_95": wilson_interval(successes, len(rows)),
            "copy_rate": sum(row["copy"] for row in rows) / len(rows),
            "empty_rate": sum(row["empty"] for row in rows) / len(rows),
            "truncation_rate": sum(row["finish_reason"] == "length" for row in rows) / len(rows),
            "seconds": seconds}


def evaluate_soft_prompt(objective, placeholder_ids, vectors, cases, *, max_tokens=24):
    """Actual greedy generation with continuous prefix vectors; labeled separately."""
    from mlx_lm.generate import generate_step
    model, tokenizer, layout = objective.model, objective.tokenizer, objective.layout
    model.eval()
    start = len(layout.prefix)
    begun = perf_counter()
    rows = []
    for case in cases:
        context = layout.context(placeholder_ids, case.question)
        embeddings = objective.trunk.embed_tokens(mx.array(context))
        embeddings = mx.concatenate([embeddings[:start], vectors.astype(embeddings.dtype),
                                      embeddings[start + len(placeholder_ids):]])
        tokens, content = [], []
        reason = "length"
        for token, _ in generate_step(mx.array(context), model, max_tokens=max_tokens,
                                      input_embeddings=embeddings):
            token = int(token)
            tokens.append(token)
            if token in tokenizer.eos_token_ids:
                reason = "stop"
                break
            content.append(token)
        output = tokenizer.decode(content)
        rows.append({**case.to_dict(), "output": output, "token_ids": tokens,
                     "finish_reason": reason, "num_tokens": len(tokens),
                     "correct": output.strip() == case.answer and reason == "stop",
                     "copy": output.strip() == case.question, "empty": not output.strip()})
    return _evaluation_metrics(rows, perf_counter() - begun), rows
