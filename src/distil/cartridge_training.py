"""FP32 cartridge-only Adam and differentiable, bounded vocabulary forwards."""

from pathlib import Path
import json
import math
from time import perf_counter
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.cache import ConcatenateKVCache
from distil.cartridge import Cartridge, validate_model, digest
from distil.losses import kl_loss


class CartridgeTrainingSession:
    def __init__(
        self,
        model,
        tokenizer,
        cartridge: Cartridge,
        *,
        learning_rate=0.005,
        warmup=5,
        gradient_clip=1.0,
        objective="ntp",
        teacher_cartridge=None,
        logit_chunk_size=32,
        **fact_options,
    ):
        validate_model(model)
        if (
            not math.isfinite(learning_rate)
            or learning_rate <= 0
            or warmup < 0
            or gradient_clip <= 0
            or logit_chunk_size <= 0
        ):
            raise ValueError("invalid cartridge optimizer settings")
        if objective not in ("ntp", "kd"):
            raise ValueError("unsupported cartridge source objective")
        self.model, self.tokenizer, self.cartridge = model, tokenizer, cartridge
        self.model.freeze()
        self.model.eval()
        self.parameters = {
            f"{kind}{i}": value[:, :, 1:].astype(mx.float32)
            for i, pair in enumerate(cartridge.tensors)
            for kind, value in zip(("k", "v"), pair)
        }
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.optimizer.init(self.parameters)
        self.learning_rate, self.warmup, self.gradient_clip = (
            learning_rate,
            warmup,
            gradient_clip,
        )
        self.iteration = 0
        self.objective = (
            "forward-kl-skd" if fact_options.get("teacher_system") else objective
        )
        self.logit_chunk_size = logit_chunk_size
        self.teacher_cartridge = teacher_cartridge
        self.fact_options = fact_options
        self.source_loss_and_grad = mx.value_and_grad(self._source_loss)
        self.trajectory_loss_and_grad = mx.value_and_grad(self._loss_sum)
        mx.eval(self.parameters, self.optimizer.state)

    def inference_cartridge(self):
        tensors = [
            entry.state
            for entry in self.cartridge.new_cache(parameters=self.parameters)
        ]
        return Cartridge(
            tensors, self.cartridge.model_identity, metadata=self.cartridge.metadata
        )

    @property
    def inference_policy(self):
        from distil.backends import CartridgePolicy

        return CartridgePolicy(
            self.model,
            self.tokenizer,
            self.cartridge.model_identity,
            self.inference_cartridge(),
        )

    def _loss_sum(
        self, parameters, ids, target_positions, target_ids, teacher_rows=None
    ):
        """Checkpoint pure cache transitions; never detach prefix or prompt state.

        Only target hidden rows pass through the vocabulary head. Recomputation
        bounds saved activation/logit allocations without truncating gradients.
        """
        states = [
            c.state
            for c in self.cartridge.new_cache(
                differentiable=True, parameters=parameters
            )
        ]
        rows = [p - 1 for p in target_positions]
        loss = mx.array(0.0, mx.float32)
        target_cursor = 0
        for start in range(0, rows[-1] + 1, self.logit_chunk_size):
            end = min(start + self.logit_chunk_size, rows[-1] + 1)
            selected = [r - start for r in rows if start <= r < end]

            # Bind row selection per checkpoint (backward runs after this loop).
            def bind_forward(selected_rows):
                def bound(piece, state):
                    cache = []
                    for pair in state:
                        entry = ConcatenateKVCache()
                        entry.state = pair
                        cache.append(entry)
                    hidden = self.model.model(piece[None], cache=cache)
                    if selected_rows:
                        hidden = hidden[:, mx.array(selected_rows)]
                        logits = (
                            self.model.model.embed_tokens.as_linear(hidden)
                            if self.model.args.tie_word_embeddings
                            else self.model.lm_head(hidden)
                        )[0].astype(mx.float32)
                    else:
                        logits = mx.zeros((0, self.model.args.vocab_size), mx.float32)
                    return logits, [entry.state for entry in cache]

                return bound

            logits, states = mx.checkpoint(bind_forward(selected))(
                mx.array(ids[start:end]), states
            )
            count = len(selected)
            if count:
                if teacher_rows is None:
                    chosen = mx.take_along_axis(
                        logits,
                        mx.array(target_ids[target_cursor : target_cursor + count])[
                            :, None
                        ],
                        axis=-1,
                    )[:, 0]
                    loss = loss + (mx.logsumexp(logits, axis=-1) - chosen).sum()
                else:
                    if logits.shape[-1] != teacher_rows.shape[-1]:
                        raise ValueError("teacher/student vocabulary mismatch")
                    loss = (
                        loss
                        + kl_loss(
                            logits,
                            teacher_rows[target_cursor : target_cursor + count],
                            "forward",
                        ).sum()
                    )
                target_cursor += count
        return loss

    def _source_loss(self, parameters, window, teacher_rows=None):
        return self._loss_sum(
            parameters,
            window.ids,
            window.target_positions,
            window.target_ids,
            teacher_rows,
        )

    def _update(self, gradients):
        norm = mx.sqrt(sum(mx.sum(g * g) for g in gradients.values()))
        mx.eval(norm, gradients)
        if not math.isfinite(norm.item()):
            raise ValueError("nonfinite cartridge gradient; update refused")
        scale = mx.minimum(1.0, self.gradient_clip / mx.maximum(norm, 1e-12))
        gradients = {name: grad * scale for name, grad in gradients.items()}
        self.optimizer.learning_rate = self.learning_rate * min(
            1.0, (self.iteration + 1) / max(1, self.warmup)
        )
        self.parameters = self.optimizer.apply_gradients(gradients, self.parameters)
        mx.eval(self.parameters, self.optimizer.state)
        self.iteration += 1
        return norm.item()

    def step_windows(self, windows, *, teacher_prefix=None):
        if not windows:
            raise ValueError("source batch must not be empty")
        from distil.prefill import select_logits

        started = perf_counter()
        gradients = {
            name: mx.zeros_like(value) for name, value in self.parameters.items()
        }
        losses = 0.0
        count = sum(len(w.target_ids) for w in windows)
        for window in windows:
            teacher = None
            if self.objective == "kd":
                if teacher_prefix is None:
                    raise ValueError(
                        "source KD requires the frozen full-document teacher prefix"
                    )
                prefix = self.tokenizer.encode(
                    teacher_prefix + window.prompt, add_special_tokens=False
                )
                ids = prefix + window.target_ids
                teacher = select_logits(
                    self.model,
                    ids,
                    list(range(len(prefix) - 1, len(ids) - 1)),
                    cache_factory=self.teacher_cartridge.new_cache
                    if self.teacher_cartridge
                    else None,
                )
            loss, grads = self.source_loss_and_grad(self.parameters, window, teacher)
            mx.eval(loss, grads)
            losses += loss.item()
            gradients = {name: gradients[name] + grads[name] for name in gradients}
            mx.eval(gradients)
        norm = self._update({name: grad / count for name, grad in gradients.items()})
        return {
            "iteration": self.iteration,
            "loss": losses / count,
            "num_tokens": count,
            "batch_size": len(windows),
            "reduction": "token-weighted",
            "gradient_norm": norm,
            "train_seconds": perf_counter() - started,
            "peak_memory_bytes": mx.get_peak_memory(),
        }

    def step_batch(self, prompts, *, capture_feedback=False, **kwargs):
        from distil.sample import sample_completion, sample_speculative_completion
        from distil.templating import prompt_messages, render_generation_prompt
        from distil.prefill import select_logits
        from distil.training import TrainStep

        if not prompts or not self.fact_options.get("teacher_system"):
            raise ValueError(
                "fact learning requires prompts and a source-authoritative teacher context"
            )
        options = self.fact_options
        thinking = options.get("enable_thinking")
        top_k, top_p = (
            options.get("speculative_top_k", 3),
            options.get("speculative_top_p"),
        )
        if top_p is not None and "speculative_top_k" not in options:
            top_k = None
        sampling = dict(
            max_tokens=options.get("max_tokens", 64),
            temperature=options.get("temperature", 1.0),
            seed=options.get("seed", 0) + self.iteration,
            enable_thinking=thinking,
        )
        memory = self.inference_cartridge()
        gradients = {
            name: mx.zeros_like(value) for name, value in self.parameters.items()
        }
        steps = []
        total_started = perf_counter()
        rollout_total = distill_total = 0.0
        for index, prompt in enumerate(prompts):
            messages = prompt_messages(prompt, options.get("student_system"))
            teacher_messages = prompt_messages(prompt, options["teacher_system"])
            started = perf_counter()
            if top_k is not None or top_p is not None:
                sample = sample_speculative_completion(
                    self.model,
                    self.tokenizer,
                    messages,
                    self.model,
                    self.tokenizer,
                    teacher_messages,
                    top_k=top_k,
                    top_p=top_p,
                    cache_factory=memory.new_cache,
                    teacher_cache_factory=self.teacher_cartridge.new_cache
                    if self.teacher_cartridge
                    else None,
                    capture_teacher_logits=True,
                    **sampling,
                )
            else:
                sample = sample_completion(
                    self.model,
                    self.tokenizer,
                    messages,
                    cache_factory=memory.new_cache,
                    **sampling,
                )
            rollout_seconds = perf_counter() - started
            ids = sample.token_ids
            if not ids:
                raise ValueError("fact rollout produced no continuation tokens")
            student_prefix = self.tokenizer.encode(
                sample.prompt_text, add_special_tokens=False
            )
            teacher_prefix = self.tokenizer.encode(
                render_generation_prompt(self.tokenizer, teacher_messages, thinking),
                add_special_tokens=False,
            )
            teacher = sample.teacher_logits
            started = perf_counter()
            if teacher is None:
                teacher_ids = teacher_prefix + ids
                teacher = select_logits(
                    self.model,
                    teacher_ids,
                    list(range(len(teacher_prefix) - 1, len(teacher_ids) - 1)),
                    cache_factory=self.teacher_cartridge.new_cache
                    if self.teacher_cartridge
                    else None,
                )
            loss, grads = self.trajectory_loss_and_grad(
                self.parameters,
                student_prefix + ids,
                list(range(len(student_prefix), len(student_prefix) + len(ids))),
                ids,
                teacher,
            )
            mx.eval(loss, grads)
            mean = loss.item() / len(ids)
            gradients = {
                name: gradients[name] + grads[name] / len(ids) / len(prompts)
                for name in gradients
            }
            mx.eval(gradients)
            distill_seconds = perf_counter() - started
            feedback = None
            if capture_feedback:
                logp = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
                probabilities = mx.take_along_axis(
                    logp, mx.array(ids)[:, None], axis=-1
                )[:, 0].tolist()
                feedback = [
                    {
                        "text": self.tokenizer.decode([token]),
                        "logprob": lp,
                        "probability": math.exp(lp),
                        "teacher_sampled": bool(
                            sample.teacher_sampled and sample.teacher_sampled[i]
                        ),
                    }
                    for i, (token, lp) in enumerate(zip(ids, probabilities))
                ]
            steps.append(
                TrainStep(
                    self.iteration + 1,
                    prompt,
                    mean,
                    len(ids),
                    sample.finish_reason,
                    sample.content,
                    speculative_top_k=top_k,
                    speculative_top_p=top_p,
                    accepted_tokens=sample.accepted_tokens,
                    replaced_tokens=sample.replaced_tokens,
                    batch_size=len(prompts),
                    batch_index=index,
                    rollout_seconds=rollout_seconds,
                    distill_seconds=distill_seconds,
                    teacher_feedback=feedback,
                )
            )
            rollout_total += rollout_seconds
            distill_total += distill_seconds
        started = perf_counter()
        self._update(gradients)
        optimizer_seconds = perf_counter() - started
        mean_loss = sum(s.loss for s in steps) / len(steps)
        for step in steps:
            step.batch_loss = mean_loss
            step.optimizer_seconds = optimizer_seconds
            step.batch_rollout_seconds = rollout_total
            step.batch_distill_seconds = distill_total
            step.batch_train_seconds = perf_counter() - total_started
        return steps

    def save(self, path: Path):
        return self.inference_cartridge().save(path)

    def save_resume(self, path: Path, state: dict):
        arrays = {f"masters.{name}": value for name, value in self.parameters.items()}
        arrays.update(
            {
                f"optimizer.{name}": value
                for name, value in tree_flatten(self.optimizer.state)
            }
        )
        arrays.update(
            {
                f"frozen.{i}.{kind}": value[:, :, :1]
                for i, pair in enumerate(self.cartridge.tensors)
                for kind, value in zip(("k", "v"), pair)
            }
        )
        arrays.update({f"rng.{i}": value for i, value in enumerate(mx.random.state)})
        mx.eval(arrays)
        mx.save_safetensors(str(path / "resume.safetensors"), arrays)
        payload = {
            "schema_version": 2,
            "backend": "cartridge",
            "iteration": self.iteration,
            "optimizer": {
                "learning_rate": self.learning_rate,
                "warmup": self.warmup,
                "gradient_clip": self.gradient_clip,
            },
            "objective": self.objective,
            "logit_chunk_size": self.logit_chunk_size,
            "state": state,
            "tensor_sha256": digest((path / "resume.safetensors").read_bytes()),
        }
        (path / "resume.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )

    def restore(self, path: Path):
        data = json.loads((path / "resume.json").read_text())
        if data.get("schema_version") != 2 or data.get("backend") != "cartridge":
            raise ValueError("unsupported cartridge optimizer resume schema")
        if data.get("tensor_sha256") != digest(
            (path / "resume.safetensors").read_bytes()
        ):
            raise ValueError("resume tensor checksum mismatch")
        if (
            data["optimizer"]
            != {
                "learning_rate": self.learning_rate,
                "warmup": self.warmup,
                "gradient_clip": self.gradient_clip,
            }
            or data["objective"] != self.objective
            or data["logit_chunk_size"] != self.logit_chunk_size
        ):
            raise ValueError("resume optimizer/recipe settings mismatch")
        arrays = mx.load(str(path / "resume.safetensors"))
        parameters = {name: arrays[f"masters.{name}"] for name in self.parameters}
        if any(
            value.dtype != mx.float32 or value.shape != self.parameters[name].shape
            for name, value in parameters.items()
        ):
            raise ValueError("invalid FP32 master resume tensors")
        for i, pair in enumerate(self.cartridge.tensors):
            for kind, value in zip(("k", "v"), pair):
                if not mx.array_equal(
                    arrays[f"frozen.{i}.{kind}"], value[:, :, :1]
                ).item():
                    raise ValueError("frozen-position resume tensor mismatch")
        self.parameters = parameters
        self.optimizer.state = tree_unflatten(
            [
                (name.removeprefix("optimizer."), value)
                for name, value in arrays.items()
                if name.startswith("optimizer.")
            ]
        )
        self.iteration = data["iteration"]
        if int(self.optimizer.state["step"].item()) != self.iteration:
            raise ValueError("optimizer counter mismatch")
        mx.random.state = [arrays[f"rng.{i}"] for i in range(len(mx.random.state))]
        return data["state"]
