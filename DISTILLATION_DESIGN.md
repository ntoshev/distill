# LoRA context distillation

## Original request

> propose a CLI update that would allow distillation training (OPSD, internalizing context), fitting the current CLI. the distillation prompt should be external (provided by the user) or minimal (if there is a good one) for now, I expect it to vary a lot.
>
> initial use case is e.g. to get a wiki article and internalize it, then be able to answer factual questions. distillation should be in LoRA. ask questions if you want

## Proposal

Add `--train PATH` as a fourth input mode alongside `--completion`, `--rollout`,
and `--sample`. Training performs on-policy self-distillation into a LoRA
adapter.

The user supplies:

- a plain-text file containing one training question per non-empty line;
- `--teacher-system @file`, containing both the distillation instructions and
  the privileged context;
- an output directory for the LoRA adapter.

This deliberately leaves question generation and prompt construction outside
the training command. Those concerns vary substantially and can be added later
without changing the training loop.

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --train questions.txt \
  --teacher-system @teacher.txt \
  --adapter-path adapters/wiki-article \
  --iters 100 \
  --learning-rate 1e-5 \
  --lora-rank 16 \
  --max-tokens 256 \
  --temperature 1.0 \
  --no-thinking
```

`questions.txt` might contain:

```text
When was the organization founded?
Who were its principal founders?
What event led to its dissolution?
```

`teacher.txt` is completely user-controlled. A minimal version is:

```text
Use the following private reference as authoritative evidence when answering
the user. Do not mention the reference in the answer.

<reference>
[article text]
</reference>
```

The student sees the optional `--system` prompt and a question from
`questions.txt`. It never sees `--teacher-system`.

## Training objective

For a question \(x\), the current LoRA student samples a response
\(\hat y\) without the privileged context. The frozen base model then evaluates
the same response prefixes while conditioned on the teacher system prompt
\(c\).

Only the LoRA parameters \(\phi\) are trained:

\[
\mathcal L(\phi)
= \frac{1}{|\hat y|}\sum_{t=1}^{|\hat y|}
D_{\mathrm{KL}}\!\left(
  p_T(\cdot \mid c,x,\hat y_{<t})
  \,\middle\|\,
  p_{S,\phi}(\cdot \mid x,\hat y_{<t})
\right).
\]

The teacher does not generate a replacement answer. It provides a dense
next-token target along the student's own sampled trajectory. Sampling uses the
current adapter every iteration, making the procedure on-policy.

V1 uses full-vocabulary forward KL, \(D_{\mathrm{KL}}(p_T\|p_S)\), only. This
matches the existing visualization and the optimized
`mlx_lm.tuner.losses.kl_div_loss` path. Reverse KL and JSD are deferred until
there is evidence that the extra choice is useful.

## CLI changes

New flags:

- `--train PATH`: one prompt per non-empty line;
- `--adapter-path PATH`: output directory in training mode and adapter to load
  in scoring or sampling modes;
- `--iters N`: successful optimizer updates;
- `--learning-rate FLOAT`;
- `--lora-rank N`.

Existing flags reused by training:

- `--teacher-system`, and optionally `--teacher-model`;
- `--system` for student-side conditioning;
- `--max-tokens`, `--temperature`, `--seed`, and `--no-thinking`;
- `--save-rollout` to optionally retain sampled training trajectories.

Validation rules:

- exactly one of `--completion`, `--rollout`, `--sample`, or `--train`;
- `--train` requires `--teacher-system` and `--adapter-path`;
- `--teacher-prompt` is rejected with `--train`, because teacher and student
  should receive the same question;
- training-only flags are rejected outside `--train`;
- scoring-only output flags are rejected with `--train`.

No automatic question generation, loss selector, batch-size setting, gradient
accumulation setting, or detailed LoRA configuration is included in V1.

## Implementation

### Model and LoRA

1. Load and freeze the base model.
2. Convert the normal `mlx-lm` target layers to LoRA layers.
3. Generate each rollout with LoRA enabled.
4. Evaluate the teacher with the LoRA contribution disabled. With the current
   `mlx-lm` implementation this means temporarily setting each LoRA layer's
   scale to zero, evaluating the teacher logits, and then restoring the scales.
5. Compute the student logits with LoRA enabled and update only LoRA weights.

The teacher is therefore the initial base policy with privileged context, while
the student is the evolving adapter without that context. A second model copy
is unnecessary when `--teacher-model` is not specified.

Teacher logits must be evaluated before LoRA is re-enabled because MLX arrays
are lazy. They should also be computed outside the gradient closure.

### Alignment and loss

Extract the alignment portion of `score_text_pair` into a shared helper that:

1. tokenizes the student and teacher renderings;
2. identifies generated-token positions;
3. verifies that generated token IDs match exactly;
4. returns the two ID sequences and aligned predictive positions.

The visual scorer continues to use this helper. Training gathers the aligned
student and teacher logits and calls:

```python
kl_div_loss(student_logits, teacher_logits)
```

In the installed `mlx-lm` implementation this computes
\(D_{\mathrm{KL}}(p_T\|p_S)\) and sends gradients only through the first
argument.

An alignment failure should skip the rollout with a warning rather than abort
the run. `--iters` counts successful optimizer updates. Length-truncated
rollouts remain valid training data and retain their finish reason when saved.

### Adapter output

The adapter directory contains standard `mlx-lm` files:

```text
adapters/wiki-article/
  adapters.safetensors
  adapter_config.json
```

`adapter_config.json` records at least the fine-tune type, number of adapted
layers, and LoRA rank, scale, and dropout. This allows the existing model loader
to use the result directly.

If `--save-rollout` is supplied, each sampled trajectory is appended using the
existing verifiers-compatible `results.jsonl` format.

## Using and evaluating the adapter

The trained adapter works with the existing sampling mode:

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --adapter-path adapters/wiki-article \
  --prompt "Who founded the organization?" \
  --sample \
  --temperature 0 \
  --no-thinking
```

It also works with teacher-mode visualization:

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --adapter-path adapters/wiki-article \
  --prompt "Who founded the organization?" \
  --sample \
  --teacher-system @teacher.txt \
  --stats \
  --no-thinking
```

When an adapter and same-base teacher are combined, the student pass must use
LoRA while the teacher pass disables it. Otherwise both branches would use the
trained adapter and the reported KL would not measure context internalization.

Useful V1 checks are:

- factual answers improve on held-out questions;
- mean teacher/student KL decreases after training;
- the adapter does not substantially change unrelated answers;
- saved training rollouts can be reopened with the existing `--rollout` mode.

## Deferred work

- automatic question generation from a document;
- JSONL datasets with per-example context;
- reverse KL, JSD, clipping, or top-k approximations;
- batched generation and gradient accumulation;
- resuming training and richer metrics;
- context chunking for documents that exceed the model context window.

