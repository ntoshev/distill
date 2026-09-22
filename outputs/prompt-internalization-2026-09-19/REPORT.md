# Shared-prompt internalization: five adapters

Five independent LoRA adapters were trained from `mlx-community/Qwen3.5-4B-6bit` using 8, 16, 32, 64, and 128 existing Dolly questions and the selected shared 85–179-word prompt. No model generated or rewrote training questions.

## Held-out results

All results use the same 64 IFEval-derived evaluation questions. The score requires all mechanical constraints: English, section structure, bullet count and placement, word limits, required/forbidden words, and the exact final sentence. Semantic answer quality is not included.

| Variant | System prompt at inference | Pass | Truncated |
|---|---|---:|---:|
| Base | no | 0/64 (0.00%) | 2 |
| Base + explicit prompt | yes | 37/64 (57.81%) | 0 |
| 8-case adapter | no | 7/64 (10.94%) | 2 |
| 16-case adapter | no | 10/64 (15.62%) | 0 |
| 32-case adapter | no | 15/64 (23.44%) | 0 |
| 64-case adapter | no | 18/64 (28.12%) | 0 |
| 128-case adapter | no | 12/64 (18.75%) | 0 |

The highest observed adapter score was 18/64 from the 64-case adapter. This is partial internalization: all adapters remain below the explicitly prompted base. The 128-case adapter recovered the section structure and exact ending on all 64 questions, but only 20 answers reached 85 words and 39 included `because`. More training cases did not consistently improve full compliance under this fixed schedule.

The explicit-prompt control above was generated afresh through the same native MLX batching path as the adapters. The earlier calibrated prompt scored 39/64 via the HTTP evaluation runner. Greedy generation can vary with batching, prefill/cache implementation, and floating-point kernels. The 64 questions were previously used to calibrate the prompt, but none were used to train these adapters.

## Training

- Questions: 128 context-free open/general QA instructions from `databricks/databricks-dolly-15k`, selected with seed 42 and saved verbatim apart from outer whitespace. No Dolly reference answers were used.
- The selected system prompt was run once on those cases with the frozen base model. Exact generated token IDs and actual stop tokens were retained. Both passing and failing teacher responses were included.
- Teacher responses passed all mechanical checks on 51/128 cases. This experiment distills that teacher policy; it does not correct its instruction failures.
- Student inputs contain only the case question; teacher inputs include the shared system prompt. The same response continuation is scored under both contexts.
- Objective: full-vocabulary forward KL from the frozen prompted teacher to the student, including end-of-turn tokens. This uses the project’s `TrainingSession.step_samples`, not its document-to-question pipeline.
- Independent base initialization for each adapter; LoRA rank 16, last 16 model layers, scale 20, dropout 0, AdamW learning rate 0.0001, seed 0, logical batch size 4, 3 epochs.
- Training accumulates independent sequence gradients sequentially to bound memory. Generation uses native MLX batches with completion batch size 8 and prefill batch size 4. Temperature is zero and thinking is disabled.
- Subsets are nested prefixes of the frozen training case order. Larger datasets receive more optimizer updates because the number of epochs is fixed. All outputs are final-epoch adapters; evaluation scores did not select checkpoints.

| Cases | Updates | Teacher passes in subset | Training time | Adapter |
|---:|---:|---:|---:|---|
| 8 | 6 | 5/8 | 1.4 min | `adapters/cases-008` |
| 16 | 12 | 11/16 | 2.8 min | `adapters/cases-016` |
| 32 | 24 | 16/32 | 5.2 min | `adapters/cases-032` |
| 64 | 48 | 26/64 | 10.3 min | `adapters/cases-064` |
| 128 | 96 | 51/128 | 20.3 min | `adapters/cases-128` |

## Prompt

```text
Answer the user's question in English.

Follow all these rules in every response:
1. Use exactly two sections, beginning with `Section 1` and `Section 2`.
2. In Section 1, answer the question directly. In Section 2, provide supporting details or examples.
3. Include exactly three bullet points across the entire response, using `* ` as the bullet marker. Place them in Section 2.
4. Write at least 85 words but fewer than 180 words in total.
5. Include the word `because` at least once.
6. Never use the words `obviously`, `simply`, or `clearly`.
7. End with this exact sentence: `This concludes the response.`
```

## Per-check evaluation counts

| Variant | English | Sections | Three star bullets in Section 2 | 85+ words | <180 words | because | No forbidden words | Exact ending |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base-with-prompt | 64 | 64 | 64 | 37 | 64 | 64 | 62 | 64 |
| base-no-prompt | 64 | 0 | 0 | 64 | 3 | 19 | 51 | 0 |
| cases-008 | 63 | 49 | 19 | 30 | 62 | 19 | 63 | 62 |
| cases-016 | 64 | 48 | 34 | 34 | 63 | 44 | 62 | 63 |
| cases-032 | 64 | 63 | 57 | 32 | 64 | 33 | 64 | 64 |
| cases-064 | 64 | 64 | 54 | 25 | 64 | 40 | 64 | 64 |
| cases-128 | 64 | 64 | 57 | 20 | 64 | 39 | 62 | 64 |

## Use an adapter

```sh
uv run distil \
  --model mlx-community/Qwen3.5-4B-6bit \
  --adapter-path /Users/nik/helper-projects/distil/outputs/prompt-internalization-2026-09-19/adapters/cases-128 \
  --prompt "Why do leaves change color in autumn?" \
  --sample --temperature 0 --max-tokens 512 --no-thinking
```

To test internalization, omit the system prompt, as in this example. The standard MLX adapter directory contains `adapters.safetensors` and `adapter_config.json`.

## Re-run evaluation

From the project root, use the saved cases, prompt, and scorer:

```sh
PYTHONPATH=src python scripts/distill_prompt_responses.py --output-dir /Users/nik/helper-projects/distil/outputs/prompt-internalization-2026-09-19 --scorer-dir /Users/nik/helper-projects/distil/outputs/prompt-internalization-2026-09-19/frozen_eval --phase evaluate
python scripts/report_prompt_distillation.py /Users/nik/helper-projects/distil/outputs/prompt-internalization-2026-09-19
```

Completed evaluation rows are reused. For a fresh run, first move the `evaluations` directory to a new name. Training adapters and teacher responses are left in place.

## Verification and provenance

The audit verifies distinct adapter hashes, nested training IDs, exactly three appearances per training case, matching evaluation questions, and absent system prompts for every adapter evaluation. Core training changes passed 186 unit tests, two model checks, and all three relevant training performance benchmarks on the M3 Pro system. See `validation.json` for measurements.

Artifacts include the frozen prompt and questions, `data_manifest.json`, the unfiltered `teacher/responses.jsonl`, all training metrics and run configurations, exact generated evaluation tokens, the frozen scorer, and `summary.json`. Dolly source: https://huggingface.co/datasets/databricks/databricks-dolly-15k (CC BY-SA 3.0).
