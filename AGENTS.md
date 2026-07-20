# Project instructions

- Render math properly.
- Build intuition when explaining concepts.

## Distillation visualizations

- Default to a dense, side-by-side checkpoint grid rather than a dashboard or chart.
- Use one row per sampled checkpoint and columns for the step, target completion, and affected canary answer.
- Color every token by its own model probability while preserving the model's actual token boundaries.
- Show the canary answer that is active at that checkpoint: the correct answer before leakage, the leaked answer during contamination, and the correct answer again after recovery.
- Never color generated text using probabilities from a different string. Score the displayed string, or label an unscored generated output separately.
- Put the geometric-mean token probability beside each sequence when useful.
- Mark only important transitions in the step column, such as `base`, `learned`, `leak`, and `recovered`.
- Keep the grid visually tight: small text, minimal token padding, narrow gaps, and no secondary output lines unless needed for correctness.
- Start with the target and the canary that changed. Add other canaries, plots, controls, cards, or narrative only when requested.
- Use real checkpoint data. Do not invent or interpolate token probabilities.
- For an inline Codex view, resolve the active thread ID from request metadata before writing; do not infer it from the newest directory.
- Read `CODEX_THREAD_ID` and write the fragment directly to the global thread-scoped directory `/Users/nik/.codex/visualizations/YYYY/MM/DD/$CODEX_THREAD_ID/`. Never write inline visualizations to the project's local `.codex/visualizations/` directory, because the Codex host will not resolve them there.
