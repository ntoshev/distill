"""distil — visualize per-token likelihood of a text/rollout according to an LLM."""

from distil.scoring import ScoreResult, TokenScore, TopToken, score_text
from distil.templating import BuiltText, build_chat, build_raw, build_rollout

__all__ = [
    "BuiltText",
    "ScoreResult",
    "TokenScore",
    "TopToken",
    "build_chat",
    "build_raw",
    "build_rollout",
    "score_text",
]
