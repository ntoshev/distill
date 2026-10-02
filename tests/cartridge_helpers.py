"""Small real Qwen3 fixtures: attention, RoPE, and autodiff run on MLX."""

import mlx.core as mx
from mlx_lm.models.qwen3 import Model, ModelArgs


class CharacterTokenizer:
    clean_up_tokenization_spaces = False
    bos_token_id = None
    eos_token_id = 0
    eos_token_ids = {0}
    chat_template = "character-chat-v1"

    def __init__(self):
        self._tokenizer = self

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(i for i in ids if i != 0).decode("utf-8", errors="replace") + (
            "" if skip_special_tokens else "<eos>" * ids.count(0)
        )

    def __call__(self, text, **kwargs):
        ids = self.encode(text)
        if len(ids) != len(text):
            raise ValueError("use ASCII offsets in this fixture")
        return {
            "input_ids": ids,
            "offset_mapping": [(i, i + 1) for i in range(len(ids))],
        }

    def get_vocab(self):
        return {chr(i): i for i in range(1, 256)}

    def to_str(self):
        return "utf8-character-tokenizer-v1"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        text += "assistant: " if add_generation_prompt else ""
        return self.encode(text) if tokenize else text


def tiny_model(seed=19):
    mx.random.seed(seed)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=32,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=256,
            num_key_value_heads=1,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            head_dim=8,
            tie_word_embeddings=True,
        )
    )
    mx.eval(model.parameters())
    return model, CharacterTokenizer()
