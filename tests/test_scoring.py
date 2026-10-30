import mlx.core as mx

from distill.scoring import score_text


class BosTokenizer:
    bos_token_id = 99

    def _tokenizer(self, text, return_offsets_mapping, add_special_tokens):
        assert return_offsets_mapping
        assert not add_special_tokens
        start = len("<bos>") if text.startswith("<bos>") else 0
        ids = ([self.bos_token_id] if start else []) + [ord(char) for char in text[start:]]
        offsets = ([(0, start)] if start else []) + [
            (i, i + 1) for i in range(start, len(text))
        ]
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids):
        return "".join("<bos>" if i == self.bos_token_id else chr(i) for i in ids)


class ZeroModel:
    def __init__(self):
        self.seen = None

    def __call__(self, ids):
        self.seen = ids.tolist()
        return mx.zeros((ids.shape[0], ids.shape[1], 128))


def test_raw_scoring_adds_bos_as_invisible_context():
    tokenizer = BosTokenizer()
    model = ZeroModel()
    result = score_text(model, tokenizer, "ab", [(1, 2)])
    assert model.seen == [[99, ord("a"), ord("b")]]
    assert [token.text for token in result.tokens] == ["a", "b"]
    assert [token.text for token in result.scored] == ["b"]
    assert result.warnings == []


def test_rendered_bos_is_not_duplicated():
    tokenizer = BosTokenizer()
    model = ZeroModel()
    score_text(model, tokenizer, "<bos>ab", [(6, 7)])
    assert model.seen == [[99, ord("a"), ord("b")]]


def test_generated_scoring_preserves_replacement_provenance(monkeypatch):
    import distill.scoring as scoring
    from distill.sample import Sample
    from distill.progress import _token_payload
    from distill.render import to_json_dict

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [99]

        def decode(self, ids):
            return ''.join({1: 'A', 2: 'B'}[i] for i in ids)

    monkeypatch.setattr(scoring, 'score_ids', lambda *a, **kw: ({1: -1., 2: -2.}, {}))
    sample = Sample('prompt', 'AB', 'AB', 'stop', 2, token_ids=[1, 2],
                    teacher_sampled=[False, True])
    result = scoring.score_generated(None, Tokenizer(), sample)
    assert [t.teacher_sampled for t in result.scored] == [False, True]
    assert _token_payload(result.scored[1])['teacher_sampled'] is True
    assert to_json_dict(result)['tokens'][2]['teacher_sampled'] is True
