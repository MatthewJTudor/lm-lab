from lm_lab.core.model import TransformerLM, TransformerLMConfig


def test_weight_tying_enabled_shares_weight_object() -> None:
    cfg = TransformerLMConfig(
        vocab_size=50,
        max_seq_len=8,
        d_model=32,
        n_layers=1,
        tie_embeddings=True,
    )
    m = TransformerLM(cfg)
    assert m.head.weight is m.token_emb.embedding.weight


def test_weight_tying_disabled_not_shared() -> None:
    cfg = TransformerLMConfig(
        vocab_size=50,
        max_seq_len=8,
        d_model=32,
        n_layers=1,
        tie_embeddings=False,
    )
    m = TransformerLM(cfg)
    assert m.head.weight is not m.token_emb.embedding.weight