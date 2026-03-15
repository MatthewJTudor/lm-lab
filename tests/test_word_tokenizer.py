from lm_lab.tokenization.word import WordTokenizer
from lm_lab.tokenization.build import build_tokenizer
from lm_lab.config.load import TokenizerConfig


def test_word_tokenizer_build_and_vocab_size() -> None:
    text = "Alice was here.\nAlice smiled."
    tok = WordTokenizer.build(text)

    assert tok.vocab_size >= 3
    assert tok.stoi["<unk>"] == 0
    assert tok.stoi["<bos>"] == 1
    assert tok.stoi["<eos>"] == 2


def test_word_tokenizer_encode_returns_ids() -> None:
    text = "Alice was here."
    tok = WordTokenizer.build(text)

    ids = tok.encode("Alice was here.")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_word_tokenizer_decode_readable_spacing() -> None:
    text = "Alice was here.\nAlice smiled."
    tok = WordTokenizer.build(text)

    ids = tok.encode("Alice smiled.")
    s = tok.decode(ids)

    assert s == "Alice smiled."


def test_word_tokenizer_unknown_maps_to_unk() -> None:
    text = "Alice was here."
    tok = WordTokenizer.build(text)

    ids = tok.encode("Bob was here.")
    assert ids[0] == tok.stoi["<unk>"]


def test_build_tokenizer_word_mode() -> None:
    text = "Alice was here."
    cfg = TokenizerConfig(mode="word")

    tok = build_tokenizer(cfg, text)
    ids = tok.encode("Alice was here.")
    s = tok.decode(ids)

    assert s == "Alice was here."