from lm_lab.config.load import TokenizerConfig
from lm_lab.tokenization.bpe import BPETokenizer
from lm_lab.tokenization.build import build_tokenizer


def test_bpe_roundtrip_ascii() -> None:
    text = "hello world"
    tok = BPETokenizer.build(text, vocab_size=64)

    ids = tok.encode(text)
    s = tok.decode(ids)

    assert s == text


def test_bpe_roundtrip_unicode() -> None:
    text = "naïve café 🙂"
    tok = BPETokenizer.build(text, vocab_size=128)

    ids = tok.encode(text)
    s = tok.decode(ids)

    assert s == text


def test_bpe_is_deterministic() -> None:
    text = "banana bandana banana"
    tok1 = BPETokenizer.build(text, vocab_size=64)
    tok2 = BPETokenizer.build(text, vocab_size=64)

    assert tok1.merges == tok2.merges
    assert tok1.stoi == tok2.stoi


def test_bpe_factory_mode() -> None:
    text = "hello world"
    cfg = TokenizerConfig(mode="bpe", bpe_vocab_size=64)

    tok = build_tokenizer(cfg, text)
    ids = tok.encode(text)
    s = tok.decode(ids)

    assert s == text


def test_bpe_no_empty_encoding_for_nonempty_text() -> None:
    text = "hello"
    tok = BPETokenizer.build(text, vocab_size=32)

    ids = tok.encode(text)
    assert len(ids) > 0

def test_bpe_preserves_space_prefixed_chunks_roundtrip() -> None:
    text = "Alice said hello."
    tok = BPETokenizer.build(text, vocab_size=64)

    ids = tok.encode(text)
    s = tok.decode(ids)

    assert s == text


def test_bpe_roundtrip_with_newlines_and_punctuation() -> None:
    text = 'Alice said:\n"Hello!"\n'
    tok = BPETokenizer.build(text, vocab_size=128)

    ids = tok.encode(text)
    s = tok.decode(ids)

    assert s == text