from lm_lab.config.load import TokenizerConfig
from lm_lab.tokenization.bpe import BPETokenizer, inspect_chunks
from lm_lab.tokenization.build import build_tokenizer

import pytest

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

def test_bpe_inspect_merges_returns_ranked_list() -> None:
    tok = BPETokenizer.build("hello hello hello", vocab_size=32)
    merges = tok.inspect_merges(top_n=5)

    assert len(merges) <= 5
    for i, stat in enumerate(merges):
        assert stat.rank == i
        assert stat.merged == stat.left + stat.right

def test_bpe_inspect_token_frequencies_sorted() -> None:
    tok = BPETokenizer.build("aaaa bbbb aaaa", vocab_size=32)
    stats = tok.inspect_token_frequencies("aaaa bbbb aaaa", top_n=10)

    assert len(stats) > 0
    for stat in stats:
        assert stat.count > 0
        assert stat.byte_length == len(stat.token)

def test_bpe_inspect_vocab_token_lengths_excludes_specials() -> None:
    tok = BPETokenizer.build("hello world", vocab_size=32)
    lengths = tok.inspect_vocab_token_lengths()

    assert all(tok_bytes not in {tok.bos_token, tok.eos_token} for tok_bytes, _ in lengths)
    assert all(n > 0 for _, n in lengths)

def test_inspect_chunks_exposes_chunk_boundaries() -> None:
    text = "hello, world\nhi"
    chunks = inspect_chunks(text)

    assert chunks == ["hello", ",", " world", "\n", "hi"]

def test_bpe_inspect_merges_rejects_negative_top_n() -> None:
    tok = BPETokenizer.build("hello world", vocab_size=32)

    with pytest.raises(ValueError):
        tok.inspect_merges(top_n=-1)


def test_bpe_inspect_token_frequencies_rejects_negative_top_n() -> None:
    tok = BPETokenizer.build("hello world", vocab_size=32)

    with pytest.raises(ValueError):
        tok.inspect_token_frequencies("hello world", top_n=-1)

def test_bpe_build_rejects_small_vocab_size() -> None:
    with pytest.raises(ValueError):
        BPETokenizer.build("hello world", vocab_size=1)