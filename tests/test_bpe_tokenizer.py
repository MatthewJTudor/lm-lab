from lm_lab.config.load import TokenizerConfig
from lm_lab.tokenization.build import build_tokenizer
from lm_lab.tokenization.bpe import (
    BPETokenizer,
    inspect_chunks,
    bytes_to_unicode,
    unicode_to_bytes,
    _chunk_to_mapped_tokens,
    _chunk_text_to_mapped_tokens,
    _get_pair_counts,
    _merge_pair_in_chunk,
    _merge_pair_in_chunks,
)

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

from lm_lab.tokenization.bpe import bytes_to_unicode, unicode_to_bytes


def test_bytes_to_unicode_covers_all_bytes() -> None:
    enc = bytes_to_unicode()

    assert len(enc) == 256
    assert set(enc.keys()) == set(range(256))

def test_bytes_to_unicode_values_are_unique() -> None:
    enc = bytes_to_unicode()

    assert len(set(enc.values())) == 256

def test_unicode_to_bytes_is_inverse_mapping() -> None:
    enc = bytes_to_unicode()
    dec = unicode_to_bytes()

    for b, ch in enc.items():
        assert dec[ch] == b

def test_byte_unicode_roundtrip_full_range() -> None:
    enc = bytes_to_unicode()
    dec = unicode_to_bytes()

    original = bytes(range(256))
    mapped = "".join(enc[b] for b in original)
    recovered = bytes(dec[ch] for ch in mapped)

    assert recovered == original

def test_byte_unicode_roundtrip_utf8_text() -> None:
    enc = bytes_to_unicode()
    dec = unicode_to_bytes()

    text = "naïve café 🙂 “quote”"
    raw = text.encode("utf-8")

    mapped = "".join(enc[b] for b in raw)
    recovered = bytes(dec[ch] for ch in mapped)

    assert recovered == raw
    assert recovered.decode("utf-8") == text

def test_chunk_to_mapped_tokens_matches_utf8_byte_length() -> None:
    chunk = "naïve 🙂"
    tokens = _chunk_to_mapped_tokens(chunk)

    assert len(tokens) == len(chunk.encode("utf-8"))

def test_chunk_to_mapped_tokens_roundtrip_utf8() -> None:
    dec = unicode_to_bytes()

    chunk = "naïve café 🙂"
    tokens = _chunk_to_mapped_tokens(chunk)

    recovered = bytes(dec[tok] for tok in tokens).decode("utf-8")
    assert recovered == chunk

def test_chunk_text_to_mapped_tokens_matches_chunk_count() -> None:
    text = "Alice said:\nHello"
    chunks = inspect_chunks(text)
    mapped = _chunk_text_to_mapped_tokens(text)

    assert len(mapped) == len(chunks)

def test_chunk_text_to_mapped_tokens_preserves_utf8_lengths_per_chunk() -> None:
    text = 'Alice said:\n"Hello" 🙂'
    chunks = inspect_chunks(text)
    mapped = _chunk_text_to_mapped_tokens(text)

    assert len(mapped) == len(chunks)

    for chunk, tokens in zip(chunks, mapped):
        assert len(tokens) == len(chunk.encode("utf-8"))

def test_chunk_text_to_mapped_tokens_roundtrip_full_text() -> None:
    dec = unicode_to_bytes()

    text = 'Alice said:\n"Hello" 🙂'
    mapped_chunks = _chunk_text_to_mapped_tokens(text)

    flat = [tok for chunk in mapped_chunks for tok in chunk]
    recovered = bytes(dec[tok] for tok in flat).decode("utf-8")

    assert recovered == text

def test_get_pair_counts_supports_mapped_string_tokens() -> None:
    chunks = [["a", "b", "a"], ["a", "b"]]

    counts = _get_pair_counts(chunks)

    assert counts[("a", "b")] == 2
    assert counts[("b", "a")] == 1

def test_merge_pair_in_chunk_supports_mapped_string_tokens() -> None:
    tokens = ["a", "b", "c", "b", "c"]
    pair = ("b", "c")

    merged = _merge_pair_in_chunk(tokens, pair)

    assert merged == ["a", "bc", "bc"]

def test_merge_pair_in_chunks_supports_mapped_string_tokens() -> None:
    chunks = [["a", "b", "c"], ["b", "c", "d"]]
    pair = ("b", "c")

    merged = _merge_pair_in_chunks(chunks, pair)

    assert merged == [["a", "bc"], ["bc", "d"]]