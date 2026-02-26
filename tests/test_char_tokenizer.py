from lm_lab.tokenization.char import CharTokenizer, CharTokenizerConfig


def test_char_tokenizer_roundtrip() -> None:
    text = "hello"
    tok = CharTokenizer.build(text, cfg=CharTokenizerConfig(add_bos=True, add_eos=True))

    ids = tok.encode("hello")
    s = tok.decode(ids)

    assert s == "hello"
    assert tok.vocab_size >= 4  # includes specials


def test_char_tokenizer_unknown() -> None:
    tok = CharTokenizer.build("abc")
    ids = tok.encode("aZc")
    # Z should become <unk>, which decodes to replacement char
    s = tok.decode(ids)
    assert s == "a�c"