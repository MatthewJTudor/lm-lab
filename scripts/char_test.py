from lm_lab.tokenization.char import CharTokenizer

text = "hello world\n"
tok = CharTokenizer.build(text)

print("Vocab size:", tok.vocab_size)
print("Vocab:", tok.itos)