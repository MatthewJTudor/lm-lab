from __future__ import annotations

import argparse
from pathlib import Path

from lm_lab.tokenization.bpe import BPETokenizer, inspect_chunks


def _safe_text(b: bytes) -> str:
    decoded = b.decode("utf-8", errors="replace")
    return f"{decoded!r} [{b!r}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BPE chunking, merges, and token usage")
    parser.add_argument("--text_path", type=str, required=True, help="Path to input text")
    parser.add_argument("--vocab_size", type=int, default=128, help="Target BPE vocab size")
    parser.add_argument("--top_n", type=int, default=20, help="How many rows to show per section")
    parser.add_argument(
        "--preview_chars",
        type=int,
        default=300,
        help="How many characters of the corpus to preview for chunk inspection",
    )
    args = parser.parse_args()

    text = Path(args.text_path).read_text(encoding="utf-8")
    tok = BPETokenizer.build(text, vocab_size=args.vocab_size)

    preview_text = text[: args.preview_chars]
    chunks = inspect_chunks(preview_text)

    print("== config ==")
    print(f"text_path     : {args.text_path}")
    print(f"vocab_size    : {args.vocab_size}")
    print(f"top_n         : {args.top_n}")
    print(f"preview_chars : {args.preview_chars}")
    print(f"learned_vocab : {tok.vocab_size}")
    print(f"merge_count   : {len(tok.merges)}")

    print("\n== chunks ==")
    for chunk in chunks[: args.top_n]:
        print(repr(chunk))

    print("\n== merges ==")
    for stat in tok.inspect_merges(args.top_n):
        print(
            f"{stat.rank:>3} | "
            f"{_safe_text(stat.left)} + {_safe_text(stat.right)} "
            f"-> {_safe_text(stat.merged)}"
        )

    print("\n== token frequencies ==")
    for stat in tok.inspect_token_frequencies(text, args.top_n):
        print(
            f"{stat.count:>6} | "
            f"len={stat.byte_length:>2} | "
            f"{_safe_text(stat.token)}"
        )

    print("\n== vocab token lengths (top by byte length) ==")
    vocab_lengths = tok.inspect_vocab_token_lengths()
    vocab_lengths.sort(key=lambda x: (-x[1], x[0]))

    for tok_bytes, n in vocab_lengths[: args.top_n]:
        print(f"len={n:>2} | {_safe_text(tok_bytes)}")


if __name__ == "__main__":
    main()