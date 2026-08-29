"""Character and byte-pair tokenizers, kept small enough to read in one sitting.

Run directly to compare how many positions each encoding costs on the same text.
"""

from __future__ import annotations

import collections
import pathlib


class CharTokenizer:
    """The honest baseline: no training step, vocabulary is whatever appeared."""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class BPETokenizer:
    """Byte-pair encoding: merge the most frequent adjacent pair, repeat.

    Trains on raw bytes so any input is encodable, which is the property that
    makes the tail of the distribution survivable.
    """

    def __init__(self, num_merges: int = 256) -> None:
        self.num_merges = num_merges
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @staticmethod
    def _pair_counts(ids: list[int]) -> collections.Counter:
        return collections.Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str) -> "BPETokenizer":
        ids = list(text.encode("utf-8"))
        for k in range(self.num_merges):
            counts = self._pair_counts(ids)
            if not counts:
                break
            pair, count = counts.most_common(1)[0]
            if count < 2:
                break
            new_id = 256 + k
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> list[int]:
        ids = list(s.encode("utf-8"))
        # Apply merges in the order they were learned; earlier merges are more frequent.
        for pair, new_id in self.merges.items():
            ids = self._merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def sample_text() -> str:
    """TinyShakespeare if it has been downloaded, otherwise a bundled fallback."""
    path = pathlib.Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise SystemExit(
        "data/tinyshakespeare.txt is missing. Fetch it with:\n"
        "  curl -sSfL -o data/tinyshakespeare.txt \\\n"
        "    https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt\n"
        "A synthetic fallback is deliberately not provided: repeated filler text "
        "makes BPE look absurdly good and produces numbers worth nothing."
    )


def main() -> None:
    text = sample_text()
    char = CharTokenizer(text)
    bpe = BPETokenizer(num_merges=256).train(text)

    char_ids = char.encode(text)
    bpe_ids = bpe.encode(text)

    assert char.decode(char_ids) == text, "char tokenizer must round-trip"
    assert bpe.decode(bpe_ids) == text, "bpe tokenizer must round-trip"

    print(f"text length (characters)  {len(text):>8,}")
    print(f"char vocab                {char.vocab_size:>8,}")
    print(f"char positions            {len(char_ids):>8,}")
    print(f"bpe vocab                 {bpe.vocab_size:>8,}")
    print(f"bpe positions             {len(bpe_ids):>8,}")
    print(f"compression ratio         {len(char_ids) / len(bpe_ids):>8.2f}x")
    print(f"attention cost ratio      {(len(char_ids) / len(bpe_ids)) ** 2:>8.2f}x")


if __name__ == "__main__":
    main()
