"""Bigram language model: the smallest thing that generates text and improves.

Every prediction depends on exactly one previous token, so this is the floor
that every later page has to beat. Run directly to train and sample.
"""

from __future__ import annotations

import math
import pathlib

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import CharTokenizer

BLOCK_SIZE = 8
BATCH_SIZE = 32
STEPS = 3000
EVAL_INTERVAL = 500
EVAL_BATCHES = 100
LEARNING_RATE = 1e-2
SEED = 1337


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_text() -> str:
    path = pathlib.Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not path.exists():
        raise SystemExit(
            "data/tinyshakespeare.txt is missing. Fetch it with:\n"
            "  curl -sSfL -o data/tinyshakespeare.txt \\\n"
            "    https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt"
        )
    return path.read_text(encoding="utf-8")


class BigramLM(nn.Module):
    """One row of logits per token. No context beyond the current character."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_logits(idx)  # (B, T, vocab)
        if targets is None:
            return logits, None
        b, t, v = logits.shape
        loss = F.cross_entropy(logits.view(b * t, v), targets.view(b * t))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -1:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def get_batch(data: torch.Tensor, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(dev), y.to(dev)


@torch.no_grad()
def estimate_loss(model: nn.Module, splits: dict, dev: torch.device) -> dict[str, float]:
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            x, y = get_batch(data, dev)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    torch.manual_seed(SEED)
    dev = device()
    text = load_text()
    tok = CharTokenizer(text)

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    model = BigramLM(tok.vocab_size).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    baseline = math.log(tok.vocab_size)
    print(f"device                {dev}")
    print(f"vocab size            {tok.vocab_size}")
    print(f"parameters            {sum(p.numel() for p in model.parameters()):,}")
    print(f"uniform-guess loss    {baseline:.4f}   (ln of vocab size)")
    print()

    for step in range(STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, splits, dev)
            print(
                f"step {step:>5}  train {losses['train']:.4f}  val {losses['val']:.4f}"
            )
        x, y = get_batch(splits["train"], dev)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final = estimate_loss(model, splits, dev)
    print()
    print(f"improvement over uniform  {baseline - final['val']:.4f} nats")
    print()
    print("--- sample ---")
    start = torch.zeros((1, 1), dtype=torch.long, device=dev)
    print(tok.decode(model.generate(start, 300)[0].tolist()))


if __name__ == "__main__":
    main()
