---
title: Bigram Model
slug: bigram-model
description: The smallest model that generates text, and the loss floor every later page has to beat.
type: wiki
status: draft
topics: [bigram, baseline, cross-entropy]
published: false
newsletter: false
created: 2026-08-30
updated: 2026-08-30
source_notes: []
implementation: src/bigram.py
---

# Concept

A bigram model predicts the next token from exactly one thing: the current
token. That is the whole model. It is a lookup table with one row of logits per
token in the vocabulary, trained by cross-entropy like anything else.

It matters because it is the baseline. Every architectural idea in this wiki has
to justify itself against a number this model already achieves, and several
ideas justify themselves by less than people expect.

# Minimal implementation

The entire model is one embedding layer whose output width happens to equal the
vocabulary size, so a lookup *is* the prediction:

```python
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        return self.token_logits(idx)
```

For a 65-character vocabulary that is 65 x 65 = **4,225 parameters**. Full code,
including training loop and sampling, is in `src/bigram.py`.

Note what is absent: no position information, no context beyond one token, no
attention, no hidden layer. Positions are fed to the model in blocks of 8 only
so that one forward pass produces 8 training signals, not because the model can
see across them.

# What breaks without it

Inverted here, because the bigram *is* the ablation. The question is what it
buys over a model that has learned nothing at all.

A uniform guess over 65 characters costs ln(65) = **4.1744** nats. Measured on
TinyShakespeare, character-level, 3000 steps:

| Model | Val loss |
|---|---|
| Uniform guess | 4.1744 |
| Bigram at init | 4.7311 |
| Bigram trained | 2.4880 |

The gain over uniform is **1.6776 nats**, and it converges within roughly 1500
steps. Training takes about 10 seconds on MPS.

The init loss being *worse* than uniform (4.73 vs 4.17) is worth pausing on: the
randomly initialized embedding is not uniform, it is confidently wrong in
arbitrary directions. Cross-entropy punishes confident errors harder than
ignorance.

What the samples show:

```
Thaysnthereliovithithandin.
ANow shtoulouel thim.
Wid n tur g'se se ailablord devorenoverd, a d
```

Character-level structure is real: capitalization after newlines, plausible
letter pairs, speaker-name shapes in caps, apostrophes in roughly the right
places. Words are not. One token of context is enough to learn orthography and
nowhere near enough to learn vocabulary.

That gap is the thing every later page exists to close.

```bash
uv run python src/bigram.py
```

# Claims

**Fact:** Trained bigram reaches 2.4880 val loss against a 4.1744 uniform
baseline on TinyShakespeare at character level. Measured locally on 2026-08-30,
seed 1337, MPS, 3000 steps. Reproducible from a clean checkout.

**Fact:** The model has 4,225 parameters, which is the entire 65 x 65 lookup
table.

**Inference:** Convergence by roughly step 1500 suggests the model saturates
what one token of context can express, rather than being limited by optimization.
Testable by training 10x longer and confirming the val loss does not move.

**Question:** How much of the remaining 2.49 nats is reducible by context at all,
versus irreducible entropy in the text? An n-gram sweep would bound this before
any transformer is introduced.

**Question:** The init loss exceeds uniform by 0.56 nats. Does that gap scale
with vocabulary size, and does it matter for larger models' warmup behavior?

# Sources

See `references/sources.md`. The structure of this page follows Karpathy's
`makemore` and Zero to Hero progression; that row is not yet verified at source.
