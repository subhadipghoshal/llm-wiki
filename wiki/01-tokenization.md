---
title: Tokenization
slug: tokenization
description: How text becomes integers, and why the choice of unit sets a ceiling on everything above it.
type: wiki
status: draft
topics: [tokenization, bpe]
published: false
newsletter: false
created: 2026-08-30
updated: 2026-08-30
source_notes: []
implementation: src/tokenizer.py
---

# Concept

A language model never sees text. It sees a sequence of integers, and
tokenization is the function that produces them. The choice of unit sets the
sequence length, the vocabulary size, and therefore the shape of the embedding
table and the cost of attention.

Character-level tokenization gives a tiny vocabulary and very long sequences.
Word-level gives short sequences and a vocabulary that cannot cover the tail.
Byte-pair encoding sits between them: start from bytes, repeatedly merge the
most frequent adjacent pair, stop at a target vocabulary size.

# Minimal implementation

`src/tokenizer.py` contains both. The character tokenizer is the honest
starting point because it has no training step at all:

```python
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
```

The BPE trainer in the same file is the smallest version that shows the real
mechanism: count adjacent pairs, merge the most frequent, repeat.

# What breaks without it

Nothing "breaks" if you skip straight to characters. That is the point worth
recording: character-level works, trains, and generates. What you lose is
sequence length.

Measured on TinyShakespeare (1,115,394 characters), with 256 merges:

| Tokenizer | Vocab | Positions |
|---|---|---|
| Character | 65 | 1,115,394 |
| BPE | 512 | 568,210 |

That is **1.96x** fewer positions. Since attention is quadratic in sequence
length, the compute saving is roughly **3.85x**.

I guessed four to five times before running it. The guess was wrong at this
vocabulary size, and the gap is the useful part: most intuitions about
tokenization are borrowed from production vocabularies of 30k to 100k tokens,
not the 512 used here. Reproduce with:

```bash
uv run python src/tokenizer.py
```

Second observation, unplanned: the naive trainer takes about 36 seconds for 256
merges, because each merge rescans the entire sequence. Real implementations
keep an incremental pair index. The slowness is not incidental to the algorithm,
it is what the algorithm costs when written the obvious way.

# Claims

**Fact:** On TinyShakespeare, 256 BPE merges cut sequence length from 1,115,394
to 568,210 positions, a 1.96x reduction. Measured locally on 2026-08-30 with
`src/tokenizer.py`; reproducible from a clean checkout.

**Fact:** Byte-pair encoding was introduced for machine translation by Sennrich
et al. and adapted to language modeling by the GPT line. Not yet read at the
source, so this is currently secondhand.

**Inference:** Compression ratio should keep climbing with vocabulary size, which
would explain why the commonly repeated figure of roughly 4x assumes a
production-scale vocabulary. Untested here: it needs a sweep over merge counts.

**Inference:** For a wiki whose goal is understanding rather than throughput,
character-level tokenization is the better default through page 06. It removes a
moving part and nothing above it depends on the choice.

**Question:** How much of the 1.96x is specific to TinyShakespeare's small and
repetitive vocabulary? Needs a second corpus before this number is quoted
anywhere else.

**Question:** At what vocabulary size does the embedding table start costing more
than the sequence length saves?

# Sources

See `references/sources.md`, rows for Sennrich et al. and the GPT-2 paper.
Neither has been read at the source yet; both are currently unverified.
