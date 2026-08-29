---
title: Assembling a Full GPT
slug: gpt
description: Stacking blocks into a working model, and the positional information the stack cannot do without.
type: wiki
status: skeleton
topics: [gpt, positional-encoding]
published: false
newsletter: false
created: 2026-08-30
updated: 2026-08-30
source_notes: []
implementation: src/gpt.py
---

# Concept

A GPT is token embeddings plus positional embeddings, a stack of transformer blocks, a final layernorm, and a projection back to vocabulary logits. Nothing new is introduced here; the page exists to show that the assembly is genuinely all that remains.

# Minimal implementation

Not written yet. Target file: `src/gpt.py`.

# What breaks without it

The ablation this page must run:

Remove positional embeddings entirely. Attention is permutation-invariant without them, so the model should lose word order while keeping character statistics. The sample text is the interesting artifact here, more than the loss number.

Record the val loss with and without, on the same seed and step count, so the
number is comparable to [the bigram baseline](02-bigram.md) of 2.4880.

# Claims

Nothing measured yet. Do not write claims here before running the ablation above.

**Question:** How far does val loss fall below the bigram's 2.4880 at a size that still trains in minutes on this laptop?

**Question:** How much does weight tying between the input embedding and the output projection save, and does it cost anything?

# Sources

See `references/sources.md`.
