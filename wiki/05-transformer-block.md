---
title: Multi-Head Attention and the Transformer Block
slug: transformer-block
description: Several attention heads in parallel, plus the residual and normalization scaffolding that makes depth trainable.
type: wiki
status: skeleton
topics: [multi-head, residual, layernorm]
published: false
newsletter: false
created: 2026-08-30
updated: 2026-08-30
source_notes: []
implementation: src/transformer.py
---

# Concept

One attention head learns one kind of relationship. Multiple heads run in parallel over lower-dimensional subspaces and concatenate. The block adds the parts that make stacking possible: residual connections so gradients reach early layers, layer normalization so activations stay in range, and a position-wise feedforward layer that does most of the actual computation.

# Minimal implementation

Not written yet. Target file: `src/transformer.py`.

# What breaks without it

The ablation this page must run:

Ablate each scaffolding piece separately at 4 and 6 layers deep: no residuals, no layernorm, pre-norm versus post-norm. The residual ablation should fail to train at depth; record where it stops converging rather than just noting that it does.

Record the val loss with and without, on the same seed and step count, so the
number is comparable to [the bigram baseline](02-bigram.md) of 2.4880.

# Claims

Nothing measured yet. Do not write claims here before running the ablation above.

**Question:** Do more heads or wider heads help more, at fixed total width?

**Question:** Where exactly does the no-residual model stop training as depth increases?

# Sources

See `references/sources.md`.
