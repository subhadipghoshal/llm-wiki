---
title: Self-Attention
slug: attention
description: The single idea the whole architecture is named after: letting each position choose what to read.
type: wiki
status: skeleton
topics: [attention, softmax]
published: false
newsletter: false
created: 2026-08-30
updated: 2026-08-30
source_notes: []
implementation: src/attention.py
---

# Concept

Fixed-width context forces every position to look at exactly the same neighbours. Self-attention replaces that with a learned, content-dependent weighting: each position emits a query, every position emits a key and a value, and the weighted average of values becomes the output. Causal masking keeps position t from reading the future.

# Minimal implementation

Not written yet. Target file: `src/attention.py`.

# What breaks without it

The ablation this page must run:

Two ablations, both cheap. First, replace attention weights with a uniform average over previous positions, which isolates how much comes from *choosing* versus merely *aggregating*. Second, remove the causal mask and confirm the val loss collapses implausibly low, which is what label leakage looks like.

Record the val loss with and without, on the same seed and step count, so the
number is comparable to [the bigram baseline](02-bigram.md) of 2.4880.

# Claims

Nothing measured yet. Do not write claims here before running the ablation above.

**Question:** How much of attention's benefit is the content-dependent weighting versus simply having a longer receptive field?

**Question:** At what block size does the quadratic cost stop being worth the loss improvement on this corpus?

# Sources

See `references/sources.md`.
