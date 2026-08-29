---
title: Embeddings and MLP
slug: embeddings-mlp
description: Giving tokens a learned vector instead of a lookup row, and a hidden layer to mix them.
type: wiki
status: skeleton
topics: [embeddings, mlp]
published: false
newsletter: false
created: 2026-08-30
updated: 2026-08-30
source_notes: []
implementation: src/mlp.py
---

# Concept

The bigram's lookup table ties the representation of a token to its prediction. Splitting them means a token gets an embedding vector, several previous embeddings get concatenated, and a hidden layer maps that to logits. This is the first model with a bottleneck, and therefore the first that can generalize between similar tokens.

# Minimal implementation

Not written yet. Target file: `src/mlp.py`.

# What breaks without it

The ablation this page must run:

Remove the hidden layer, leaving a linear map from concatenated embeddings to logits. If the nonlinearity earns nothing here, say so plainly.

Record the val loss with and without, on the same seed and step count, so the
number is comparable to [the bigram baseline](02-bigram.md) of 2.4880.

# Claims

Nothing measured yet. Do not write claims here before running the ablation above.

**Question:** How much of the gain over bigram comes from more context versus from the learned embedding bottleneck? Vary context length at fixed embedding size to separate them.

**Question:** What embedding dimension stops helping on a 65-token vocabulary?

# Sources

See `references/sources.md`.
