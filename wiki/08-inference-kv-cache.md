---
title: Inference and the KV Cache
slug: inference-kv-cache
description: Why naive generation is quadratic, and the caching that makes it linear.
type: wiki
status: skeleton
topics: [inference, kv-cache, sampling]
published: false
newsletter: false
created: 2026-08-30
updated: 2026-08-30
source_notes: []
implementation: src/sample.py
---

# Concept

Training and generation have different cost structures. Generating token t naively recomputes attention over the entire prefix, repeating work done for every earlier token. Caching keys and values makes each new token cost one step against a growing cache. This page also covers the sampling decisions, temperature and top-k, that shape output as much as the weights do.

# Minimal implementation

Not written yet. Target file: `src/sample.py`.

# What breaks without it

The ablation this page must run:

Measure tokens per second with and without the cache at several sequence lengths, and confirm the outputs are identical under a fixed seed. Identical output is the correctness check; the speed difference is the point.

Record the val loss with and without, on the same seed and step count, so the
number is comparable to [the bigram baseline](02-bigram.md) of 2.4880.

# Claims

Nothing measured yet. Do not write claims here before running the ablation above.

**Question:** At what sequence length does the cache start paying for its memory cost on this hardware?

**Question:** How much does temperature change measured val loss versus merely change how the samples read?

# Sources

See `references/sources.md`.
