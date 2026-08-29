---
title: Training Dynamics
slug: training-dynamics
description: Learning rate, initialization, batch size, and reading a loss curve honestly.
type: wiki
status: skeleton
topics: [optimization, learning-rate, initialization]
published: false
newsletter: false
created: 2026-08-30
updated: 2026-08-30
source_notes: []
implementation: src/train.py
---

# Concept

The architecture is settled by page 06, so everything left is optimization. This page covers learning rate and its schedule, initialization scale, batch size versus gradient noise, and the difference between a model that has converged and one that has stopped improving for a different reason.

# Minimal implementation

Not written yet. Target file: `src/train.py`.

# What breaks without it

The ablation this page must run:

Sweep learning rate across several orders of magnitude and plot the resulting curves together. The failure shapes are the lesson: divergence, plateau, and the slow-but-still-descending case that is easy to mistake for convergence.

Record the val loss with and without, on the same seed and step count, so the
number is comparable to [the bigram baseline](02-bigram.md) of 2.4880.

# Claims

Nothing measured yet. Do not write claims here before running the ablation above.

**Question:** Does the bigram's apparent convergence at step 1500 survive a 10x longer run, or was it a learning rate artifact?

**Question:** What is the largest batch size that still improves wall-clock time to a target loss on MPS?

# Sources

See `references/sources.md`.
