---
title: Post-Training
slug: post-training
description: What separates a model that continues text from one that answers a question.
type: wiki
status: skeleton
topics: [sft, rlhf, dpo, alignment]
published: false
newsletter: false
created: 2026-08-30
updated: 2026-08-30
source_notes: []
implementation: src/posttrain.py
---

# Concept

Everything up to page 08 produces a model that continues text. Turning that into a model that follows instructions is a separate phase: supervised fine-tuning on demonstration pairs, then preference optimization to rank responses. This page is the most likely to be built from reading rather than running, and any claim that comes from reading must be labeled as such.

# Minimal implementation

Not written yet. Target file: `src/posttrain.py`.

# What breaks without it

The ablation this page must run:

Fine-tune the page 06 model on a small instruction-shaped dataset and compare completions on held-out prompts against the base model. At this scale expect the format to shift while capability does not, and record that honestly rather than overclaiming.

Record the val loss with and without, on the same seed and step count, so the
number is comparable to [the bigram baseline](02-bigram.md) of 2.4880.

# Claims

Nothing measured yet. Do not write claims here before running the ablation above.

**Question:** What is the smallest model where instruction tuning produces a visible behavior change rather than only a format change?

**Question:** How much of the observed change is capability and how much is style? What would distinguish them?

# Sources

See `references/sources.md`.
