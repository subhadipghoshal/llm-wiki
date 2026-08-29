# LLM Wiki

Start with [tokenization](wiki/01-tokenization.md), then [the bigram model](wiki/02-bigram.md). That first pass takes about 20 minutes and ends with a model that generates text.

This is a build-it-to-understand-it wiki about language models. Every page pairs an explanation with the smallest implementation that actually runs, and records what breaks when you remove the piece being explained.

## Core idea

You do not understand a component until you have watched the model get worse without it. Each page therefore carries an ablation, not just a description.

## Read in this order

1. [Tokenization](wiki/01-tokenization.md)
2. [Bigram model](wiki/02-bigram.md)
3. [Embeddings and MLP](wiki/03-embeddings-mlp.md)
4. [Self-attention](wiki/04-attention.md)
5. [Multi-head and the transformer block](wiki/05-transformer-block.md)
6. [Full GPT](wiki/06-gpt.md)
7. [Training dynamics](wiki/07-training-dynamics.md)
8. [Inference and the KV cache](wiki/08-inference-kv-cache.md)
9. [Post-training](wiki/09-post-training.md)

Pages 3 through 9 are skeletons. They carry the question each page must answer; the prose and code get written as the work happens.

## Evidence labels

Borrowed from the maintenance discipline in my Docusign knowledge base, because
ML writing mixes published results, folklore, and personal measurement more
freely than almost any other technical subject.

**Fact:** supported by a linked paper, official documentation, or a run in
`experiments/` that can be reproduced from this repo.

**Inference:** a conclusion drawn from behavior or architecture. Reasonable, not
established. Say so.

**Question:** a gap worth closing. Leaving these visible is the point.

An unlabeled claim is a bug. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Running the code

```bash
uv run python src/bigram.py
```

Targets the MPS backend on Apple silicon, falls back to CPU. Everything here is
sized to train on a laptop in seconds to minutes. If a page needs a GPU cluster
to demonstrate its point, the page is wrong.

## Layout

```text
wiki/          One concept per page, prose plus minimal code
src/           Runnable reference implementations
experiments/   Logged runs and what they actually showed
references/    Source ledger
```
