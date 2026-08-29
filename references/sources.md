# Source Ledger

Check the source before relying on a claim that affects an implementation
decision. Record the date you last verified it.

| Source | Used by | Verified |
|---|---|---|
| Vaswani et al., *Attention Is All You Need* (arXiv:1706.03762) | 04, 05 | not yet checked |
| Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2) | 06 | not yet checked |
| Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (BPE, arXiv:1508.07909) | 01 | not yet checked |
| Karpathy, `nanoGPT` and *Neural Networks: Zero to Hero* | structure of the whole wiki | not yet checked |
| PyTorch MPS backend documentation | running the code | not yet checked |

Rows start as "not yet checked" deliberately. A row is only dated once the
claim built on it has actually been read at the source, not recalled.

## Own measurements

Runs recorded in `experiments/` count as Fact for this repo, provided the run is
reproducible from a clean checkout. Record the command, the hardware, and the
numbers.

| Run | Command | Hardware | Result |
|---|---|---|---|
| [Tokenizer compression](../experiments/2026-08-30-tokenizer-compression.md) | `uv run python src/tokenizer.py` | arm64 CPU | 1.96x fewer positions, BPE 256 merges vs char |
| [Bigram baseline](../experiments/2026-08-30-bigram-baseline.md) | `uv run python src/bigram.py` | arm64 MPS, torch 2.13.0 | val loss 2.4880 vs 4.1744 uniform |
