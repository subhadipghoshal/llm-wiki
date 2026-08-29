# Bigram baseline

**Date:** 2026-08-30
**Command:** `uv run python src/bigram.py`
**Hardware:** Apple silicon (arm64), MPS backend, torch 2.13.0
**Corpus:** TinyShakespeare, character level, 90/10 train/val split
**Config:** seed 1337, block 8, batch 32, lr 1e-2, AdamW, 3000 steps

## Result

| Checkpoint | Train | Val |
|---|---|---|
| Uniform guess, ln(65) | | 4.1744 |
| Step 0 | 4.7252 | 4.7311 |
| Step 500 | 2.5912 | 2.6101 |
| Step 1000 | 2.4740 | 2.5197 |
| Step 1500 | 2.4654 | 2.4928 |
| Step 2000 | 2.4611 | 2.4755 |
| Step 3000 | 2.4685 | 2.4880 |

Improvement over uniform: 1.6776 nats. Parameters: 4,225. Wall clock: about 10
seconds.

## Notes

Val loss is flat from step 1500 onward and step 3000 is marginally worse than
step 2000, which is noise at this eval size (100 batches), not overfitting.

Init loss of 4.7311 is worse than the 4.1744 uniform baseline. A random
embedding is not uniform, it is confidently wrong, and cross-entropy charges
more for that than for ignorance.

**This is the number every later page must beat: val 2.4880.**
