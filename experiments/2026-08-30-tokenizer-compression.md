# Tokenizer compression, char vs BPE

**Date:** 2026-08-30
**Command:** `uv run python src/tokenizer.py`
**Hardware:** Apple silicon (arm64), CPU only
**Corpus:** TinyShakespeare, 1,115,394 characters

## Result

| Tokenizer | Vocab | Positions |
|---|---|---|
| Character | 65 | 1,115,394 |
| BPE, 256 merges | 512 | 568,210 |

Compression 1.96x. Implied attention cost ratio 3.85x.

Trainer wall clock: about 36 seconds for 256 merges.

## Notes

First run used a synthetic fallback corpus built by repeating one passage 40
times. It reported 990x compression, because BPE merged entire repeated blocks
into single tokens. That number is meaningless and the fallback was removed
rather than fixed: a tokenizer benchmark on repeated filler measures the filler.

Worth repeating on a second corpus before the 1.96x figure is cited anywhere
outside this repo.
